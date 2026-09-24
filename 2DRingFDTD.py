"""Educational 2D FDTD simulation of a circular ring transducer.

Continuous model
----------------
Pressure u(x, y, t) obeys the scalar wave equation

    d^2 u / dt^2 = c(x, y)^2 (u_xx + u_yy)

Density is constant, so the only medium contrast is in sound speed. The FDTD
stepper does not depend on this choice: later models (variable density,
attenuation, elastic waves) can replace ScalarWave2D without changing the
ring array, chirp, or time loop.

A ring of point elements lies on a circle. Each element is both a transmitter
and a receiver. One element radiates a short Hann-windowed linear chirp; the
other 255 record the resulting pressure. Optional sound-speed phantoms
(a disk, a modified Shepp–Logan head, or an abdominal CT map produced by
ct_to_speed.py) sit inside the ring. CT maps are isotropically downsampled,
centered, and padded with coupling water on a selectable 256 x 256 or
512 x 512 grid.

Discretization
--------------
A uniform Cartesian grid and a second-order leapfrog scheme:

    u^{n+1} = 2 u^n - u^{n-1} + dt^2 c^2 Laplacian(u^n)

2D CFL stability requires c_max dt sqrt(1/dx^2 + 1/dy^2) < 1. Domain edges
use first-order Mur absorbing boundary conditions.

The forward simulation uses PyTorch tensor kernels and can run on CPU, CUDA,
or Apple Metal (MPS). The time steps are sequential, while every spatial
update and receiver gather is parallelized across the selected device.

This is an educational model, not a validated ultrasound imaging chain.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from fdtd2d import (
    RingArray,
    axis_centers,
    load_ct_ring_grid,
    n_steps_for_crossing,
    stable_dt,
)
from fdtd2d.phantoms import SHEPP_DC, shepp_logan_speed
from fdtd2d.plot import animate_field, plot_dashboard
from fdtd2d.torch_backend import (
    TorchFDTD2D,
    TorchRingSampler,
    TorchScalarWave2D,
    configure_runtime,
    device_summary,
    linear_chirp,
    resolve_device,
    simulate_shot,
)


# Domain. Lengths in meters; time in seconds.
NX = 512
NY = 512
DX = 0.50e-3
DY = DX
C0 = 1500.0
CFL = 0.45
RING_RADIUS = 0.070
N_ELEMENTS = 256

# Short linear chirp.
F_START_HZ = 1.00e5
F_END_HZ = 2.50e5
CHIRP_DURATION = 20.0e-6

# Centered disk phantom (faster than the background water-like medium).
PHANTOM_RADIUS = 0.020
PHANTOM_C = 1800.0

SNAPSHOT_STRIDE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D FDTD ring transducer: one chirp shot, 255 receivers",
    )
    parser.add_argument(
        "--tx", type=int, default=0, metavar="I",
        help="Transmitter element index (default: 0)",
    )
    parser.add_argument(
        "--n-elements", type=int, default=N_ELEMENTS, metavar="N",
        help=f"Number of ring elements (default: {N_ELEMENTS})",
    )
    medium = parser.add_mutually_exclusive_group()
    medium.add_argument(
        "--phantom",
        choices=("disk", "shepp-logan", "none"),
        default="disk",
        help="Sound-speed phantom at the ring center (default: disk)",
    )
    medium.add_argument(
        "--ct-speed",
        type=Path,
        metavar="PATH",
        help="Load the .npz sound-speed map produced by ct_to_speed.py",
    )
    parser.add_argument(
        "--ring-clearance-mm",
        type=float,
        default=10.0,
        metavar="MM",
        help="CT body-to-ring clearance (default: 10 mm)",
    )
    parser.add_argument(
        "--ct-padding-speed",
        type=float,
        default=1480.0,
        metavar="M_S",
        help="Coupling-medium speed outside the CT body (default: 1480 m/s)",
    )
    parser.add_argument(
        "--ct-edge-margin",
        type=int,
        default=8,
        metavar="PIXELS",
        help="Grid boundary margin outside the CT ring (default: 8 pixels)",
    )
    parser.add_argument(
        "--ct-grid-size",
        type=int,
        choices=(256, 512),
        default=512,
        metavar="N",
        help="Downsample and pad a CT map to N x N cells (default: 512)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="PyTorch device: auto, cpu, cuda, cuda:N, or mps (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Simulation precision (default: float32)",
    )
    parser.add_argument(
        "--compile",
        choices=("auto", "on", "off"),
        default="auto",
        help="Compile the FDTD step; auto enables it on CUDA (default: auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        metavar="N",
        help="PyTorch CPU threads; zero uses the runtime default (default: 0)",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow faster reduced-mantissa TensorFloat-32 kernels on CUDA",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Animate the wave field in a separate window before the traces",
    )
    parser.add_argument(
        "--interval", type=int, default=40, metavar="MS",
        help="Animation frame delay in milliseconds (default: 40)",
    )
    parser.add_argument(
        "--save-traces", type=Path, metavar="PATH",
        help="Save time series to a .npz archive",
    )
    parser.add_argument(
        "--save-figure", type=Path, metavar="PATH",
        help="Save the results dashboard as an image",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Run without opening windows",
    )
    return parser.parse_args()


def build_medium(x_m, y_m, phantom="disk"):
    """Return c(x, y) in m/s."""
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full_like(x, C0)
    if phantom == "disk":
        c[x**2 + y**2 <= PHANTOM_RADIUS**2] = PHANTOM_C
    elif phantom == "shepp-logan":
        c = shepp_logan_speed(x, y, RING_RADIUS, C0)
    return c


def print_results(
    array,
    solver,
    traces,
    tx,
    dt,
    n_steps,
    pulse,
    phantom="disk",
    ct_grid=None,
    propagation_speed=C0,
    execution_device="CPU",
    dtype_name="float32",
    compiled=False,
    simulation_time=None,
):
    opposite = (tx + array.n_elements // 2) % array.n_elements
    peak_rx = np.max(np.abs(np.delete(traces, tx, axis=0)))
    arrival = dt * int(np.argmax(np.abs(traces[opposite])))
    expected = 2.0 * array.radius_m / propagation_speed
    print("2D ring FDTD results")
    print(f"  Elements:                        {array.n_elements}")
    print(f"  Transmitter index:               {tx}")
    if ct_grid is not None:
        print("  Phantom:                         abdominal CT speed map")
        print(
            f"  Source CT grid:                  "
            f"{ct_grid.source_shape[1]} x {ct_grid.source_shape[0]}"
        )
        print(
            f"  Source spacing:                  "
            f"{ct_grid.source_spacing_mm[1]:.4g} x "
            f"{ct_grid.source_spacing_mm[0]:.4g} mm"
        )
        print(
            f"  Ring radius / clearance:         "
            f"{1e3 * array.radius_m:.2f} / "
            f"{1e3 * (array.radius_m - body_radius(ct_grid)):.2f} mm"
        )
    elif phantom == "disk":
        print(f"  Phantom:                         disk, r = {1e3 * PHANTOM_RADIUS:.1f} mm, "
              f"c = {PHANTOM_C:.0f} m/s  (background {C0:.0f} m/s)")
    elif phantom == "shepp-logan":
        print(f"  Phantom:                         Shepp–Logan  "
              f"(c = {C0:.0f}–{C0 + SHEPP_DC:.0f} m/s)")
    else:
        print(f"  Phantom:                         none  (c = {C0:.0f} m/s)")
    print(f"  Grid:                            {solver.u.shape[1]} x {solver.u.shape[0]}")
    print(f"  PyTorch device:                  {execution_device}")
    print(f"  Precision / compiled step:       {dtype_name} / {compiled}")
    print(
        f"  dx, dy, dt:                      {solver.model.dx:.4g}, "
        f"{solver.model.dy:.4g} m, {dt:.4g} s"
    )
    print(f"  CFL:                             {solver.cfl:.3f}")
    print(f"  Time steps:                      {n_steps}")
    print(f"  Chirp samples:                   {pulse.size}")
    if simulation_time is not None:
        print(f"  Simulation time:                 {simulation_time:.3f} s")
    print(f"  Peak received amplitude:         {peak_rx:.4g}")
    print(f"  Opposite-element peak time:      {1e6 * arrival:.2f} µs")
    print(f"  Geometric diameter travel time:  {1e6 * expected:.2f} µs")


def body_radius(ct_grid):
    """Maximum transformed body radius in meters."""
    rows, cols = np.nonzero(ct_grid.body_mask)
    center = 0.5 * (ct_grid.body_mask.shape[0] - 1)
    return ct_grid.spacing_m * float(
        np.max(np.hypot(rows - center, cols - center))
    )


def save_traces(
    path,
    traces,
    time_s,
    array,
    tx,
    pulse,
    dt,
    c_map=None,
    device="cpu",
    dtype="float32",
    compiled=False,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        traces=traces,
        time_s=time_s,
        pulse=pulse,
        dt=dt,
        tx=tx,
        theta=array.theta,
        x=array.x,
        y=array.y,
        radius_m=array.radius_m,
        sound_speed_m_s=c_map,
        torch_device=np.asarray(device),
        torch_dtype=np.asarray(dtype),
        torch_compiled=np.asarray(bool(compiled)),
    )


def main():
    args = parse_args()
    if args.interval < 1:
        raise SystemExit("--interval must be at least 1 ms")
    if args.n_elements < 2:
        raise SystemExit("Need at least 2 ring elements.")
    if args.ring_clearance_mm <= 0:
        raise SystemExit("--ring-clearance-mm must be positive.")
    if args.ct_padding_speed <= 0:
        raise SystemExit("--ct-padding-speed must be positive.")
    if args.cpu_threads < 0:
        raise SystemExit("--cpu-threads must be zero or positive.")
    try:
        device = resolve_device(args.device)
        configure_runtime(
            device,
            cpu_threads=args.cpu_threads,
            allow_tf32=args.allow_tf32,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not configure PyTorch: {exc}") from exc
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise SystemExit("Apple MPS does not support this simulation in float64.")
    compile_step = args.compile == "on" or (
        args.compile == "auto" and device.type == "cuda"
    )

    ct_grid = None
    if args.ct_speed is not None:
        try:
            ct_grid = load_ct_ring_grid(
                args.ct_speed,
                grid_shape=(args.ct_grid_size, args.ct_grid_size),
                padding_speed_m_s=args.ct_padding_speed,
                ring_clearance_mm=args.ring_clearance_mm,
                edge_margin_pixels=args.ct_edge_margin,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Could not prepare CT speed map: {exc}") from exc
        c = ct_grid.c
        x_m = ct_grid.x_m
        y_m = ct_grid.y_m
        dx = dy = ct_grid.spacing_m
        ring_radius = ct_grid.ring_radius_m
    else:
        dx, dy = DX, DY
        x_m = axis_centers(NX, dx)
        y_m = axis_centers(NY, dy)
        c = build_medium(x_m, y_m, phantom=args.phantom)
        ring_radius = RING_RADIUS

    model = TorchScalarWave2D(c, dx, dy, device=device, dtype=dtype)
    dt = stable_dt(model.c_max, dx, dy, cfl=CFL)
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    array = RingArray(args.n_elements, ring_radius, x_m, y_m)
    sampler = TorchRingSampler(array, device=device, dtype=dtype)
    tx = array.transmitter_index(args.tx)
    pulse_device = linear_chirp(
        dt,
        CHIRP_DURATION,
        F_START_HZ,
        F_END_HZ,
        device=device,
        dtype=dtype,
    )
    bulk_speeds = c[c >= 1000.0]
    travel_speed = float(bulk_speeds.min()) if bulk_speeds.size else float(c.min())
    n_steps = n_steps_for_crossing(ring_radius, travel_speed, dt)
    n_steps = max(n_steps, pulse_device.numel() + 1)
    snapshot_stride = SNAPSHOT_STRIDE if args.animate and not args.no_show else 0

    source_row, source_col = array.inject_rows_cols(tx)
    t0 = time.perf_counter()
    traces_device, snapshots = simulate_shot(
        solver,
        sampler,
        pulse_device,
        source_row,
        source_col,
        n_steps,
        snapshot_stride=snapshot_stride,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    simulation_time = time.perf_counter() - t0
    traces = traces_device.detach().cpu().numpy()
    final_field = solver.u.detach().cpu().numpy()
    pulse = pulse_device.detach().cpu().numpy()
    time_s = np.arange(n_steps) * dt
    execution_device = device_summary(device)
    print_results(
        array,
        solver,
        traces,
        tx,
        dt,
        n_steps,
        pulse,
        phantom=args.phantom,
        ct_grid=ct_grid,
        propagation_speed=(args.ct_padding_speed if ct_grid is not None else C0),
        execution_device=execution_device,
        dtype_name=args.dtype,
        compiled=compile_step,
        simulation_time=simulation_time,
    )

    if args.save_traces:
        save_traces(
            args.save_traces,
            traces,
            time_s,
            array,
            tx,
            pulse,
            dt,
            c_map=c,
            device=str(device),
            dtype=args.dtype,
            compiled=compile_step,
        )

    c_plot = c if ct_grid is not None or args.phantom != "none" else None
    if args.animate and not args.no_show:
        if not snapshots:
            raise SystemExit("Animation requested but no snapshots were stored.")
        anim_fig = animate_field(
            snapshots, array, tx, x_m, y_m, dt,
            interval=args.interval, c_map=c_plot,
        )
        plt.show(block=True)
        plt.close(anim_fig)

    if not args.no_show or args.save_figure:
        figure = plot_dashboard(
            final_field, traces, time_s, array, pulse, dt, tx, x_m, y_m,
            c_map=c_plot,
        )
        if args.save_figure:
            args.save_figure.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save_figure, dpi=180)
        if not args.no_show:
            plt.show(block=True)
        else:
            plt.close(figure)


if __name__ == "__main__":
    main()
