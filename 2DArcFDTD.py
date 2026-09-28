"""2D FDTD simulation of a focused transmit arc and an opposing receive arc.

Geometry
--------
Two curved (concave) arrays sit on a circle of radius R about the origin:

* a transmit arc of 64 elements spanning ~30 degrees at the bottom (270 deg),
* a receive arc spanning ~30 degrees at the top (90 deg).

Both arcs face the origin, which is the geometric focus of the transmit arc.
All transmit elements fire the same Hann-windowed linear chirp. Element i is
delayed by

    tau_i = (max_j |x_j - F| - |x_i - F|) / c0

so every wavelet reaches the focus F at the same time. With the default
F = (0, 0) the arc is geometrically focused and all delays are zero; moving F
steers and refocuses the beam electronically. ``--tx I`` fires a single
element instead, and ``--each-element`` fires every transmit element on its
own, one shot at a time (batched on the device), giving a full
(transmitter, receiver, time) dataset. The receive arc records pressure with
bilinear sampling.

Physics and numerics match 2DRingFDTD.py: the constant-density scalar wave
equation, a second-order leapfrog scheme, and first-order Mur absorbing
boundaries, run with the PyTorch backend on CPU, CUDA, or Apple Metal (MPS).

This is an educational model, not a validated ultrasound imaging chain.
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from fdtd2d import ArcArray, axis_centers, stable_dt
from fdtd2d.phantoms import shepp_logan_speed
from fdtd2d.torch_backend import (
    TorchFDTD2D,
    TorchFDTD2DBatch,
    TorchRingSampler,
    TorchScalarWave2D,
    configure_runtime,
    device_summary,
    resolve_device,
    simulate_shots_batch,
)


# Domain. Lengths in meters; time in seconds. Same field of view as invert_ring.py.
NX = 401
NY = 401
DX = 0.50e-3
DY = DX
C0 = 1500.0
CFL = 0.45

# Array defaults.
ARC_RADIUS = 0.070
N_TX = 64
N_RX = 64
TX_SPAN_DEG = 30.0
RX_SPAN_DEG = 30.0
TX_CENTER_DEG = 270.0  # bottom
RX_CENTER_DEG = 90.0   # top

# Short linear chirp, as in 2DRingFDTD.py.
F_START_HZ = 1.00e5
F_END_HZ = 2.50e5
CHIRP_DURATION = 20.0e-6

# Centered disk phantom.
PHANTOM_RADIUS = 0.020
PHANTOM_C = 1800.0

# Record long enough for the slowest straight path plus this safety factor.
TIME_MARGIN = 1.3
SNAPSHOT_STRIDE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D FDTD: focused transmit arc (bottom) and receive arc (top)",
    )
    parser.add_argument("--n-tx", type=int, default=N_TX, metavar="N",
                        help=f"Transmit-arc elements (default: {N_TX})")
    parser.add_argument("--n-rx", type=int, default=N_RX, metavar="N",
                        help=f"Receive-arc elements (default: {N_RX})")
    parser.add_argument("--tx-span-deg", type=float, default=TX_SPAN_DEG, metavar="DEG",
                        help="Angle covered by the transmit arc, centered at the bottom; "
                             f"180 is the lower half circle (default: {TX_SPAN_DEG:g})")
    parser.add_argument("--rx-span-deg", type=float, default=RX_SPAN_DEG, metavar="DEG",
                        help="Angle covered by the receive arc, centered at the top "
                             f"(default: {RX_SPAN_DEG:g})")
    parser.add_argument("--radius-mm", type=float, default=1e3 * ARC_RADIUS, metavar="MM",
                        help=f"Radius of curvature of the transmit arc (default: {1e3 * ARC_RADIUS:g})")
    parser.add_argument("--rx-radius-mm", type=float, metavar="MM",
                        help="Radius of the receive arc (default: same as --radius-mm)")
    parser.add_argument("--focus-mm", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("X", "Y"),
                        help="Electronic focus in mm (default: 0 0, the geometric focus)")
    parser.add_argument("--tx", type=int, metavar="I",
                        help="Fire only transmit element I instead of the focused arc; "
                             "with --each-element, the shot to display (default: center)")
    parser.add_argument("--each-element", action="store_true",
                        help="Fire every transmit element individually, one shot at a time")
    parser.add_argument("--parallel-shots", type=int, default=8, metavar="N",
                        help="Single-element shots advanced together with --each-element "
                             "(default: 8)")
    parser.add_argument("--phantom", choices=("disk", "shepp-logan", "none"), default="disk",
                        help="Sound-speed phantom at the center (default: disk)")
    parser.add_argument("--device", default="auto", metavar="DEVICE",
                        help="PyTorch device: auto, cpu, cuda, cuda:N, or mps (default: auto)")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                        help="Simulation precision (default: float32)")
    parser.add_argument("--compile", choices=("auto", "on", "off"), default="auto",
                        help="Compile the FDTD step; auto enables it on CUDA (default: auto)")
    parser.add_argument("--cpu-threads", type=int, default=0, metavar="N",
                        help="PyTorch CPU threads; zero uses the runtime default (default: 0)")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="Allow TensorFloat-32 kernels on CUDA")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Show the array geometry and exit without simulating")
    parser.add_argument("--animate", action="store_true",
                        help="Animate the wave field in a separate window before the dashboard")
    parser.add_argument("--interval", type=int, default=40, metavar="MS",
                        help="Animation frame delay in milliseconds (default: 40)")
    parser.add_argument("--save-traces", type=Path, metavar="PATH",
                        help="Save received traces and geometry to a .npz archive")
    parser.add_argument("--save-figure", type=Path, metavar="PATH",
                        help="Save the dashboard (or geometry figure) as an image")
    parser.add_argument("--no-show", action="store_true",
                        help="Run without opening windows")
    return parser.parse_args()


def build_medium(x_m, y_m, phantom="disk", fit_radius=ARC_RADIUS):
    """Return c(x, y) in m/s."""
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full_like(x, C0)
    if phantom == "disk":
        c[x**2 + y**2 <= PHANTOM_RADIUS**2] = PHANTOM_C
    elif phantom == "shepp-logan":
        c = shepp_logan_speed(x, y, fit_radius, C0)
    return c


def arcs_overlap(tx_array, rx_array, dx):
    """True when two arcs on (nearly) the same circle share angular coverage."""
    if abs(tx_array.radius_m - rx_array.radius_m) >= dx:
        return False
    separation = abs((tx_array.center_deg - rx_array.center_deg + 180.0) % 360.0 - 180.0)
    return separation <= 0.5 * (tx_array.span_deg + rx_array.span_deg)


def chord_aperture(radius_m, span_deg):
    """Straight-line width of an arc; a full diameter once the span reaches 180°."""
    return 2.0 * radius_m * math.sin(math.radians(min(span_deg, 180.0) / 2.0))


def focusing_delays(tx_array, focus, speed, single_tx=None):
    """Per-element firing delays (s) and active-element mask."""
    active = np.ones(tx_array.n_elements, dtype=bool)
    if single_tx is not None:
        active[:] = False
        active[tx_array.transmitter_index(single_tx)] = True
        return np.zeros(tx_array.n_elements), active
    distance = np.hypot(tx_array.x - focus[0], tx_array.y - focus[1])
    return (distance.max() - distance) / speed, active


def delayed_chirps(n_steps, dt, delays, n_pulse):
    """Chirp sampled at t - tau_i for every element: shape (n_elements, n_steps).

    Uses the same waveform as ``linear_chirp`` (duration (n_pulse - 1) dt), but
    evaluated analytically so sub-sample delays are exact.
    """
    duration = (n_pulse - 1) * dt
    sweep = (F_END_HZ - F_START_HZ) / duration
    tau = np.arange(n_steps)[None, :] * dt - delays[:, None]
    inside = (tau >= 0.0) & (tau <= duration)
    phase = 2.0 * np.pi * (F_START_HZ * tau + 0.5 * sweep * tau**2)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * tau / duration)
    return np.where(inside, window * np.sin(phase), 0.0)


def expected_arrivals(tx_array, rx_array, focus, delays, active, speed, pulse_center):
    """Straight-ray water arrival (chirp center) at each receiver, in seconds."""
    if active.sum() == 1:
        i = int(np.flatnonzero(active)[0])
        path = np.hypot(rx_array.x - tx_array.x[i], rx_array.y - tx_array.y[i])
        return path / speed + pulse_center
    to_focus = delays[0] + np.hypot(tx_array.x[0] - focus[0], tx_array.y[0] - focus[1]) / speed
    from_focus = np.hypot(rx_array.x - focus[0], rx_array.y - focus[1]) / speed
    return to_focus + from_focus + pulse_center


def simulate(solver, sampler, tx_array, source_values, n_steps, snapshot_steps):
    """Fire every active transmit element with its own waveform; record the rx arc."""
    model = solver.model
    solver.reset()
    rows = torch.as_tensor(tx_array.rows, device=model.device, dtype=torch.long)
    cols = torch.as_tensor(tx_array.cols, device=model.device, dtype=torch.long)
    values = torch.as_tensor(source_values, device=model.device, dtype=model.dtype)
    active_steps = np.flatnonzero(np.any(source_values != 0.0, axis=0))
    last_source_step = int(active_steps[-1]) if active_steps.size else -1
    source_field = torch.zeros(model.shape, device=model.device, dtype=model.dtype)
    zero = torch.zeros((), device=model.device, dtype=model.dtype)
    traces = torch.empty((sampler.n_elements, n_steps), device=model.device, dtype=model.dtype)
    snapshots = {}
    with torch.inference_mode():
        for step in range(n_steps):
            if step <= last_source_step:
                source_field.zero_()
                # Accumulate so elements that round to one pixel still add up.
                source_field.index_put_((rows, cols), values[:, step], accumulate=True)
                field = solver.inject_field_and_step(source_field, adjoint=False)
            else:
                field = solver.step(zero)
            traces[:, step] = sampler.record(field)
            if step in snapshot_steps:
                snapshots[step] = field.detach().cpu().numpy().copy()
    return traces.detach().cpu().numpy(), snapshots


def simulate_each_element(solver, sampler, tx_array, waveform, n_steps, batch_size):
    """Fire each transmit element alone; return traces (n_tx, n_rx, n_steps)."""
    model = solver.model
    values = torch.as_tensor(waveform, device=model.device, dtype=model.dtype)
    batched_solvers = {}
    gathers = []
    for start in range(0, tx_array.n_elements, batch_size):
        shots = np.arange(start, min(start + batch_size, tx_array.n_elements))
        batched = batched_solvers.get(shots.size)
        if batched is None:
            batched = TorchFDTD2DBatch(solver, shots.size)
            batched_solvers[shots.size] = batched
        traces, _ = simulate_shots_batch(
            solver, sampler, values, tx_array.rows[shots], tx_array.cols[shots], n_steps,
            batched=batched, source_values=values,
        )
        gathers.append(traces.detach().cpu().numpy())
        print(f"    shots {shots[0]:3d}-{shots[-1]:3d} of {tx_array.n_elements} done")
    return np.concatenate(gathers, axis=0)


def extent_mm(x_m, y_m):
    dx, dy = x_m[1] - x_m[0], y_m[1] - y_m[0]
    return [1e3 * (x_m[0] - dx / 2), 1e3 * (x_m[-1] + dx / 2),
            1e3 * (y_m[0] - dy / 2), 1e3 * (y_m[-1] + dy / 2)]


def draw_arrays(ax, tx_array, rx_array, focus, active, legend=True):
    ax.plot(1e3 * tx_array.x[~active], 1e3 * tx_array.y[~active], ".",
            color="lightcoral", ms=3)
    ax.plot(1e3 * tx_array.x[active], 1e3 * tx_array.y[active], ".", color="red", ms=4,
            label=f"Tx arc ({tx_array.n_elements} el, {tx_array.span_deg:g}°)")
    ax.plot(1e3 * rx_array.x, 1e3 * rx_array.y, ".", color="deepskyblue", ms=4,
            label=f"Rx arc ({rx_array.n_elements} el, {rx_array.span_deg:g}°)")
    if active.sum() > 1:
        ax.plot(1e3 * focus[0], 1e3 * focus[1], "*", color="gold", mec="black",
                ms=12, label="Focus")
        for end in (0, -1):
            ax.plot(1e3 * np.array([tx_array.x[end], focus[0]]),
                    1e3 * np.array([tx_array.y[end], focus[1]]),
                    "--", color="red", lw=0.8, alpha=0.7)
    if legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def plot_geometry(ax, c, x_m, y_m, tx_array, rx_array, focus, active):
    im = ax.imshow(c, origin="lower", extent=extent_mm(x_m, y_m), cmap="viridis")
    plt.colorbar(im, ax=ax, label="Sound speed (m/s)")
    draw_arrays(ax, tx_array, rx_array, focus, active)
    ax.set(title="Geometry and sound speed", xlabel="x (mm)", ylabel="y (mm)")
    ax.set_aspect("equal")


def plot_dashboard(c, x_m, y_m, tx_array, rx_array, focus, active, snapshot, snapshot_time,
                   traces, time_s, arrivals, center_rx, all_shots=None, shown_tx=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
    extent = extent_mm(x_m, y_m)

    plot_geometry(axes[0, 0], c, x_m, y_m, tx_array, rx_array, focus, active)

    ax = axes[0, 1]
    vmax = float(np.percentile(np.abs(snapshot), 99.9)) or 1.0
    im = ax.imshow(snapshot, origin="lower", extent=extent, cmap="seismic",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Pressure (arb.)")
    if np.ptp(c) > 0:
        ax.contour(1e3 * x_m, 1e3 * y_m, c, levels=[0.5 * (c.min() + c.max())],
                   colors="black", linewidths=0.7)
    draw_arrays(ax, tx_array, rx_array, focus, active, legend=False)
    ax.set(title=f"Pressure at t = {1e6 * snapshot_time:.1f} µs",
           xlabel="x (mm)", ylabel="y (mm)")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    clip = float(np.percentile(np.abs(traces), 99.5)) or 1.0
    ax.imshow(traces, aspect="auto", origin="lower", cmap="gray", vmin=-clip, vmax=clip,
              extent=[1e6 * time_s[0], 1e6 * time_s[-1], -0.5, rx_array.n_elements - 0.5])
    ax.plot(1e6 * arrivals, np.arange(rx_array.n_elements), "--", color="orange", lw=1,
            label=f"Straight-ray arrival in water ({C0:.0f} m/s)")
    title = "Received traces (receive arc)"
    if shown_tx is not None:
        title += f", Tx element {shown_tx} fired"
    ax.set(title=title, xlabel="Time (µs)", ylabel="Receiver element")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    if all_shots is not None:
        # Receiver at the arc center, one row per individually fired element.
        gather = all_shots[:, center_rx]
        clip = float(np.percentile(np.abs(gather), 99.5)) or 1.0
        ax.imshow(gather, aspect="auto", origin="lower", cmap="gray", vmin=-clip, vmax=clip,
                  extent=[1e6 * time_s[0], 1e6 * time_s[-1], -0.5, tx_array.n_elements - 0.5])
        ax.axhline(shown_tx, color="red", lw=0.8, ls=":", label=f"Tx {shown_tx} (shown left)")
        ax.set(title=f"Receiver {center_rx} (arc center) for every Tx shot",
               xlabel="Time (µs)", ylabel="Transmit element fired")
        ax.legend(loc="upper left", fontsize=8)
        return fig
    ax.plot(1e6 * time_s, traces[center_rx], color="tab:blue", lw=0.9)
    ax.axvline(1e6 * arrivals[center_rx], color="orange", ls="--", lw=1,
               label="Straight-ray arrival in water")
    ax.set(title=f"Receiver {center_rx} (arc center)", xlabel="Time (µs)",
           ylabel="Pressure (arb.)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def animate_field(snapshots, dt, x_m, y_m, tx_array, rx_array, focus, active, interval):
    steps = sorted(snapshots)
    vmax = max(float(np.percentile(np.abs(snapshots[s]), 99.9)) for s in steps) or 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(snapshots[steps[0]], origin="lower", extent=extent_mm(x_m, y_m),
                   cmap="seismic", vmin=-vmax, vmax=vmax)
    draw_arrays(ax, tx_array, rx_array, focus, active, legend=False)
    ax.set(xlabel="x (mm)", ylabel="y (mm)")

    def update(frame):
        im.set_data(snapshots[steps[frame]])
        ax.set_title(f"t = {1e6 * steps[frame] * dt:.1f} µs")
        return (im,)

    return fig, FuncAnimation(fig, update, frames=len(steps), interval=interval, blit=False)


def finish(figure, args):
    if args.save_figure:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_figure, dpi=180)
        print(f"  Saved figure:                    {args.save_figure}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show(block=True)


def main():
    args = parse_args()
    if args.n_tx < 2 or args.n_rx < 2:
        raise SystemExit("Each arc needs at least 2 elements.")
    if args.radius_mm <= 0 or (args.rx_radius_mm is not None and args.rx_radius_mm <= 0):
        raise SystemExit("Arc radii must be positive.")
    if args.cpu_threads < 0:
        raise SystemExit("--cpu-threads must be zero or positive.")
    if args.interval < 1:
        raise SystemExit("--interval must be at least 1 ms.")
    if args.parallel_shots < 1:
        raise SystemExit("--parallel-shots must be at least 1.")
    if args.each_element and args.tx is None:
        args.tx = args.n_tx // 2  # Shot shown in the dashboard.

    x_m = axis_centers(NX, DX)
    y_m = axis_centers(NY, DY)
    tx_radius = 1e-3 * args.radius_mm
    rx_radius = 1e-3 * (args.rx_radius_mm if args.rx_radius_mm is not None else args.radius_mm)
    focus = (1e-3 * args.focus_mm[0], 1e-3 * args.focus_mm[1])
    try:
        tx_array = ArcArray(args.n_tx, tx_radius, args.tx_span_deg, TX_CENTER_DEG, x_m, y_m)
        rx_array = ArcArray(args.n_rx, rx_radius, args.rx_span_deg, RX_CENTER_DEG, x_m, y_m)
        delays, active = focusing_delays(tx_array, focus, C0, args.tx)
    except ValueError as exc:
        raise SystemExit(f"Invalid array geometry: {exc}") from exc
    if arcs_overlap(tx_array, rx_array, DX):
        raise SystemExit(
            f"Transmit ({args.tx_span_deg:g}°) and receive ({args.rx_span_deg:g}°) arcs "
            "overlap on the same circle; their spans must total less than 360° "
            "(or use a different --rx-radius-mm)."
        )
    if not (x_m[0] < focus[0] < x_m[-1] and y_m[0] < focus[1] < y_m[-1]):
        raise SystemExit("--focus-mm must lie inside the simulation domain.")
    c = build_medium(x_m, y_m, args.phantom, fit_radius=min(tx_radius, rx_radius))

    print("2D arc-array FDTD")
    print(f"  Tx arc: {args.n_tx} elements, {args.tx_span_deg:g}° at {TX_CENTER_DEG:g}°, "
          f"R = {1e3 * tx_radius:g} mm, pitch {1e3 * tx_array.pitch_m:.3f} mm, "
          f"aperture {1e3 * chord_aperture(tx_radius, args.tx_span_deg):.1f} mm")
    print(f"  Rx arc: {args.n_rx} elements, {args.rx_span_deg:g}° at {RX_CENTER_DEG:g}°, "
          f"R = {1e3 * rx_radius:g} mm, pitch {1e3 * rx_array.pitch_m:.3f} mm")
    if args.each_element:
        print(f"  Firing each of the {args.n_tx} transmit elements individually "
              f"(batches of {args.parallel_shots}); displaying shot {args.tx}")
    elif args.tx is None:
        print(f"  Focus: ({args.focus_mm[0]:g}, {args.focus_mm[1]:g}) mm; "
              f"max firing delay {1e6 * delays.max():.3f} µs")
    else:
        print(f"  Single transmit element: {args.tx}")

    if args.geometry_only:
        fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
        plot_geometry(ax, c, x_m, y_m, tx_array, rx_array, focus, active)
        finish(fig, args)
        return

    try:
        device = resolve_device(args.device)
        configure_runtime(device, cpu_threads=args.cpu_threads, allow_tf32=args.allow_tf32)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not configure PyTorch: {exc}") from exc
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise SystemExit("Apple MPS does not support this simulation in float64.")
    compile_step = args.compile == "on" or (args.compile == "auto" and device.type == "cuda")

    model = TorchScalarWave2D(c, DX, DY, device=device, dtype=dtype)
    dt = stable_dt(model.c_max, DX, DY, cfl=CFL)
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    sampler = TorchRingSampler(rx_array, device=device, dtype=dtype)

    n_pulse = max(int(round(CHIRP_DURATION / dt)), 2)
    pulse_center = 0.5 * (n_pulse - 1) * dt
    c_slow = float(c.min())
    arrivals = expected_arrivals(tx_array, rx_array, focus, delays, active, C0, pulse_center)
    if args.each_element:
        # Record long enough for the longest element-to-receiver path of any shot.
        slowest = np.concatenate([
            expected_arrivals(tx_array, rx_array, focus, delays,
                              np.arange(tx_array.n_elements) == i, c_slow, pulse_center)
            for i in range(tx_array.n_elements)
        ])
    else:
        slowest = expected_arrivals(tx_array, rx_array, focus, delays, active, c_slow,
                                    pulse_center)
    n_steps = int(math.ceil(TIME_MARGIN * (slowest.max() + pulse_center) / dt))
    source_values = delayed_chirps(n_steps, dt, delays, n_pulse)
    source_values[~active] = 0.0

    # Snapshot when the beam converges on the focus (or, for one element,
    # halfway between the arcs).
    if args.tx is None:
        focal_time = delays[0] + np.hypot(tx_array.x[0] - focus[0],
                                          tx_array.y[0] - focus[1]) / C0 + pulse_center
    else:
        focal_time = 0.5 * arrivals[rx_array.n_elements // 2]
    focal_step = min(int(round(focal_time / dt)), n_steps - 1)
    snapshot_steps = {focal_step}
    if args.animate and not args.no_show:
        snapshot_steps |= set(range(0, n_steps, SNAPSHOT_STRIDE))

    t0 = time.perf_counter()
    all_shots = None
    if args.each_element:
        all_shots = simulate_each_element(solver, sampler, tx_array, source_values[args.tx],
                                          n_steps, args.parallel_shots)
    # The displayed shot is re-run on its own to capture wavefield snapshots.
    traces, snapshots = simulate(solver, sampler, tx_array, source_values, n_steps, snapshot_steps)
    if all_shots is not None:
        traces = all_shots[args.tx]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    time_s = np.arange(n_steps) * dt

    center_rx = rx_array.n_elements // 2
    peak_time = time_s[int(np.argmax(np.abs(traces[center_rx])))]
    print(f"  Grid: {NX} x {NY}, dx = {1e3 * DX:g} mm; phantom: {args.phantom}")
    print(f"  Device: {device_summary(device)}; {args.dtype}; compiled: {compile_step}")
    print(f"  dt = {1e9 * dt:.2f} ns, CFL = {solver.cfl:.3f}, steps = {n_steps} "
          f"({1e6 * time_s[-1]:.1f} µs)")
    print(f"  Simulation time: {elapsed:.2f} s"
          + (f" ({args.n_tx} shots + 1 display shot)" if args.each_element else ""))
    print(f"  Peak received amplitude: {np.abs(traces).max():.4g}")
    print(f"  Receiver {center_rx}: peak at {1e6 * peak_time:.2f} µs; "
          f"straight-ray water arrival {1e6 * arrivals[center_rx]:.2f} µs")

    if args.save_traces:
        args.save_traces.parent.mkdir(parents=True, exist_ok=True)
        # With --each-element, traces has shape (n_tx, n_rx, n_steps) and
        # traces[i] is the shot fired by transmit element i alone.
        np.savez(
            args.save_traces,
            traces=all_shots if all_shots is not None else traces,
            each_element=np.asarray(args.each_element),
            displayed_tx=np.asarray(-1 if args.tx is None else args.tx),
            time_s=time_s, dt=dt,
            tx_x=tx_array.x, tx_y=tx_array.y, tx_theta=tx_array.theta,
            rx_x=rx_array.x, rx_y=rx_array.y, rx_theta=rx_array.theta,
            delays_s=delays, active_tx=active, focus_m=np.asarray(focus),
            pulse=delayed_chirps(n_pulse, dt, np.zeros(1), n_pulse)[0],
            sound_speed_m_s=c, x_m=x_m, y_m=y_m,
        )
        print(f"  Saved traces:                    {args.save_traces}")

    if args.animate and not args.no_show:
        anim_fig, _animation = animate_field(snapshots, dt, x_m, y_m, tx_array, rx_array,
                                             focus, active, args.interval)
        plt.show(block=True)
        plt.close(anim_fig)

    figure = plot_dashboard(c, x_m, y_m, tx_array, rx_array, focus, active,
                            snapshots[focal_step], focal_step * dt,
                            traces, time_s, arrivals, center_rx,
                            all_shots=all_shots, shown_tx=args.tx)
    finish(figure, args)


if __name__ == "__main__":
    main()
