"""Steady-state finite-difference frequency-domain ring-array simulator.

This is the FDFD analogue of ``../2DRingFDTD.py``.  A point element on the
same circular ring emits a monochromatic source, the 2-D Helmholtz equation is
solved with an absorbing outer layer, and every ring element records complex
pressure (amplitude and phase) at the chosen frequency.

The sparse Helmholtz matrix is never assembled.  CUDA runs the finite-
difference stencil and batches independent transmitter solves on the GPU.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fdtd2d import RingArray, axis_centers, load_ct_ring_grid
from fdtd2d.phantoms import shepp_logan_speed
from fdtd2d.torch_backend import configure_runtime, device_summary, resolve_device
from invert_ring_fdfd import (
    BACKGROUND_C,
    DISK_C,
    DISK_RADIUS_M,
    DOMAIN_WIDTH_M,
    Helmholtz2D,
    RING_RADIUS_M,
    RingReceiver,
    sponge_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", type=int, default=0, metavar="I",
                        help="Transmitter element index (default: 0).")
    parser.add_argument("--n-elements", type=int, default=128, metavar="N")
    medium = parser.add_mutually_exclusive_group()
    medium.add_argument("--phantom", choices=("disk", "shepp-logan", "none"), default="disk")
    medium.add_argument("--ct-speed", type=Path, metavar="PATH")
    parser.add_argument("--grid-size", type=int, default=161,
                        help="Odd square FDFD grid for the 200 mm field of view.")
    parser.add_argument("--ring-clearance-mm", type=float, default=10.0)
    parser.add_argument("--ct-padding-speed", type=float, default=1480.0)
    parser.add_argument("--ct-edge-margin", type=int, default=8)
    parser.add_argument("--frequency-khz", type=float, default=75.0,
                        help="Single steady-state source frequency (default: 75 kHz).")
    parser.add_argument("--source-phase-deg", type=float, default=0.0,
                        help="Phase of the point monopole source (default: 0).")
    parser.add_argument("--pml-width-mm", type=float, default=18.0)
    parser.add_argument("--pml-strength", type=float, default=2.5)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-maxiter", type=int, default=500)
    parser.add_argument("--solver-restart", type=int, default=40)
    parser.add_argument("--parallel-shots", type=int, metavar="N",
                        help="Simulate every element as transmitter, in batches of N.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--compile", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--save-data", type=Path, metavar="PATH",
                        help="Save complex receiver data and geometry as .npz.")
    parser.add_argument("--save-figure", type=Path, metavar="PATH")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def build_medium(x_m, y_m, phantom):
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full(x.shape, BACKGROUND_C)
    if phantom == "disk":
        c[x * x + y * y <= DISK_RADIUS_M**2] = DISK_C
    elif phantom == "shepp-logan":
        c = shepp_logan_speed(x, y, RING_RADIUS_M, BACKGROUND_C)
    return c


def medium_from_args(args):
    if args.ct_speed is not None:
        ct = load_ct_ring_grid(
            args.ct_speed, grid_shape=(args.grid_size, args.grid_size),
            padding_speed_m_s=args.ct_padding_speed,
            ring_clearance_mm=args.ring_clearance_mm,
            edge_margin_pixels=args.ct_edge_margin,
        )
        return ct.c, ct.x_m, ct.y_m, ct.spacing_m, ct.ring_radius_m
    spacing = DOMAIN_WIDTH_M / (args.grid_size - 1)
    x_m = axis_centers(args.grid_size, spacing)
    y_m = axis_centers(args.grid_size, spacing)
    return build_medium(x_m, y_m, args.phantom), x_m, y_m, spacing, RING_RADIUS_M


def source_rhs(array, transmitters, shape, spacing, phase_degrees, device):
    rhs = torch.zeros((len(transmitters), *shape), device=device, dtype=torch.complex64)
    rows_cols = [array.inject_rows_cols(int(tx)) for tx in transmitters]
    rows = torch.as_tensor([item[0] for item in rows_cols], device=device)
    cols = torch.as_tensor([item[1] for item in rows_cols], device=device)
    phase = torch.tensor(math.radians(phase_degrees), device=device)
    amplitude = torch.polar(torch.ones((), device=device), phase).to(torch.complex64)
    rhs[torch.arange(len(transmitters), device=device), rows, cols] = amplitude / spacing**2
    return rhs


def simulate(args, c_map, spacing, array, device):
    compiled = args.compile == "on" or (args.compile == "auto" and device.type == "cuda")
    sponge = sponge_profile(c_map.shape, spacing, args.pml_width_mm * 1e-3,
                            args.pml_strength, device)
    operator = Helmholtz2D(c_map.shape, spacing, 2 * math.pi * args.frequency_khz * 1e3,
                           sponge, device, compile_stencil=compiled)
    operator.set_model(torch.as_tensor(1.0 / c_map**2, device=device, dtype=torch.float32))
    receiver = RingReceiver(array, device)
    transmitters = (np.arange(array.n_elements, dtype=int)
                    if args.parallel_shots is not None else np.asarray([args.tx]))
    batch_size = args.parallel_shots or 1
    traces, selected_field, worst_residual = [], None, 0.0
    for start in range(0, len(transmitters), batch_size):
        tx_batch = transmitters[start:start + batch_size]
        rhs = source_rhs(array, tx_batch, c_map.shape, spacing, args.source_phase_deg, device)
        fields, residual, _ = operator.solve(rhs, args.solver_tol, args.solver_maxiter,
                                             restart=args.solver_restart)
        worst_residual = max(worst_residual, float(residual.max().item()))
        traces.append(receiver.sample(fields).detach().cpu())
        if selected_field is None:
            selected_field = fields[0].detach().cpu().numpy()
    return transmitters, torch.cat(traces).numpy(), selected_field, worst_residual, compiled


def make_figure(c_map, field, traces, tx, x_m, y_m, array, frequency_khz):
    dx, dy = x_m[1] - x_m[0], y_m[1] - y_m[0]
    extent = [1e3 * (x_m[0] - dx / 2), 1e3 * (x_m[-1] + dx / 2),
              1e3 * (y_m[0] - dy / 2), 1e3 * (y_m[-1] + dy / 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    speed = axes[0].imshow(c_map, origin="lower", extent=extent, cmap="viridis")
    axes[0].plot(array.x * 1e3, array.y * 1e3, ".", color="white", ms=1.5)
    axes[0].plot(array.x[tx] * 1e3, array.y[tx] * 1e3, "o", color="tab:red", ms=5)
    axes[0].set(title="Sound speed and ring", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(speed, ax=axes[0], label="m/s")
    amplitude = np.abs(field)
    image = axes[1].imshow(amplitude, origin="lower", extent=extent, cmap="magma")
    axes[1].set(title=f"|p| at {frequency_khz:g} kHz", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=axes[1], label="arbitrary pressure")
    elements = np.arange(array.n_elements)
    axes[2].plot(elements, np.abs(traces), label="amplitude", color="tab:blue")
    phase_axis = axes[2].twinx()
    phase_axis.plot(elements, np.unwrap(np.angle(traces)), label="unwrapped phase", color="tab:orange", alpha=0.8)
    axes[2].axvline(tx, color="tab:red", ls="--", lw=1, label="transmitter")
    axes[2].set(title="Complex ring data", xlabel="Receiver element", ylabel="|p|")
    phase_axis.set_ylabel("phase (rad)")
    axes[2].grid(alpha=0.25)
    return fig


def save_data(path, c_map, traces, transmitters, array, x_m, y_m, frequency_khz, tx):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, pressure=traces, transmitter_indices=transmitters, tx=tx,
        frequency_hz=frequency_khz * 1e3, sound_speed_m_s=c_map,
        x_m=x_m, y_m=y_m, element_x_m=array.x, element_y_m=array.y,
        element_theta_rad=array.theta, ring_radius_m=array.radius_m,
        convention=np.asarray("p(x,t)=Re[p(x) exp(-i omega t)]"),
    )


def main():
    args = parse_args()
    if args.grid_size < 33 or args.grid_size % 2 == 0:
        raise SystemExit("--grid-size must be an odd integer at least 33.")
    if args.n_elements < 2 or not 0 <= args.tx < args.n_elements:
        raise SystemExit("--tx must select one of at least two elements.")
    if args.frequency_khz <= 0 or args.pml_width_mm <= 0 or args.pml_strength <= 0:
        raise SystemExit("Frequency and PML settings must be positive.")
    if args.parallel_shots is not None and args.parallel_shots < 1:
        raise SystemExit("--parallel-shots must be positive.")
    device = resolve_device(args.device)
    if device.type == "mps":
        raise SystemExit("Complex FDFD is supported on CPU or CUDA; choose --device cpu/cuda.")
    configure_runtime(device, allow_tf32=args.allow_tf32)
    c_map, x_m, y_m, spacing, radius = medium_from_args(args)
    array = RingArray(args.n_elements, radius, x_m, y_m)
    started = time.perf_counter()
    transmitters, traces, field, residual, compiled = simulate(args, c_map, spacing, array, device)
    elapsed = time.perf_counter() - started
    print("2D ring FDFD results")
    print(f"  device: {device_summary(device)}")
    print(f"  grid / spacing: {args.grid_size} x {args.grid_size} / {spacing * 1e3:.3f} mm")
    print(f"  ring / elements: {radius * 1e3:.1f} mm / {args.n_elements}")
    print(f"  frequency: {args.frequency_khz:g} kHz; source tx: {args.tx}")
    print(f"  transmitters solved: {len(transmitters)}; compiled stencil: {compiled}")
    print(f"  worst relative linear residual: {residual:.2e}; wall time: {elapsed:.2f} s")
    if residual > args.solver_tol * 10:
        print("  warning: increase --solver-maxiter or adjust PML settings for a tighter solve.")
    if args.save_data:
        save_data(args.save_data, c_map, traces, transmitters, array, x_m, y_m,
                  args.frequency_khz, args.tx)
        print(f"  saved complex data: {args.save_data}")
    figure = make_figure(c_map, field, traces[0], int(transmitters[0]), x_m, y_m,
                         array, args.frequency_khz)
    if args.save_figure:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_figure, dpi=180)
        print(f"  saved dashboard: {args.save_figure}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
