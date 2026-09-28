"""Limited-angle nonlinear FWI for sound speed with opposing arc arrays.

Same algorithm as invert_ring.py, but the acquisition covers only part of the
circle. A transmit arc (default: 64 elements over 30 degrees at the bottom)
fires one element per shot, and a separate receive arc (default: 64 elements
over 30 degrees at the top) records every shot. Both arcs lie on a circle
about the origin, and the unknown slowness-squared m = 1/c^2 is estimated on
pixels inside that circle; outside, c stays at the water background.

Observed traces are generated from a known phantom with the same forward
operator. The adjoint-state gradient, Polak–Ribière CG / gradient descent /
Adam optimizers, random shot mini-batches, and multiscale continuation are all
reused from invert_ring.py. Because transmitters are not receivers here, no
receiver channel is muted in the residual.

With limited angular coverage, only wavenumbers roughly along the
transmit-receive axis are illuminated, so expect elongated, smeared
reconstructions compared with the full ring.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

from fdtd2d import (
    ArcArray,
    axis_centers,
    load_ct_ring_grid,
    n_steps_for_crossing,
    stable_dt,
)
from fdtd2d.inversion import (
    c_from_m,
    interior_mask,
    m_from_c,
    polak_ribiere,
)
from fdtd2d.phantoms import SHEPP_DC, shepp_logan_speed
from fdtd2d.torch_backend import (
    TorchFDTD2D,
    TorchFDTD2DBatch,
    TorchRingSampler,
    TorchScalarWave2D,
    configure_runtime,
    device_summary,
    linear_chirp,
    resolve_device,
    simulate_shots_batch,
)
from invert_ring import (
    C0,
    CFL,
    PHANTOM_C,
    PHANTOM_RADIUS,
    IterationVisualizer,
    TorchLeastSquaresFWI,
    adam,
    fixed_step_gradient_descent,
    multiscale_stages,
    resize_slowness,
    rms_c_error,
)


ARC_RADIUS = 0.070
N_TX = 64
N_RX = 64
TX_SPAN_DEG = 30.0
RX_SPAN_DEG = 30.0
TX_CENTER_DEG = 270.0  # bottom
RX_CENTER_DEG = 90.0   # top


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D limited-angle arc FWI on slowness-squared with a PyTorch adjoint",
    )
    medium = parser.add_mutually_exclusive_group()
    medium.add_argument(
        "--phantom",
        choices=("disk", "shepp-logan"),
        default="disk",
        help="True sound-speed phantom (default: disk)",
    )
    medium.add_argument(
        "--ct-speed",
        type=Path,
        metavar="PATH",
        help="Load the .npz sound-speed map produced by ct_to_speed.py.",
    )
    parser.add_argument(
        "--ring-clearance-mm",
        type=float,
        default=10.0,
        metavar="MM",
        help="CT body-to-arc-circle clearance (default: 10 mm).",
    )
    parser.add_argument(
        "--ct-padding-speed",
        type=float,
        default=1480.0,
        metavar="M_S",
        help="Coupling-medium speed outside the CT body (default: 1480 m/s).",
    )
    parser.add_argument(
        "--ct-edge-margin",
        type=int,
        default=8,
        metavar="PIXELS",
        help="Grid boundary margin outside the CT arc circle (default: 8 pixels).",
    )
    parser.add_argument(
        "--n-tx", type=int, default=N_TX, metavar="N",
        help=f"Transmit-arc elements (default: {N_TX})",
    )
    parser.add_argument(
        "--n-rx", type=int, default=N_RX, metavar="N",
        help=f"Receive-arc elements (default: {N_RX})",
    )
    parser.add_argument(
        "--tx-span-deg", type=float, default=TX_SPAN_DEG, metavar="DEG",
        help="Angle covered by the transmit arc, centered at the bottom; "
             f"180 is the lower half circle (default: {TX_SPAN_DEG:g})",
    )
    parser.add_argument(
        "--rx-span-deg", type=float, default=RX_SPAN_DEG, metavar="DEG",
        help="Angle covered by the receive arc, centered at the top "
             f"(default: {RX_SPAN_DEG:g})",
    )
    parser.add_argument(
        "--radius-mm", type=float, default=1e3 * ARC_RADIUS, metavar="MM",
        help="Radius of the circle both arcs lie on; ignored with --ct-speed "
             f"(default: {1e3 * ARC_RADIUS:g})",
    )
    parser.add_argument(
        "--n-shots", type=int, default=8, metavar="K",
        help="Random transmitters per gradient evaluation (default: 8)",
    )
    parser.add_argument(
        "--parallel-shots", type=int, default=16, metavar="N",
        help=(
            "Maximum independent shots per GPU batch for observed and "
            "inversion modeling (default: 16)."
        ),
    )
    parser.add_argument(
        "--shot-seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Seed for random transmitter mini-batches (default: 0)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=15, metavar="K",
        help="Optimization iterations (default: 15)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("cg", "gradient-descent", "adam"),
        default="adam",
        help="Optimization method (default: adam)",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help=(
            "Use 25–75 kHz / 2 mm, 50–125 kHz / 1 mm, and "
            "100–250 kHz / 0.5 mm coarse-to-fine stages."
        ),
    )
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="Run only the 25–75 kHz, 101 x 101 coarse multiscale stage.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0e-15,
        metavar="ALPHA",
        help="Fixed gradient-descent step in slowness-squared units (default: 1e-15)",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=1.0e-9,
        metavar="ALPHA",
        help="Adam learning rate in slowness-squared units (default: 1e-9)",
    )
    parser.add_argument(
        "--reg", type=float, default=0.0, metavar="ALPHA",
        help="Tikhonov weight on interior ∇m (default: 0)",
    )
    parser.add_argument(
        "--c-min", type=float, default=1400.0, metavar="M/S",
        help="Lower clip on reconstructed sound speed (default: 1400)",
    )
    parser.add_argument(
        "--c-max", type=float, default=2000.0, metavar="M/S",
        help="Upper clip on reconstructed sound speed (default: 2000)",
    )
    parser.add_argument(
        "--margin-pixels", type=int, default=2, metavar="K",
        help="Interior mask shrinks this many pixels inside the arc circle (default: 2)",
    )
    parser.add_argument(
        "--gradient-check", action="store_true",
        help="Run a tiny-grid finite-difference adjoint test and exit",
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
        help="Propagation precision (default: float32)",
    )
    parser.add_argument(
        "--compile",
        choices=("auto", "on", "off"),
        default="auto",
        help="Compile leapfrog modes; auto enables this on CUDA (default: auto)",
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
        "--show-iterations",
        action="store_true",
        help="Live view of the speed estimate, update, and misfit after each iteration",
    )
    parser.add_argument(
        "--save-figure", type=Path, metavar="PATH",
        help="Save the inversion dashboard as an image",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Run without opening windows",
    )
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


def build_arcs(args, radius, x_m, y_m, allow_shared_pixels=False):
    """Transmit arc at the bottom and receive arc at the top of one circle."""
    tx_array = ArcArray(args.n_tx, radius, args.tx_span_deg, TX_CENTER_DEG, x_m, y_m,
                        allow_shared_pixels=allow_shared_pixels)
    rx_array = ArcArray(args.n_rx, radius, args.rx_span_deg, RX_CENTER_DEG, x_m, y_m,
                        allow_shared_pixels=allow_shared_pixels)
    return tx_array, rx_array


def arcs_share_pixels(args, radius, spacing):
    """True if either arc's element pitch is below the grid spacing."""
    pitches = (radius * np.deg2rad(args.tx_span_deg) / (args.n_tx - 1),
               radius * np.deg2rad(args.rx_span_deg) / (args.n_rx - 1))
    return min(pitches) < spacing


class ArcLeastSquaresFWI(TorchLeastSquaresFWI):
    """Ring FWI with separate transmit (``array``) and receive (``sampler``) arcs.

    The ring version zeroes the transmitter's own receiver channel. Here no
    transmitter is also a receiver, so the full residual is kept.
    """

    @staticmethod
    def _residual(predicted, observed, tx):
        return predicted - observed

    @staticmethod
    def _residual_batch(predicted, observed, sources):
        return predicted - observed


def validate_shot_count(n_tx, n_shots):
    if n_shots < 1:
        raise SystemExit("Need at least 1 shot.")
    if n_shots > n_tx:
        raise SystemExit("Cannot have more shots than transmit elements.")


def simulate_observed(solver, sampler, tx_array, pulse, source_values, n_steps,
                      parallel_shots, batched_solvers):
    """Fire every transmit element once; return (n_tx, n_rx, n_steps) traces."""
    model = solver.model
    observed = torch.empty(
        (tx_array.n_elements, sampler.n_elements, n_steps),
        device=model.device, dtype=model.dtype,
    )
    batch_size = min(parallel_shots, tx_array.n_elements)
    for start in range(0, tx_array.n_elements, batch_size):
        sources = np.arange(start, min(start + batch_size, tx_array.n_elements))
        batched = batched_solvers.get(len(sources))
        if batched is None:
            batched = TorchFDTD2DBatch(solver, len(sources))
            batched_solvers[len(sources)] = batched
        traces, _ = simulate_shots_batch(
            solver, sampler, pulse, tx_array.rows[sources], tx_array.cols[sources],
            n_steps, batched=batched, source_values=source_values,
        )
        observed[start:start + len(sources)].copy_(traces)
    return observed


def run_arc_gradient_check(device, dtype, compile_step, seed=0):
    """Directional finite-difference check with distinct transmit/receive arcs."""
    n = 41
    spacing = 2.0e-3
    radius = 0.030
    x_m = axis_centers(n, spacing)
    y_m = axis_centers(n, spacing)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c_true = np.full((n, n), C0)
    c_true[x * x + y * y <= 0.012**2] = PHANTOM_C
    dt = stable_dt(2000.0, spacing, spacing, cfl=CFL)
    pulse = linear_chirp(dt, 20.0e-6, 1.0e5, 2.5e5, device=device, dtype=dtype)
    n_steps = max(n_steps_for_crossing(radius, float(c_true.min()), dt), pulse.numel() + 1)
    source_values = torch.zeros(n_steps, device=device, dtype=dtype)
    source_values[:pulse.numel()].copy_(pulse)
    model = TorchScalarWave2D(c_true, spacing, spacing, device=device, dtype=dtype)
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    # Unequal element counts exercise the separate transmit/receive shapes.
    tx_array = ArcArray(3, radius, 60.0, TX_CENTER_DEG, x_m, y_m)
    rx_array = ArcArray(5, radius, 60.0, RX_CENTER_DEG, x_m, y_m)
    sampler = TorchRingSampler(rx_array, device=device, dtype=dtype)
    observed = simulate_observed(solver, sampler, tx_array, pulse, source_values, n_steps,
                                 parallel_shots=tx_array.n_elements, batched_solvers={})
    mask = interior_mask(x_m, y_m, radius, margin_m=2.0 * spacing)
    problem = ArcLeastSquaresFWI(
        solver, sampler, tx_array, pulse, n_steps, tx_array.n_elements, observed, mask,
        c_min=1400.0, c_max=2000.0, c_background=C0, shot_seed=seed,
    )
    rng = np.random.default_rng(seed)
    m0 = np.full((n, n), m_from_c(C0))
    m0[mask] *= 1.0 + 0.02 * rng.standard_normal(int(mask.sum()))
    m0 = problem.project(m0)
    value, gradient = problem.misfit_and_grad(m0)
    direction = np.zeros_like(m0)
    direction[mask] = rng.standard_normal(int(mask.sum()))
    direction *= np.linalg.norm(m0[mask]) / np.linalg.norm(direction[mask])
    directional = float(np.dot(gradient[mask], direction[mask]))
    print("PyTorch arc adjoint gradient check")
    print(f"  Device: {device_summary(device)}")
    print(f"  Transmit / receive elements: {tx_array.n_elements} / {rx_array.n_elements}")
    print(f"  J = {value:.6e}")
    print(f"  g · δm = {directional:.6e}")
    print(f"  {'eps':>10}  {'FD':>14}  {'rel. error':>12}")
    for epsilon in (1e-2, 1e-3, 1e-4):
        plus = problem.misfit(problem.project(m0 + epsilon * direction))
        minus = problem.misfit(problem.project(m0 - epsilon * direction))
        finite_difference = (plus - minus) / (2.0 * epsilon)
        relative_error = abs(finite_difference - directional) / max(abs(directional), 1e-30)
        print(f"  {epsilon:10.1e}  {finite_difference:14.6e}  {relative_error:12.4e}")


def overlay_arcs(ax, tx_array, rx_array, mask_radius_m=None):
    ax.plot(1e3 * tx_array.x, 1e3 * tx_array.y, ".", color="red", ms=3,
            label=f"Tx arc ({tx_array.n_elements} el, {tx_array.span_deg:g}°)")
    ax.plot(1e3 * rx_array.x, 1e3 * rx_array.y, ".", color="deepskyblue", ms=3,
            label=f"Rx arc ({rx_array.n_elements} el, {rx_array.span_deg:g}°)")
    if mask_radius_m is not None:
        ax.add_patch(Circle((0.0, 0.0), 1e3 * mask_radius_m, fill=False,
                            linestyle="--", color="0.5", lw=0.8))


def plot_arc_inversion(c_true, c_est, history, x_m, y_m, tx_array, rx_array, mask,
                       optimizer="adam"):
    """True / reconstructed / difference sound speed and misfit history."""
    labels = {
        "cg": ("adjoint Polak–Ribière CG", "CG iteration"),
        "gradient-descent": ("adjoint fixed-step gradient descent", "Gradient-descent iteration"),
        "adam": ("adjoint projected Adam", "Adam iteration"),
    }
    method_title, iteration_label = labels[optimizer]
    dx, dy = x_m[1] - x_m[0], y_m[1] - y_m[0]
    extent = [1e3 * (x_m[0] - dx / 2), 1e3 * (x_m[-1] + dx / 2),
              1e3 * (y_m[0] - dy / 2), 1e3 * (y_m[-1] + dy / 2)]
    vmin = float(min(c_true.min(), c_est.min()))
    vmax = float(max(c_true.max(), c_est.max()))
    diff = np.where(mask, c_est - c_true, 0.0)
    vabs = max(float(np.abs(diff).max()), 1e-12)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    mask_radius = float(np.sqrt(x[mask]**2 + y[mask]**2).max()) if mask.any() else None

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    fig.suptitle(f"Limited-angle FWI  (slowness-squared, {method_title})", fontsize=14)
    for ax, data, title, cmap, lo, hi in (
        (axes[0, 0], c_true, "True $c$ (m/s)", "viridis", vmin, vmax),
        (axes[0, 1], c_est, "Reconstructed $c$ (m/s)", "viridis", vmin, vmax),
        (axes[1, 0], diff, "Difference (est. $-$ true)", "seismic", -vabs, vabs),
    ):
        image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                          vmin=lo, vmax=hi, aspect="equal")
        overlay_arcs(ax, tx_array, rx_array, mask_radius)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc="upper right", fontsize=7)

    ax = axes[1, 1]
    misfit = np.asarray(history["misfit"], dtype=float)
    ax.plot(np.arange(misfit.size), misfit, "o-", color="tab:blue", lw=1.5)
    ax.set(title="Waveform misfit", xlabel=iteration_label, ylabel="$J(m)$")
    if misfit.size and np.all(misfit > 0.0):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def print_summary(
    c_true, c_est, mask, history, args, source_history, dt, n_steps,
    execution_device, compiled, tx_array, rx_array, wall_time_s=None, stage_reports=(),
):
    print("2D limited-angle arc FWI results")
    if args.ct_speed is None:
        print(f"  Phantom:                         {args.phantom}")
    else:
        print(f"  Phantom:                         CT sound-speed map ({args.ct_speed})")
    print(f"  Transmit arc:                    {tx_array.n_elements} elements, "
          f"{tx_array.span_deg:g}° at {TX_CENTER_DEG:g}°")
    print(f"  Receive arc:                     {rx_array.n_elements} elements, "
          f"{rx_array.span_deg:g}° at {RX_CENTER_DEG:g}°")
    print(f"  Arc radius:                      {1e3 * tx_array.radius_m:.2f} mm")
    print(f"  Precomputed observed shots:      all {tx_array.n_elements} transmitters")
    print(f"  Gradient shots per evaluation:   {args.n_shots}")
    print(f"  Random shot seed:                {args.shot_seed}")
    if source_history:
        print(f"  Last gradient transmitters:      {[int(tx) for tx in source_history[-1]]}")
    print(f"  Time steps:                      {n_steps}")
    print(f"  dt:                              {dt:.4g} s")
    print(f"  PyTorch device:                  {execution_device}")
    print(f"  Precision / compiled step:       {args.dtype} / {compiled}")
    print(f"  Optimizer:                       {args.optimizer}")
    if args.optimizer == "gradient-descent":
        print(f"  Fixed step size:                 {args.step_size:.4g}")
    elif args.optimizer == "adam":
        print(f"  Adam learning rate:              {args.adam_lr:.4g}")
    print(f"  Optimization iterations:         {max(len(history['misfit']) - 1, 0)}")
    if wall_time_s is not None:
        print(f"  Total wall time:                 {wall_time_s:.3f} s")
    if stage_reports:
        print("  Stage timing:")
        for report in stage_reports:
            print(
                f"    {report['index']}. {report['grid_size']} x "
                f"{report['grid_size']}, {report['spacing_mm']:.3g} mm, "
                f"{report['f_start_khz']:.0f}–{report['f_end_khz']:.0f} kHz"
            )
            print(
                f"       observed data {report['observed_s']:.3f} s; "
                f"optimization {report['optimization_s']:.3f} s; "
                f"stage total {report['total_s']:.3f} s; "
                f"{report['iterations']} iterations"
            )
    print(f"  Initial J:                       {history['misfit'][0]:.6e}")
    print(f"  Final J:                         {history['misfit'][-1]:.6e}")
    print(f"  Interior RMS c error:            {rms_c_error(c_est, c_true, mask):.3f} m/s")
    print(f"  Interior c range (true):         "
          f"{float(c_true[mask].min()):.0f}–{float(c_true[mask].max()):.0f} m/s")
    print(f"  Interior c range (reconstructed): "
          f"{float(c_est[mask].min()):.0f}–{float(c_est[mask].max()):.0f} m/s")
    if args.ct_speed is not None:
        return
    if args.phantom == "disk":
        print(f"  Disk target:                     r = {1e3 * PHANTOM_RADIUS:.1f} mm, "
              f"c = {PHANTOM_C:.0f} m/s  (background {C0:.0f} m/s)")
    else:
        print(f"  Shepp–Logan target:              c = {C0:.0f}–{C0 + SHEPP_DC:.0f} m/s")


def run_inversion_stage(
    args, device, dtype, compile_step, grid_size, spacing, f_start, f_end,
    chirp_duration, initial_m=None, stage_index=1,
):
    """Generate limited-angle data and invert one frequency/grid level."""
    stage_start = time.perf_counter()
    ct_grid = None
    if args.ct_speed is not None:
        ct_grid = load_ct_ring_grid(
            args.ct_speed,
            grid_shape=(grid_size, grid_size),
            padding_speed_m_s=args.ct_padding_speed,
            ring_clearance_mm=args.ring_clearance_mm,
            edge_margin_pixels=args.ct_edge_margin,
        )
        c_true = ct_grid.c
        x_m = ct_grid.x_m
        y_m = ct_grid.y_m
        spacing = ct_grid.spacing_m
        radius = ct_grid.ring_radius_m
        background_speed = args.ct_padding_speed
    else:
        x_m = axis_centers(grid_size, spacing)
        y_m = axis_centers(grid_size, spacing)
        radius = 1e-3 * args.radius_mm
        c_true = build_medium(x_m, y_m, phantom=args.phantom, fit_radius=radius)
        background_speed = C0
    if args.ct_speed is not None and (
        float(c_true.min()) < args.c_min or float(c_true.max()) > args.c_max
    ):
        print(
            "  Warning: CT speeds exceed the reconstruction bounds; "
            f"use --c-min {float(c_true.min()):.0f} and --c-max "
            f"{float(c_true.max()):.0f} (or wider) to recover their full range."
        )
    shared = arcs_share_pixels(args, radius, spacing)
    if shared:
        print(f"  Note: element pitch is below the {1e3 * spacing:.3g} mm grid spacing; "
              "neighboring elements share pixels on this stage.")
    tx_array, rx_array = build_arcs(args, radius, x_m, y_m, allow_shared_pixels=shared)

    c_ceiling = max(float(c_true.max()), float(args.c_max), C0)
    dt = stable_dt(c_ceiling, spacing, spacing, cfl=CFL)
    # Longest straight path between the arcs is at most the circle diameter.
    n_steps = n_steps_for_crossing(radius, float(c_true.min()), dt)
    pulse = linear_chirp(dt, chirp_duration, f_start, f_end, device=device, dtype=dtype)
    n_steps = max(n_steps, pulse.numel() + 1)
    source_values = torch.zeros(n_steps, device=device, dtype=dtype)
    source_values[:pulse.numel()].copy_(pulse)

    model = TorchScalarWave2D(c_true, spacing, spacing, device=device, dtype=dtype)
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    sampler = TorchRingSampler(rx_array, device=device, dtype=dtype)
    print(
        "  Precomputing observed traces "
        f"({min(args.parallel_shots, args.n_tx)} shots per GPU batch)..."
    )
    batched_solvers = {}
    observed = simulate_observed(solver, sampler, tx_array, pulse, source_values, n_steps,
                                 args.parallel_shots, batched_solvers)
    observed_time_s = time.perf_counter() - stage_start

    mask = interior_mask(
        x_m, y_m, radius, margin_m=args.margin_pixels * spacing, center=tx_array.center,
    )
    problem = ArcLeastSquaresFWI(
        solver, sampler, tx_array, pulse, n_steps, args.n_shots, observed, mask,
        alpha=args.reg, c_min=args.c_min, c_max=args.c_max,
        c_background=background_speed,
        shot_seed=args.shot_seed, parallel_shots=args.parallel_shots,
        batched_solvers=batched_solvers,
    )
    if initial_m is None:
        m0 = np.full(c_true.shape, m_from_c(background_speed))
    else:
        m0 = problem.project(resize_slowness(initial_m, c_true.shape))

    visualizer = None
    if args.show_iterations:
        visualizer = IterationVisualizer(x_m, y_m, mask, args.c_min, args.c_max)

    def evaluated_objective(m):
        value, gradient = problem.misfit_and_grad(m)
        if visualizer is not None:
            visualizer.update(m, value)
        return value, gradient

    optimization_start = time.perf_counter()
    if args.optimizer == "gradient-descent":
        m_est, history = fixed_step_gradient_descent(
            m0, evaluated_objective, problem.project, mask,
            step_size=args.step_size, max_iter=args.max_iter, verbose=True,
        )
    elif args.optimizer == "adam":
        m_est, history = adam(
            m0, evaluated_objective, problem.project, mask,
            learning_rate=args.adam_lr, max_iter=args.max_iter, verbose=True,
        )
    else:
        m_est, history = polak_ribiere(
            m0, problem.misfit, evaluated_objective, problem.project, mask,
            max_iter=args.max_iter, verbose=True,
        )
    if visualizer is not None:
        plt.close(visualizer.figure)
    optimization_time_s = time.perf_counter() - optimization_start
    stage_report = {
        "index": stage_index,
        "grid_size": grid_size,
        "spacing_mm": 1.0e3 * spacing,
        "f_start_khz": f_start / 1.0e3,
        "f_end_khz": f_end / 1.0e3,
        "observed_s": observed_time_s,
        "optimization_s": optimization_time_s,
        "total_s": time.perf_counter() - stage_start,
        "iterations": max(len(history["misfit"]) - 1, 0),
    }
    return (
        c_true, c_from_m(m_est), m_est, mask, history, problem, tx_array, rx_array,
        x_m, y_m, dt, n_steps, stage_report,
    )


def main():
    args = parse_args()
    if args.n_tx < 2 or args.n_rx < 2:
        raise SystemExit("Each arc needs at least 2 elements.")
    validate_shot_count(args.n_tx, args.n_shots)
    if args.tx_span_deg + args.rx_span_deg >= 360.0:
        raise SystemExit("Transmit and receive spans must total less than 360°.")
    if args.radius_mm <= 0:
        raise SystemExit("--radius-mm must be positive.")
    if args.parallel_shots < 1:
        raise SystemExit("--parallel-shots must be at least 1.")
    if args.ring_clearance_mm <= 0:
        raise SystemExit("--ring-clearance-mm must be positive.")
    if args.ct_padding_speed <= 0:
        raise SystemExit("--ct-padding-speed must be positive.")
    if args.ct_edge_margin < 3:
        raise SystemExit("--ct-edge-margin must be at least 3.")
    if args.multiscale and args.coarse_only:
        raise SystemExit("--multiscale and --coarse-only cannot be combined.")
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be at least 1.")
    if args.step_size <= 0.0:
        raise SystemExit("--step-size must be positive.")
    if args.adam_lr <= 0.0:
        raise SystemExit("--adam-lr must be positive.")
    if args.c_min <= 0.0 or args.c_max <= args.c_min:
        raise SystemExit("Need 0 < --c-min < --c-max.")
    if args.margin_pixels < 0:
        raise SystemExit("--margin-pixels must be non-negative.")
    if args.cpu_threads < 0:
        raise SystemExit("--cpu-threads must be zero or positive.")
    if args.show_iterations and args.no_show:
        raise SystemExit("--show-iterations cannot be combined with --no-show.")
    try:
        device = resolve_device(args.device)
        configure_runtime(device, cpu_threads=args.cpu_threads, allow_tf32=args.allow_tf32)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not configure PyTorch: {exc}") from exc
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise SystemExit("Apple MPS does not support this inversion in float64.")
    compile_step = args.compile == "on" or (args.compile == "auto" and device.type == "cuda")
    if args.gradient_check:
        run_arc_gradient_check(device, dtype, compile_step)
        return

    wall_time_start = time.perf_counter()
    transferred_m = None
    stage_reports = []
    stages = multiscale_stages(args.multiscale or args.coarse_only)
    if args.coarse_only:
        stages = stages[:1]
    for stage_index, stage in enumerate(stages, start=1):
        grid_size, spacing, f_start, f_end, duration = stage
        print(
            f"Stage {stage_index}/{len(stages)}: {grid_size} x {grid_size}, "
            f"{f_start / 1e3:.0f}–{f_end / 1e3:.0f} kHz"
        )
        try:
            (
                c_true, c_est, transferred_m, mask, history, problem, tx_array, rx_array,
                x_m, y_m, dt, n_steps, stage_report,
            ) = run_inversion_stage(
                args, device, dtype, compile_step, grid_size, spacing, f_start,
                f_end, duration, initial_m=transferred_m, stage_index=stage_index,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid acquisition setup: {exc}") from exc
        stage_reports.append(stage_report)
    wall_time_s = time.perf_counter() - wall_time_start
    print_summary(
        c_true, c_est, mask, history, args, problem.source_history, dt, n_steps,
        device_summary(device), compile_step, tx_array, rx_array,
        wall_time_s=wall_time_s, stage_reports=stage_reports,
    )

    figure_path = args.save_figure
    if args.no_show and figure_path is None:
        figure_path = Path("invert_arc_dashboard.png")
    if not args.no_show or figure_path:
        figure = plot_arc_inversion(
            c_true, c_est, history, x_m, y_m, tx_array, rx_array, mask,
            optimizer=args.optimizer,
        )
        if figure_path:
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(figure_path, dpi=180)
            print(f"Saved inversion dashboard: {figure_path}")
        if not args.no_show:
            plt.show(block=True)
        else:
            plt.close(figure)


if __name__ == "__main__":
    main()
