"""Nonlinear FWI for sound speed inside the ring-array FDTD model.

Observed traces are generated from a known phantom with the same forward
operator used by 2DRingFDTD.py. The unknown is slowness-squared m = 1/c^2
on pixels inside the ring; outside, c stays at the water background.
Polak–Ribière CG uses the adjoint-state gradient of the leapfrog scheme.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fdtd2d import (
    FDTD2D,
    RingArray,
    ScalarWave2D,
    axis_centers,
    linear_chirp,
    n_steps_for_crossing,
    simulate_shot,
    stable_dt,
)
from fdtd2d.inversion import (
    LeastSquaresFWI,
    c_from_m,
    interior_mask,
    m_from_c,
    polak_ribiere,
    run_gradient_check,
)
from fdtd2d.phantoms import SHEPP_DC, shepp_logan_speed
from fdtd2d.plot import plot_inversion


# Same domain / chirp as 2DRingFDTD.py. Lengths in meters; time in seconds.
NX = 401
NY = 401
DX = 0.50e-3
DY = DX
C0 = 1500.0
CFL = 0.45
RING_RADIUS = 0.070
N_ELEMENTS = 256

F_START_HZ = 1.00e5
F_END_HZ = 2.50e5
CHIRP_DURATION = 20.0e-6

PHANTOM_RADIUS = 0.020
PHANTOM_C = 1800.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D ring FWI: adjoint CG on slowness-squared inside the ring",
    )
    parser.add_argument(
        "--phantom",
        choices=("disk", "shepp-logan"),
        default="disk",
        help="True sound-speed phantom (default: disk)",
    )
    parser.add_argument(
        "--n-elements", type=int, default=N_ELEMENTS, metavar="N",
        help=f"Number of ring elements (default: {N_ELEMENTS})",
    )
    parser.add_argument(
        "--n-shots", type=int, default=4, metavar="K",
        help="Number of equally spaced transmitters (default: 4)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=15, metavar="K",
        help="Polak–Ribière iterations (default: 15)",
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
        help="Interior mask shrinks this many pixels inside the ring (default: 2)",
    )
    parser.add_argument(
        "--gradient-check", action="store_true",
        help="Run a tiny-grid finite-difference adjoint test and exit",
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


def build_medium(x_m, y_m, phantom="disk"):
    """Return c(x, y) in m/s."""
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full_like(x, C0)
    if phantom == "disk":
        c[x**2 + y**2 <= PHANTOM_RADIUS**2] = PHANTOM_C
    elif phantom == "shepp-logan":
        c = shepp_logan_speed(x, y, RING_RADIUS, C0)
    return c


def source_indices(n_elements, n_shots):
    if n_shots < 1:
        raise SystemExit("Need at least 1 shot.")
    if n_shots > n_elements:
        raise SystemExit("Cannot have more shots than ring elements.")
    return np.linspace(0, n_elements, n_shots, endpoint=False).astype(int)


def rms_c_error(c_est, c_true, mask):
    err = c_est[mask] - c_true[mask]
    return float(np.sqrt(np.mean(err * err)))


def print_summary(c_true, c_est, mask, history, sources, phantom, dt, n_steps):
    print("2D ring FWI results")
    print(f"  Phantom:                         {phantom}")
    print(f"  Shots (TX indices):              {list(sources)}")
    print(f"  Time steps:                      {n_steps}")
    print(f"  dt:                              {dt:.4g} s")
    print(f"  CG iterations:                   {max(len(history['misfit']) - 1, 0)}")
    print(f"  Initial J:                       {history['misfit'][0]:.6e}")
    print(f"  Final J:                         {history['misfit'][-1]:.6e}")
    print(f"  Interior RMS c error:            {rms_c_error(c_est, c_true, mask):.3f} m/s")
    print(f"  Interior c range (true):         "
          f"{float(c_true[mask].min()):.0f}–{float(c_true[mask].max()):.0f} m/s")
    print(f"  Interior c range (reconstructed): "
          f"{float(c_est[mask].min()):.0f}–{float(c_est[mask].max()):.0f} m/s")
    if phantom == "disk":
        print(f"  Disk target:                     r = {1e3 * PHANTOM_RADIUS:.1f} mm, "
              f"c = {PHANTOM_C:.0f} m/s  (background {C0:.0f} m/s)")
    else:
        print(f"  Shepp–Logan target:              "
              f"c = {C0:.0f}–{C0 + SHEPP_DC:.0f} m/s")


def main():
    args = parse_args()
    if args.gradient_check:
        run_gradient_check()
        return
    if args.n_elements < 2:
        raise SystemExit("Need at least 2 ring elements.")
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be at least 1.")
    if args.c_min <= 0.0 or args.c_max <= args.c_min:
        raise SystemExit("Need 0 < --c-min < --c-max.")
    if args.margin_pixels < 0:
        raise SystemExit("--margin-pixels must be non-negative.")

    x_m = axis_centers(NX, DX)
    y_m = axis_centers(NY, DY)
    c_true = build_medium(x_m, y_m, phantom=args.phantom)
    c_ceiling = max(float(c_true.max()), float(args.c_max), C0)
    dt = stable_dt(c_ceiling, DX, DY, cfl=CFL)
    n_steps = n_steps_for_crossing(RING_RADIUS, float(c_true.min()), dt)
    pulse = linear_chirp(dt, CHIRP_DURATION, F_START_HZ, F_END_HZ)
    n_steps = max(n_steps, pulse.size + 1)

    model = ScalarWave2D(c_true, DX, DY)
    solver = FDTD2D(model, dt)
    array = RingArray(args.n_elements, RING_RADIUS, x_m, y_m)
    sources = source_indices(args.n_elements, args.n_shots)

    print("Generating observed traces from the true phantom...")
    observed = []
    for tx in sources:
        traces, _ = simulate_shot(solver, array, pulse, tx, n_steps)
        observed.append(traces)
        print(f"  shot TX {tx}: peak |p| = {np.max(np.abs(traces)):.4g}")

    mask = interior_mask(
        x_m, y_m, RING_RADIUS, margin_m=args.margin_pixels * DX,
        center=array.center,
    )
    problem = LeastSquaresFWI(
        solver, array, pulse, n_steps, sources, observed, mask,
        alpha=args.reg, c_min=args.c_min, c_max=args.c_max, c_background=C0,
    )
    m0 = np.full(c_true.shape, m_from_c(C0))
    print("Running Polak–Ribière CG...")
    m_est, history = polak_ribiere(
        m0,
        problem.misfit,
        problem.misfit_and_grad,
        problem.project,
        mask,
        max_iter=args.max_iter,
        verbose=True,
    )
    c_est = c_from_m(m_est)
    print_summary(c_true, c_est, mask, history, sources, args.phantom, dt, n_steps)

    if not args.no_show or args.save_figure:
        figure = plot_inversion(
            c_true, c_est, history, x_m, y_m, array, mask=mask,
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
