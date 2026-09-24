"""Directional finite-difference check of the discrete adjoint gradient.

Uses a tiny ring-array problem so the test finishes in seconds. Mur ABC is
not the exact discrete adjoint of itself, so the relative error should drop
toward a small floor rather than machine zero.
"""

import numpy as np

from ..array import RingArray
from ..experiment import (
    axis_centers,
    n_steps_for_crossing,
    simulate_shot,
    stable_dt,
)
from ..solver import FDTD2D
from ..sources import linear_chirp
from ..wave import ScalarWave2D
from .cg import masked_dot
from .gradient import LeastSquaresFWI
from .medium import interior_mask, m_from_c


def _tiny_problem(seed=0):
    nx = ny = 41
    dx = dy = 2.0e-3
    c0 = 1500.0
    c_disk = 1800.0
    ring_radius = 0.030
    disk_radius = 0.012
    cfl = 0.45
    x_m = axis_centers(nx, dx)
    y_m = axis_centers(ny, dy)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c_true = np.full((ny, nx), c0)
    c_true[x**2 + y**2 <= disk_radius**2] = c_disk
    dt = stable_dt(max(float(c_true.max()), 2000.0), dx, dy, cfl=cfl)
    pulse = linear_chirp(dt, 20.0e-6, 1.00e5, 2.50e5)
    n_steps = n_steps_for_crossing(ring_radius, float(c_true.min()), dt)
    n_steps = max(n_steps, pulse.size + 1)
    model = ScalarWave2D(c_true, dx, dy)
    solver = FDTD2D(model, dt)
    array = RingArray(2, ring_radius, x_m, y_m)
    sources = [0]
    observed = [
        simulate_shot(solver, array, pulse, tx, n_steps)[0] for tx in sources
    ]
    mask = interior_mask(x_m, y_m, ring_radius, margin_m=2.0 * dx)
    problem = LeastSquaresFWI(
        solver, array, pulse, n_steps, sources, observed, mask,
        alpha=0.0, c_min=1400.0, c_max=2000.0, c_background=c0,
    )
    rng = np.random.default_rng(seed)
    m0 = np.full((ny, nx), m_from_c(c0))
    # Perturb the start so the residual (and gradient) are nonzero.
    m0[mask] *= 1.0 + 0.02 * rng.standard_normal(int(mask.sum()))
    m0 = problem.project(m0)
    return problem, m0, mask, rng


def run_gradient_check(seed=0, epsilons=None):
    """Compare g·δm against a central difference of J.

    Returns a list of (eps, directional, finite_difference, relative_error).
    """
    if epsilons is None:
        epsilons = (1e-2, 1e-3, 1e-4, 1e-5)
    problem, m0, mask, rng = _tiny_problem(seed=seed)
    J, g = problem.misfit_and_grad(m0)
    delta = np.zeros_like(m0)
    delta[mask] = rng.standard_normal(int(mask.sum()))
    m_norm = np.sqrt(masked_dot(m0, m0, mask))
    d_norm = np.sqrt(masked_dot(delta, delta, mask))
    if d_norm <= 0.0 or m_norm <= 0.0:
        raise RuntimeError("Random direction or m was zero on the interior mask.")
    # Scale δm to ||m|| so eps is a relative perturbation of slowness-squared.
    delta *= m_norm / d_norm
    directional = masked_dot(g, delta, mask)
    rows = []
    print("Adjoint gradient check (tiny ring FWI)")
    print(f"  J = {J:.6e}")
    print(f"  g · δm = {directional:.6e}")
    print(f"  {'eps':>10}  {'FD':>14}  {'rel. error':>12}")
    for eps in epsilons:
        Jp = problem.misfit(problem.project(m0 + eps * delta))
        Jm = problem.misfit(problem.project(m0 - eps * delta))
        fd = (Jp - Jm) / (2.0 * eps)
        rel = abs(fd - directional) / max(abs(directional), 1e-30)
        rows.append((float(eps), float(directional), float(fd), float(rel)))
        print(f"  {eps:10.1e}  {fd:14.6e}  {rel:12.4e}")
    return rows


if __name__ == "__main__":
    run_gradient_check()
