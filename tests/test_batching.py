"""Tests for the fused batched FDTD steps, wavefield subsampling and batch sizing."""

import numpy as np
import pytest
import torch

from fdtd2d import RingArray, axis_centers, n_steps_for_crossing, stable_dt
from fdtd2d.inversion import interior_mask, m_from_c
from fdtd2d.torch_backend import (
    TorchFDTD2D,
    TorchRingSampler,
    TorchScalarWave2D,
    auto_batch_size,
    linear_chirp,
    simulate_shot,
    simulate_shots_batch,
)
from invert_ring import TorchLeastSquaresFWI, forward_store

DTYPE = torch.float64
N, SPACING, RADIUS = 41, 2.0e-3, 0.030


def ring_problem(n_elements=6, **options):
    x_m = axis_centers(N, SPACING)
    y_m = axis_centers(N, SPACING)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c_true = np.full((N, N), 1500.0)
    c_true[x**2 + y**2 <= 0.012**2] = 1800.0
    dt = stable_dt(2000.0, SPACING, SPACING, cfl=0.45)
    pulse = linear_chirp(dt, 20.0e-6, 1.0e5, 2.5e5, device="cpu", dtype=DTYPE)
    n_steps = max(n_steps_for_crossing(RADIUS, 1500.0, dt), pulse.numel() + 1)
    solver = TorchFDTD2D(TorchScalarWave2D(c_true, SPACING, device="cpu", dtype=DTYPE), dt)
    array = RingArray(n_elements, RADIUS, x_m, y_m)
    sampler = TorchRingSampler(array, device="cpu", dtype=DTYPE)
    observed = torch.stack([
        simulate_shot(solver, sampler, pulse, *array.inject_rows_cols(tx), n_steps)[0]
        for tx in range(n_elements)
    ])
    mask = interior_mask(x_m, y_m, RADIUS, margin_m=2 * SPACING)
    problem = TorchLeastSquaresFWI(
        solver, sampler, array, pulse, n_steps, n_elements, observed, mask,
        c_min=1400.0, c_max=2000.0, c_background=1500.0, **options,
    )
    rng = np.random.default_rng(0)
    m0 = np.full((N, N), m_from_c(1500.0))
    m0[mask] *= 1.0 + 0.02 * rng.standard_normal(int(mask.sum()))
    return problem, problem.project(m0)


def reference_gradient(problem, m):
    """Unbatched, unfused gradient over every transmitter."""
    m = problem.apply_medium(m)
    g_c2 = torch.zeros(problem.model.shape, dtype=DTYPE)
    for tx in range(problem.array.n_elements):
        traces, wave = forward_store(
            problem.solver, problem.sampler, problem.array, problem.pulse, tx,
            problem.n_steps,
        )
        residual = problem._residual(traces, problem.observed[tx], tx)
        problem._adjoint_accumulate(wave, residual, g_c2)
    gradient = -g_c2.numpy() / (m * m)
    gradient[~problem.mask] = 0.0
    return gradient


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_batched_traces_match_single_shots():
    problem, _ = ring_problem()
    rows, cols = problem.array.rows, problem.array.cols
    traces, _ = simulate_shots_batch(
        problem.solver, problem.sampler, problem.pulse, rows, cols, problem.n_steps,
    )
    np.testing.assert_allclose(traces.numpy(), problem.observed.numpy(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("parallel_shots", [6, 4, 1])
def test_fused_gradient_matches_reference(parallel_shots):
    problem, m0 = ring_problem(parallel_shots=parallel_shots)
    _, gradient = problem.misfit_and_grad(m0)
    assert relative_error(gradient, reference_gradient(problem, m0)) < 1e-12


@pytest.mark.parametrize("options, tolerance", [
    ({"store_every": 2}, 1e-6),
    ({"store_every": 4}, 1e-5),
    ({"wave_dtype": torch.float32}, 1e-6),
    ({"wave_dtype": torch.float16}, 1e-3),
])
def test_reduced_storage_gradient_is_close(options, tolerance):
    exact, m0 = ring_problem()
    reduced, _ = ring_problem(**options)
    value, gradient = reduced.misfit_and_grad(m0)
    exact_value, exact_gradient = exact.misfit_and_grad(m0)
    assert value == pytest.approx(exact_value, rel=1e-12)  # the misfit is never approximated
    error = relative_error(gradient, exact_gradient)
    print(f"{options}: relative gradient error {error:.2e}")
    assert error < tolerance


def test_auto_batch_size_splits_evenly_on_cpu():
    assert auto_batch_size("cpu", 64, 1) == 16
    assert auto_batch_size("cpu", 20, 1) == 10  # two batches of 10, not 16 + 4
    assert auto_batch_size("cpu", 5, 1) == 5


def test_auto_parallel_shots_option():
    problem, _ = ring_problem(parallel_shots="auto")
    assert problem.parallel_shots == 6  # CPU fallback of 16, capped at the shot count
