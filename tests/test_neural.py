"""Tests for the neural sound-speed parameterization (fdtd2d.inversion.neural)."""

import numpy as np
import pytest
import torch

from fdtd2d import ArcArray, RingArray, axis_centers, n_steps_for_crossing, stable_dt
from fdtd2d.inversion import interior_mask, m_from_c
from fdtd2d.inversion.neural import (
    NeuralFWIObjective,
    NeuralSpeedModel,
    SoundSpeedNet,
    theta_gradient_check,
)
from fdtd2d.torch_backend import (
    TorchFDTD2D,
    TorchRingSampler,
    TorchScalarWave2D,
    linear_chirp,
)

C_MIN, C_MAX, C_BACKGROUND = 1400.0, 2000.0, 1500.0
DTYPE = torch.float64


def make_model(x_m, y_m, mask, width=32, depth=2, fourier_scale=2.0, output_std=None, seed=0):
    net = SoundSpeedNet(n_fourier=16, fourier_scale=fourier_scale, width=width,
                        depth=depth, seed=seed).to(DTYPE)
    if output_std is not None:
        # Leave the zero start so every layer, not just the output, is exercised.
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            net.output.weight.copy_(output_std * torch.randn(net.output.weight.shape,
                                                             generator=generator, dtype=DTYPE))
    model = NeuralSpeedModel(net, C_MIN, C_MAX, C_BACKGROUND, length_scale_m=0.07)
    return model.set_grid(x_m, y_m, mask)


def small_grid(n=41, spacing=2.0e-3, radius=0.030, margin_pixels=2):
    x_m = axis_centers(n, spacing)
    y_m = axis_centers(n, spacing)
    mask = interior_mask(x_m, y_m, radius, margin_m=margin_pixels * spacing)
    return x_m, y_m, mask


# --------------------------------------------------------------------------
# Phase 1: parameterization
# --------------------------------------------------------------------------

def test_zero_initialization_is_background():
    x_m, y_m, mask = small_grid()
    c = make_model(x_m, y_m, mask).speed().detach().numpy()
    np.testing.assert_allclose(c, C_BACKGROUND, rtol=0, atol=1e-9)


def test_speed_is_bounded_and_background_outside_mask():
    x_m, y_m, mask = small_grid()
    model = make_model(x_m, y_m, mask, output_std=50.0)  # drives the sigmoid hard
    c = model.speed().detach().numpy()
    assert np.all(c[mask] >= C_MIN) and np.all(c[mask] <= C_MAX)
    assert c[mask].min() < 1450.0 and c[mask].max() > 1900.0  # both bounds are reached
    np.testing.assert_array_equal(c[~mask], C_BACKGROUND)


def test_slowness_matches_speed():
    x_m, y_m, mask = small_grid()
    model = make_model(x_m, y_m, mask, output_std=0.5)
    c = model.speed().detach().numpy()
    np.testing.assert_allclose(model.slowness().detach().numpy(), m_from_c(c), rtol=1e-12)


def test_same_network_on_coarse_and_fine_grids():
    # 101 x 2 mm and 401 x 0.5 mm grids share every fourth fine-grid point.
    coarse = axis_centers(101, 2.0e-3)
    fine = axis_centers(401, 0.5e-3)
    np.testing.assert_allclose(fine[::4], coarse, atol=1e-15)
    model = make_model(coarse, coarse, interior_mask(coarse, coarse, 0.07, 0.0), output_std=0.5)
    c_coarse = model.speed().detach().numpy()
    model.set_grid(fine, fine, interior_mask(fine, fine, 0.07, 0.0))
    c_fine = model.speed().detach().numpy()
    np.testing.assert_allclose(c_fine[::4, ::4], c_coarse, rtol=0, atol=1e-9)


def test_fit_to_smooth_target():
    x_m, y_m, mask = small_grid(n=61, spacing=2.0e-3, radius=0.055)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    target = C_BACKGROUND + 150.0 * np.exp(-(x**2 + y**2) / (2 * 0.015**2))
    model = make_model(x_m, y_m, mask, width=64, depth=3)
    rms = model.fit_to(target, iterations=400, learning_rate=3e-3)
    assert rms < 5.0


def test_rejects_background_outside_bounds():
    net = SoundSpeedNet(n_fourier=4, width=4, depth=1)
    with pytest.raises(ValueError):
        NeuralSpeedModel(net, 1500.0, 2000.0, 1500.0, 0.07)


# --------------------------------------------------------------------------
# Phase 2: chain rule and theta-space gradient check
# --------------------------------------------------------------------------

class QuadraticProblem:
    """J(m) = 1/2 sum w (m - m_target)^2 with an analytic gradient."""

    def __init__(self, m_target, weights):
        self.m_target = m_target
        self.weights = weights

    def misfit(self, m):
        return 0.5 * float(np.sum(self.weights * (m - self.m_target) ** 2))

    def misfit_and_grad(self, m):
        return self.misfit(m), self.weights * (m - self.m_target)


def test_objective_normalizes_first_loss_to_one():
    x_m, y_m, mask = small_grid()
    target = np.full(mask.shape, m_from_c(C_BACKGROUND))
    target[mask] = m_from_c(1600.0)
    objective = NeuralFWIObjective(QuadraticProblem(target, 1e14), make_model(x_m, y_m, mask))
    loss, value = objective.loss_and_backward()
    assert loss == pytest.approx(1.0)
    assert value > 0.0


def test_theta_gradient_quadratic_problem():
    x_m, y_m, mask = small_grid()
    rng = np.random.default_rng(0)
    target = m_from_c(np.where(mask, 1500.0 + 100.0 * rng.standard_normal(mask.shape), 1500.0))
    problem = QuadraticProblem(target, 1e14 * rng.uniform(0.5, 1.5, mask.shape))
    objective = NeuralFWIObjective(problem, make_model(x_m, y_m, mask, output_std=0.3))
    results = theta_gradient_check(objective, epsilons=(1e-3, 1e-4, 1e-5), verbose=False)
    assert min(error for _, _, error in results) < 1e-7


def tiny_fdtd_problem(geometry):
    """41 x 41 ring or arc problem with every shot active (deterministic)."""
    from invert_arc import ArcLeastSquaresFWI, simulate_observed
    from invert_ring import TorchLeastSquaresFWI

    radius = 0.030
    x_m, y_m, mask = small_grid(radius=radius)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c_true = np.full(mask.shape, C_BACKGROUND)
    c_true[x**2 + y**2 <= 0.012**2] = 1800.0
    dt = stable_dt(C_MAX, 2.0e-3, 2.0e-3, cfl=0.45)
    pulse = linear_chirp(dt, 20.0e-6, 1.0e5, 2.5e5, device="cpu", dtype=DTYPE)
    n_steps = max(n_steps_for_crossing(radius, C_BACKGROUND, dt), pulse.numel() + 1)
    source_values = torch.zeros(n_steps, dtype=DTYPE)
    source_values[:pulse.numel()] = pulse
    solver = TorchFDTD2D(TorchScalarWave2D(c_true, 2.0e-3, device="cpu", dtype=DTYPE), dt)
    if geometry == "ring":
        tx_array = RingArray(4, radius, x_m, y_m)
        rx_array = tx_array
        problem_class = TorchLeastSquaresFWI
    else:
        tx_array = ArcArray(3, radius, 60.0, 270.0, x_m, y_m)
        rx_array = ArcArray(5, radius, 60.0, 90.0, x_m, y_m)
        problem_class = ArcLeastSquaresFWI
    sampler = TorchRingSampler(rx_array, device="cpu", dtype=DTYPE)
    observed = simulate_observed(solver, sampler, tx_array, pulse, source_values, n_steps,
                                 parallel_shots=tx_array.n_elements, batched_solvers={})
    problem = problem_class(
        solver, sampler, tx_array, pulse, n_steps, tx_array.n_elements, observed, mask,
        c_min=C_MIN, c_max=C_MAX, c_background=C_BACKGROUND,
    )
    return problem, x_m, y_m, mask


@pytest.mark.parametrize("geometry", ["ring", "arc"])
def test_theta_gradient_fdtd_adjoint(geometry):
    problem, x_m, y_m, mask = tiny_fdtd_problem(geometry)
    objective = NeuralFWIObjective(problem, make_model(x_m, y_m, mask, output_std=0.3))
    results = theta_gradient_check(objective, epsilons=(1e-2, 1e-3, 1e-4), verbose=True)
    errors = [error for _, _, error in results]
    assert min(errors) < 1e-4
    # Central differences: error falls ~100x per 10x smaller step until round-off.
    assert errors[1] < 0.05 * errors[0]


# --------------------------------------------------------------------------
# Phase 3: optimizer loop shared by invert_ring.py and invert_arc.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("optimizer_name", ["adam", "gradient-descent"])
def test_optimize_neural_reduces_misfit(optimizer_name):
    from fdtd2d.inversion.neural import optimize_neural

    x_m, y_m, mask = small_grid()
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    target = m_from_c(np.where(mask, 1500.0 + 100.0 * np.exp(-(x**2 + y**2) / 0.01**2 / 2),
                               1500.0))
    objective = NeuralFWIObjective(QuadraticProblem(target, 1e14), make_model(x_m, y_m, mask))
    seen = []
    learning_rate = 3e-3 if optimizer_name == "adam" else 1e-1
    m, history = optimize_neural(objective, optimizer_name, max_iter=20,
                                 learning_rate=learning_rate,
                                 on_evaluate=lambda m, value: seen.append(value),
                                 verbose=False)
    assert len(history["misfit"]) == 21 == len(seen)
    assert history["misfit"][-1] < 0.5 * history["misfit"][0]
    np.testing.assert_allclose(m, objective.slowness_numpy())
