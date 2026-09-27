"""GPU-first frequency-domain sound-speed FWI for the ring array.

This is deliberately separate from :mod:`invert_ring`: it solves the steady
state, constant-density Helmholtz equation at one or more frequencies instead
of advancing the FDTD leapfrog scheme in time.  The physical geometry matches
the time-domain example (200 mm square field of view and a 70 mm ring).

The implementation is matrix-free.  On CUDA it keeps all fields, the finite
difference stencil, receiver sampling, and independent transmitter solves on
the GPU.  That is much more practical on an A6000 than assembling a dense
``(N**2) x (N**2)`` Helmholtz matrix.  Each right hand side is solved by a
Jacobi-preconditioned restarted complex GMRES iteration; transmitter batches expose
the useful parallelism to the GPU.

The convention is ``p(x,t) = Re[p(x) exp(-i omega t)]`` and

    (laplacian + omega**2 (1 + i*eta) / c**2) p = q.

``eta`` is a fixed quadratic absorbing layer near the outer boundary.  It is
not an estimate: only squared slowness inside the ring is updated.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Permit the documented ``python fdfd_ring/invert_ring_fdfd.py`` invocation.
# Python otherwise inserts only ``fdfd_ring/`` (not the repository root) into
# sys.path, so the shared fdtd2d geometry package would not be importable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fdtd2d import RingArray, axis_centers, load_ct_ring_grid
from fdtd2d.inversion import interior_mask
from fdtd2d.phantoms import shepp_logan_speed
from fdtd2d.torch_backend import configure_runtime, device_summary, resolve_device


# Same default physical geometry as invert_ring.py.
DOMAIN_WIDTH_M = 0.200
RING_RADIUS_M = 0.070
BACKGROUND_C = 1500.0
DISK_RADIUS_M = 0.020
DISK_C = 1800.0


def parse_frequencies(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) * 1e3 for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frequencies must be comma-separated kHz") from exc
    if not result or any(not math.isfinite(f) or f <= 0 for f in result):
        raise argparse.ArgumentTypeError("frequencies must be positive kHz values")
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    medium = parser.add_mutually_exclusive_group()
    medium.add_argument("--phantom", choices=("disk", "shepp-logan"), default="disk")
    medium.add_argument("--ct-speed", type=Path, metavar="PATH")
    parser.add_argument("--grid-size", type=int, default=161,
                        help="Square FDFD grid; 161 keeps the 200 mm FOV at 1.25 mm.")
    parser.add_argument("--n-elements", type=int, default=128)
    parser.add_argument("--n-shots", type=int, default=24,
                        help="Evenly spaced transmitters used for data and inversion.")
    parser.add_argument("--shot-batch", type=int, default=8,
                        help="Independent transmitter systems solved concurrently on the GPU.")
    parser.add_argument("--frequencies-khz", type=parse_frequencies, default=(50e3, 75e3, 100e3),
                        help="Comma-separated steady-state frequencies, e.g. 50,75,100.")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam",
                        help="Projected optimizer; lbfgs uses torch.optim.LBFGS on the selected device.")
    parser.add_argument("--adam-lr", type=float, default=0.005,
                        help="Adam rate for dimensionless relative squared slowness.")
    parser.add_argument("--lbfgs-lr", type=float, default=0.8,
                        help="Initial step scale for projected L-BFGS (default: 0.8).")
    parser.add_argument("--lbfgs-history-size", type=int, default=10,
                        help="Number of GPU-resident curvature pairs retained by L-BFGS.")
    parser.add_argument("--lbfgs-max-eval", type=int, default=8,
                        help="Maximum objective evaluations per strong-Wolfe line search.")
    parser.add_argument("--reg", type=float, default=2e-3,
                        help="Smoothness weight on dimensionless relative squared slowness.")
    parser.add_argument("--c-min", type=float, default=1400.0)
    parser.add_argument("--c-max", type=float, default=2000.0)
    parser.add_argument("--pml-width-mm", type=float, default=18.0)
    parser.add_argument("--pml-strength", type=float, default=2.5)
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-maxiter", type=int, default=500)
    parser.add_argument("--solver-restart", type=int, default=40,
                        help="Krylov vectors per restarted GMRES cycle.")
    parser.add_argument("--margin-pixels", type=int, default=2)
    parser.add_argument("--ct-padding-speed", type=float, default=1480.0)
    parser.add_argument("--ring-clearance-mm", type=float, default=10.0)
    parser.add_argument("--ct-edge-margin", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="Permit TF32 where PyTorch uses real matrix kernels (complex fields remain FP32).")
    parser.add_argument("--compile", choices=("auto", "on", "off"), default="auto",
                        help="Compile the stencil on CUDA when possible.")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--save-figure", type=Path, default=Path("fdfd_ring_dashboard.png"))
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def build_phantom(x_m, y_m, kind):
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full(x.shape, BACKGROUND_C)
    if kind == "disk":
        c[x * x + y * y <= DISK_RADIUS_M**2] = DISK_C
    else:
        c = shepp_logan_speed(x, y, RING_RADIUS_M, BACKGROUND_C)
    return c


def choose_shots(n_elements, n_shots):
    if not 1 <= n_shots <= n_elements:
        raise ValueError("--n-shots must be in [1, --n-elements].")
    # Unique, evenly distributed views are deterministic and better than a
    # random subset for a reproducible first FDFD reconstruction.
    return np.floor(np.arange(n_shots) * n_elements / n_shots).astype(np.int64)


@dataclass
class RingReceiver:
    """GPU implementation of RingArray's bilinear sampling and its adjoint."""

    array: RingArray
    device: torch.device

    def __post_init__(self):
        self.r0 = torch.as_tensor(self.array._row0, device=self.device, dtype=torch.long)
        self.r1 = torch.as_tensor(self.array._row1, device=self.device, dtype=torch.long)
        self.c0 = torch.as_tensor(self.array._col0, device=self.device, dtype=torch.long)
        self.c1 = torch.as_tensor(self.array._col1, device=self.device, dtype=torch.long)
        # RingArray stores NumPy weights as float64.  Explicit float32 avoids
        # silently promoting CUDA complex64 wavefields to complex128.
        self.w00 = torch.as_tensor(self.array._w00, device=self.device, dtype=torch.float32)
        self.w10 = torch.as_tensor(self.array._w10, device=self.device, dtype=torch.float32)
        self.w01 = torch.as_tensor(self.array._w01, device=self.device, dtype=torch.float32)
        self.w11 = torch.as_tensor(self.array._w11, device=self.device, dtype=torch.float32)

    def sample(self, field):
        """Batched bilinear samples. field is (batch, y, x)."""
        return (field[:, self.r0, self.c0] * self.w00
                + field[:, self.r0, self.c1] * self.w10
                + field[:, self.r1, self.c0] * self.w01
                + field[:, self.r1, self.c1] * self.w11)

    def adjoint(self, values, shape):
        """Scatter receiver values; adjoint of :meth:`sample`."""
        out = torch.zeros((values.shape[0], *shape), device=values.device, dtype=values.dtype)
        for rows, cols, weight in ((self.r0, self.c0, self.w00), (self.r0, self.c1, self.w10),
                                   (self.r1, self.c0, self.w01), (self.r1, self.c1, self.w11)):
            out[:, rows, cols] += values * weight
        return out


class Helmholtz2D:
    """Matrix-free finite-difference Helmholtz operator and batched solver."""

    def __init__(self, shape, spacing, omega, sponge, device, compile_stencil=False):
        self.shape = tuple(shape)
        self.spacing = float(spacing)
        self.omega = float(omega)
        self.device = device
        self.real_dtype = torch.float32
        self.dtype = torch.complex64
        self.sponge = sponge.to(device=device, dtype=self.real_dtype)
        self.mass_scale = torch.complex(torch.ones_like(self.sponge), self.sponge)
        self.inv_h2 = 1.0 / self.spacing**2
        self.diag_lap = -4.0 * self.inv_h2
        self._matvec = self._matvec_impl
        if compile_stencil and hasattr(torch, "compile"):
            # The operation is entirely shape-static during an inversion.
            self._matvec = torch.compile(self._matvec_impl, mode="reduce-overhead")

    def set_model(self, m):
        self.mass = self.mass_scale * m.to(device=self.device, dtype=self.real_dtype)
        self.diag = self.diag_lap + self.omega**2 * self.mass
        self.inverse_diag = self.diag.reciprocal()

    def _matvec_impl(self, field, adjoint=False):
        padded = F.pad(field, (1, 1, 1, 1))
        lap = (padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
               + padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] - 4.0 * field) * self.inv_h2
        mass = self.mass.conj() if adjoint else self.mass
        return lap + self.omega**2 * mass * field

    def solve(self, rhs, tol, maxiter, adjoint=False, restart=40):
        """Independently solve RHS with preconditioned, restarted GMRES.

        Helmholtz systems are indefinite, so conventional conjugate gradient
        is not applicable.  Restarted GMRES is appreciably more robust than
        BiCGSTAB for this small-to-medium GPU FDFD problem while retaining the
        batched transmitter dimension.
        """
        x = torch.zeros_like(rhs)
        matvec = lambda z: self._matvec(z, adjoint)
        inv_diag = self.inverse_diag.conj() if adjoint else self.inverse_diag
        rhs_norm = torch.sqrt(dot(rhs, rhs).real).clamp_min(1e-20)
        identity_cache = {}
        iteration = 0
        residual = torch.full_like(rhs_norm, float("inf"))
        while iteration < maxiter:
            r = rhs - matvec(x)
            residual = torch.sqrt(dot(r, r).real.clamp_min(0.0)) / rhs_norm
            if not bool(torch.isfinite(residual).all()):
                break
            if bool((residual <= tol).all()):
                break
            k = min(int(restart), maxiter - iteration)
            # Left Jacobi preconditioning.  A vector list avoids allocating a
            # giant dense Helmholtz matrix; the only dense solve is k x k.
            preconditioned_r = inv_diag * r
            beta = torch.sqrt(dot(preconditioned_r, preconditioned_r).real.clamp_min(1e-30))
            basis = [preconditioned_r / beta[:, None, None]]
            hessenberg = torch.zeros((rhs.shape[0], k + 1, k), device=rhs.device, dtype=self.dtype)
            for column in range(k):
                vector = inv_diag * matvec(basis[column])
                for row in range(column + 1):
                    coefficient = dot(basis[row], vector)
                    hessenberg[:, row, column] = coefficient
                    vector = vector - coefficient[:, None, None] * basis[row]
                next_norm = torch.sqrt(dot(vector, vector).real.clamp_min(1e-30))
                hessenberg[:, column + 1, column] = next_norm
                basis.append(vector / next_norm[:, None, None])
            g = torch.zeros((rhs.shape[0], k + 1), device=rhs.device, dtype=self.dtype)
            g[:, 0] = beta
            # Solve the tiny normal equations.  The scale-aware diagonal term
            # guards a rare Arnoldi breakdown without perturbing resolved RHSs.
            h_dagger = hessenberg.mH
            normal = h_dagger @ hessenberg
            normal_scale = normal.abs().amax(dim=(-2, -1)).clamp_min(1.0)
            eye = identity_cache.get(k)
            if eye is None:
                eye = torch.eye(k, device=rhs.device, dtype=self.dtype)[None]
                identity_cache[k] = eye
            coefficients = torch.linalg.solve(
                normal + (1e-7 * normal_scale)[:, None, None] * eye,
                (h_dagger @ g[..., None]),
            )[:, :, 0]
            x = x + sum(coefficients[:, index, None, None] * basis[index] for index in range(k))
            iteration += k
        residual = torch.sqrt(dot(rhs - matvec(x), rhs - matvec(x)).real.clamp_min(0.0)) / rhs_norm
        return x, residual, iteration


def dot(a, b):
    return (a.conj() * b).sum(dim=(-2, -1))


def sponge_profile(shape, spacing, width_m, strength, device):
    """Quadratic imaginary mass layer, zero over the acquisition aperture."""
    ny, nx = shape
    y = torch.arange(ny, device=device) * spacing
    x = torch.arange(nx, device=device) * spacing
    distance = torch.minimum(torch.minimum(x, x[-1] - x)[None, :],
                             torch.minimum(y, y[-1] - y)[:, None])
    return strength * ((width_m - distance).clamp_min(0.0) / width_m).square()


def relative_smoothness(q, mask, weight):
    """Return 1/2 weight ||grad q||² and its gradient, restricted to mask."""
    if weight <= 0:
        return torch.zeros((), device=q.device), torch.zeros_like(q)
    grad = torch.zeros_like(q)
    both_x = mask[:, 1:] & mask[:, :-1]
    dx = q[:, 1:] - q[:, :-1]
    edge_x = torch.where(both_x, dx, torch.zeros_like(dx))
    grad[:, 1:] += weight * edge_x
    grad[:, :-1] -= weight * edge_x
    both_y = mask[1:, :] & mask[:-1, :]
    dy = q[1:, :] - q[:-1, :]
    edge_y = torch.where(both_y, dy, torch.zeros_like(dy))
    grad[1:, :] += weight * edge_y
    grad[:-1, :] -= weight * edge_y
    grad[~mask] = 0
    value = 0.5 * weight * ((edge_x.square()).sum() + (edge_y.square()).sum())
    return value, grad


class FrequencyDomainFWI:
    """Complex least-squares objective and hand-coded adjoint gradient."""

    def __init__(self, c_true, spacing, array, frequencies, shots, mask, args, device):
        self.c_true = torch.as_tensor(c_true, device=device, dtype=torch.float32)
        self.m_background = 1.0 / args.ct_padding_speed**2 if args.ct_speed else 1.0 / BACKGROUND_C**2
        self.spacing, self.array, self.frequencies, self.shots = spacing, array, frequencies, shots
        self.mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        self.args, self.device = args, device
        self.receivers = RingReceiver(array, device)
        self.shape = tuple(c_true.shape)
        sponge = sponge_profile(self.shape, spacing, args.pml_width_mm * 1e-3,
                                args.pml_strength, device)
        compiled = args.compile == "on" or (args.compile == "auto" and device.type == "cuda")
        self.operators = [Helmholtz2D(self.shape, spacing, 2 * math.pi * f, sponge,
                                      device, compile_stencil=compiled) for f in frequencies]
        self.source_rhs = self._sources(shots)
        self.observed = self._make_observed()

    def _sources(self, shots):
        out = torch.zeros((len(shots), *self.shape), device=self.device, dtype=torch.complex64)
        rows_cols = [self.array.inject_rows_cols(int(tx)) for tx in shots]
        rows = torch.as_tensor([p[0] for p in rows_cols], device=self.device)
        cols = torch.as_tensor([p[1] for p in rows_cols], device=self.device)
        # Grid-normalized monopole. Its absolute calibration cancels because
        # synthetic and predicted data use the same source definition.
        out[torch.arange(len(shots), device=self.device), rows, cols] = 1.0 / self.spacing**2
        return out

    def _make_observed(self):
        m_true = 1.0 / self.c_true.square()
        data = []
        with torch.inference_mode():
            for operator in self.operators:
                operator.set_model(m_true)
                fields, residual, _ = operator.solve(self.source_rhs, self.args.solver_tol,
                                                     self.args.solver_maxiter, restart=self.args.solver_restart)
                self._warn_solver("observed", residual)
                data.append(self.receivers.sample(fields).detach())
        return data

    def _warn_solver(self, name, residual):
        worst = float(residual.max().item())
        if worst > self.args.solver_tol * 10:
            print(f"  warning: {name} Helmholtz solve relative residual {worst:.2e}")

    def project(self, q):
        lower = (1.0 / self.args.c_max**2) / self.m_background - 1.0
        upper = (1.0 / self.args.c_min**2) / self.m_background - 1.0
        result = q.clamp(lower, upper).clone()
        result[~self.mask] = 0.0
        return result

    def speed(self, q):
        m = self.m_background * (1.0 + self.project(q))
        return torch.rsqrt(m)

    def value_and_gradient(self, q):
        q = self.project(q)
        m = self.m_background * (1.0 + q)
        gradient_m = torch.zeros_like(m)
        value = torch.zeros((), device=self.device)
        count = len(self.operators) * len(self.shots) * self.array.n_elements
        for frequency_index, operator in enumerate(self.operators):
            operator.set_model(m)
            for start in range(0, len(self.shots), self.args.shot_batch):
                stop = min(start + self.args.shot_batch, len(self.shots))
                fields, residual, _ = operator.solve(self.source_rhs[start:stop], self.args.solver_tol,
                                                     self.args.solver_maxiter, restart=self.args.solver_restart)
                self._warn_solver("forward", residual)
                prediction = self.receivers.sample(fields)
                data_residual = prediction - self.observed[frequency_index][start:stop]
                # Direct transmitter pressure is a poorly conditioned datum
                # for a point source, so omit that channel like invert_ring.
                data_residual[torch.arange(stop - start, device=self.device),
                              torch.as_tensor(self.shots[start:stop], device=self.device)] = 0
                value += 0.5 * (data_residual.abs().square().sum() / count)
                rhs_adjoint = self.receivers.adjoint(data_residual / count, self.shape)
                adjoint, residual, _ = operator.solve(rhs_adjoint, self.args.solver_tol,
                                                       self.args.solver_maxiter, adjoint=True,
                                                       restart=self.args.solver_restart)
                self._warn_solver("adjoint", residual)
                # δJ = Re[-lambdaᴴ (ω² mass_scale δm) p].
                contribution = -(adjoint.conj() * (operator.omega**2 * operator.mass_scale)
                                  * fields).sum(dim=0).real
                gradient_m += contribution
        regularization, regularization_gradient = relative_smoothness(q, self.mask, self.args.reg)
        value += regularization
        gradient_q = gradient_m * self.m_background + regularization_gradient
        gradient_q[~self.mask] = 0
        return value, gradient_q


def adam(problem, max_iter, learning_rate):
    q = torch.zeros(problem.shape, device=problem.device)
    first, second = torch.zeros_like(q), torch.zeros_like(q)
    history = []
    for iteration in range(max_iter + 1):
        value, gradient = problem.value_and_gradient(q)
        history.append(float(value.item()))
        speed = problem.speed(q)
        print(f"  iter {iteration:3d}  J = {history[-1]:.6e}  "
              f"|g| = {float(torch.linalg.vector_norm(gradient[problem.mask])):.3e}  "
              f"c = {float(speed[problem.mask].min()):.1f}–{float(speed[problem.mask].max()):.1f} m/s")
        if iteration == max_iter:
            break
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient.square()
        step = learning_rate * (first / (1.0 - 0.9**(iteration + 1))) / (
            torch.sqrt(second / (1.0 - 0.999**(iteration + 1))) + 1e-8)
        q = problem.project(q - step)
    return q, history


def projected_lbfgs(problem, max_iter, learning_rate, history_size, max_eval):
    """Projected L-BFGS using PyTorch's GPU optimizer and manual adjoint grads.

    PyTorch has ``torch.optim.LBFGS`` but no native L-BFGS-B implementation.
    The objective already evaluates its sound-speed bounds through
    :meth:`FrequencyDomainFWI.project`; after every accepted strong-Wolfe step
    this function explicitly projects the parameter back into that feasible
    set.  All L-BFGS history vectors, dot products, line-search trial points,
    and FDFD forward/adjoint work stay on ``problem.device``.
    """
    if history_size < 1 or max_eval < 1:
        raise ValueError("L-BFGS history size and maximum evaluations must be positive.")
    q = torch.nn.Parameter(torch.zeros(problem.shape, device=problem.device))
    optimizer = torch.optim.LBFGS(
        [q],
        lr=learning_rate,
        max_iter=1,  # One accepted quasi-Newton update per reported iteration.
        max_eval=max_eval,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    with torch.no_grad():
        value, gradient = problem.value_and_gradient(q)
    history = [float(value.item())]
    print(f"  iter {0:3d}  J = {history[-1]:.6e}  "
          f"|g| = {float(torch.linalg.vector_norm(gradient[problem.mask])):.3e}  "
          f"c = {float(problem.speed(q)[problem.mask].min()):.1f}–"
          f"{float(problem.speed(q)[problem.mask].max()):.1f} m/s")

    for iteration in range(1, max_iter + 1):
        def closure():
            # The adjoint gradient is explicitly derived, not an autograd
            # graph through GMRES; assign it to the GPU Parameter for LBFGS.
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                closure_value, closure_gradient = problem.value_and_gradient(q)
            q.grad = closure_gradient.detach().clone()
            return closure_value

        optimizer.step(closure)
        with torch.no_grad():
            q.copy_(problem.project(q))
            value, gradient = problem.value_and_gradient(q)
            speed = problem.speed(q)
        history.append(float(value.item()))
        print(f"  iter {iteration:3d}  J = {history[-1]:.6e}  "
              f"|g| = {float(torch.linalg.vector_norm(gradient[problem.mask])):.3e}  "
              f"c = {float(speed[problem.mask].min()):.1f}–{float(speed[problem.mask].max()):.1f} m/s")
    return q.detach(), history


def make_figure(c_true, c_est, history, x_m, y_m, array):
    dx, dy = x_m[1] - x_m[0], y_m[1] - y_m[0]
    extent = [1e3 * (x_m[0] - dx / 2), 1e3 * (x_m[-1] + dx / 2),
              1e3 * (y_m[0] - dy / 2), 1e3 * (y_m[-1] + dy / 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    vmin, vmax = min(float(c_true.min()), float(c_est.min())), max(float(c_true.max()), float(c_est.max()))
    for axis, image, title in ((axes[0], c_true, "True sound speed"),
                               (axes[1], c_est, "FDFD estimate")):
        im = axis.imshow(image, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.plot(array.x * 1e3, array.y * 1e3, ".", color="white", ms=1)
        axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=axis, label="m/s")
    axes[2].semilogy(history, "o-", color="tab:blue")
    axes[2].set(title="Complex data misfit", xlabel="Adam iteration", ylabel="J")
    axes[2].grid(alpha=0.3)
    return fig


def gradient_check(args, device):
    """Small directional test; catches sign/conjugation mistakes in the adjoint."""
    args.grid_size, args.n_elements, args.n_shots, args.shot_batch = 49, 16, 3, 3
    args.frequencies_khz, args.solver_maxiter, args.reg = (50e3,), 800, 0.0
    c, x_m, y_m, spacing, radius, background = medium_from_args(args)
    array = RingArray(args.n_elements, radius, x_m, y_m)
    mask = interior_mask(x_m, y_m, radius, args.margin_pixels * spacing)
    problem = FrequencyDomainFWI(c, spacing, array, args.frequencies_khz, choose_shots(args.n_elements, args.n_shots), mask, args, device)
    torch.manual_seed(0)
    q = problem.project(0.01 * torch.randn(problem.shape, device=device))
    direction = torch.zeros_like(q)
    direction[problem.mask] = torch.randn_like(direction[problem.mask])
    direction /= torch.linalg.vector_norm(direction[problem.mask])
    value, gradient = problem.value_and_gradient(q)
    predicted = (gradient * direction).sum().item()
    print("FDFD adjoint directional gradient check")
    print(f"  J={value.item():.6e}; g·d={predicted:.6e}")
    for epsilon in (2e-3, 1e-3, 5e-4):
        plus, _ = problem.value_and_gradient(problem.project(q + epsilon * direction))
        minus, _ = problem.value_and_gradient(problem.project(q - epsilon * direction))
        finite_difference = (plus - minus).item() / (2 * epsilon)
        error = abs(finite_difference - predicted) / max(abs(predicted), 1e-12)
        print(f"  eps={epsilon:.1e}  FD={finite_difference:.6e}  rel.err={error:.3e}")


def medium_from_args(args):
    if args.ct_speed:
        ct = load_ct_ring_grid(args.ct_speed, grid_shape=(args.grid_size, args.grid_size),
                               padding_speed_m_s=args.ct_padding_speed,
                               ring_clearance_mm=args.ring_clearance_mm,
                               edge_margin_pixels=args.ct_edge_margin)
        return ct.c, ct.x_m, ct.y_m, ct.spacing_m, ct.ring_radius_m, args.ct_padding_speed
    spacing = DOMAIN_WIDTH_M / (args.grid_size - 1)
    x_m = axis_centers(args.grid_size, spacing)
    y_m = axis_centers(args.grid_size, spacing)
    return build_phantom(x_m, y_m, args.phantom), x_m, y_m, spacing, RING_RADIUS_M, BACKGROUND_C


def main():
    args = arguments()
    if args.grid_size < 33 or args.grid_size % 2 == 0:
        raise SystemExit("--grid-size must be an odd integer of at least 33.")
    if args.n_elements < 2 or args.shot_batch < 1 or args.max_iter < 1:
        raise SystemExit("Need at least 2 elements, a positive batch size, and --max-iter >= 1.")
    if args.adam_lr <= 0 or args.lbfgs_lr <= 0 or args.lbfgs_history_size < 1 or args.lbfgs_max_eval < 1:
        raise SystemExit("Optimizer rates, L-BFGS history size, and L-BFGS max evaluations must be positive.")
    if args.c_min <= 0 or args.c_max <= args.c_min or args.pml_width_mm <= 0 or args.pml_strength <= 0:
        raise SystemExit("Invalid speed bounds or absorbing-layer settings.")
    try:
        device = resolve_device(args.device)
        configure_runtime(device, allow_tf32=args.allow_tf32)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if device.type == "mps":
        raise SystemExit("Complex FDFD is supported on CPU or CUDA; use --device cpu or cuda.")
    if args.gradient_check:
        gradient_check(args, device)
        return
    c_true, x_m, y_m, spacing, radius, background = medium_from_args(args)
    if radius + args.pml_width_mm * 1e-3 >= DOMAIN_WIDTH_M / 2 and not args.ct_speed:
        print("warning: the absorbing layer overlaps the ring; reduce --pml-width-mm.")
    array = RingArray(args.n_elements, radius, x_m, y_m)
    mask = interior_mask(x_m, y_m, radius, args.margin_pixels * spacing)
    shots = choose_shots(args.n_elements, args.n_shots)
    print("2D ring FDFD inversion")
    print(f"  device: {device_summary(device)}")
    print(f"  grid: {args.grid_size} x {args.grid_size}; spacing: {spacing * 1e3:.3f} mm")
    print(f"  ring: {radius * 1e3:.1f} mm; elements/shots/batch: {args.n_elements}/{args.n_shots}/{args.shot_batch}")
    print("  frequencies: " + ", ".join(f"{f / 1e3:g} kHz" for f in args.frequencies_khz))
    started = time.perf_counter()
    problem = FrequencyDomainFWI(c_true, spacing, array, args.frequencies_khz, shots, mask, args, device)
    print(f"  observed steady-state data: {time.perf_counter() - started:.2f} s")
    if args.optimizer == "adam":
        q, history = adam(problem, args.max_iter, args.adam_lr)
    else:
        q, history = projected_lbfgs(
            problem, args.max_iter, args.lbfgs_lr, args.lbfgs_history_size,
            args.lbfgs_max_eval,
        )
    c_est = problem.speed(q).detach().cpu().numpy()
    mask_np = mask
    rms = np.sqrt(np.mean((c_est[mask_np] - c_true[mask_np])**2))
    print(f"  completed in {time.perf_counter() - started:.2f} s; interior RMS error: {rms:.2f} m/s")
    figure = make_figure(c_true, c_est, history, x_m, y_m, array)
    args.save_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.save_figure, dpi=180)
    print(f"  saved dashboard: {args.save_figure}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
