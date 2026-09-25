"""Nonlinear FWI for sound speed inside the ring-array FDTD model.

Observed traces are generated from a known phantom with the same forward
operator used by 2DRingFDTD.py. The unknown is slowness-squared m = 1/c^2
on pixels inside the ring; outside, c stays at the water background.
Polak–Ribière CG or fixed-step gradient descent uses the adjoint-state
gradient of the leapfrog scheme.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from fdtd2d import (
    RingArray,
    axis_centers,
    n_steps_for_crossing,
    stable_dt,
)
from fdtd2d.inversion import (
    c_from_m,
    interior_mask,
    m_from_c,
    polak_ribiere,
    tikhonov,
)
from fdtd2d.phantoms import SHEPP_DC, shepp_logan_speed
from fdtd2d.plot import plot_inversion
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
        description="2D ring FWI on slowness-squared with a PyTorch adjoint",
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
        help="Random transmitters per gradient evaluation (default: 4)",
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
        choices=("cg", "gradient-descent"),
        default="cg",
        help="Optimization method (default: cg)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0e-15,
        metavar="ALPHA",
        help="Fixed gradient-descent step in slowness-squared units (default: 1e-15)",
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


def build_medium(x_m, y_m, phantom="disk"):
    """Return c(x, y) in m/s."""
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c = np.full_like(x, C0)
    if phantom == "disk":
        c[x**2 + y**2 <= PHANTOM_RADIUS**2] = PHANTOM_C
    elif phantom == "shepp-logan":
        c = shepp_logan_speed(x, y, RING_RADIUS, C0)
    return c


def validate_shot_count(n_elements, n_shots):
    if n_shots < 1:
        raise SystemExit("Need at least 1 shot.")
    if n_shots > n_elements:
        raise SystemExit("Cannot have more shots than ring elements.")


def rms_c_error(c_est, c_true, mask):
    err = c_est[mask] - c_true[mask]
    return float(np.sqrt(np.mean(err * err)))


def forward_store(solver, sampler, array, pulse, tx, n_steps):
    """Forward shot with the wavefield history retained on the torch device."""
    row, col = array.inject_rows_cols(tx)
    solver.reset()
    solver.set_source(row, col)
    traces = torch.empty(
        (array.n_elements, n_steps),
        device=solver.model.device,
        dtype=solver.model.dtype,
    )
    wave = torch.empty(
        (n_steps, *solver.model.shape),
        device=solver.model.device,
        dtype=solver.model.dtype,
    )
    zero = torch.zeros((), device=solver.model.device, dtype=solver.model.dtype)
    with torch.inference_mode():
        for step in range(n_steps):
            if step < pulse.numel():
                field = solver.inject_and_step(pulse[step])
            else:
                field = solver.step(zero)
            traces[:, step] = sampler.record(field)
            wave[step].copy_(field)
    return traces, wave


class TorchLeastSquaresFWI:
    """Least-squares FWI using one torch engine in forward and adjoint modes."""

    def __init__(
        self,
        solver,
        sampler,
        array,
        pulse,
        n_steps,
        n_shots,
        observed,
        mask,
        alpha=0.0,
        c_min=1400.0,
        c_max=2000.0,
        c_background=1500.0,
        shot_seed=0,
    ):
        self.solver = solver
        self.model = solver.model
        self.sampler = sampler
        self.array = array
        self.pulse = pulse
        self.n_steps = int(n_steps)
        self.n_shots = int(n_shots)
        if self.n_shots < 1 or self.n_shots > array.n_elements:
            raise ValueError("n_shots must be between 1 and the number of elements.")
        if len(observed) != array.n_elements:
            raise ValueError("Observed data must contain one gather per transmitter.")
        self.observed = list(observed)
        self._shot_rng = np.random.default_rng(shot_seed)
        self.active_sources = None
        self.source_history = []
        self.mask = np.asarray(mask, dtype=bool)
        self.alpha = float(alpha)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.c_background = float(c_background)
        self._adj_source = torch.zeros(
            self.model.shape, device=self.model.device, dtype=self.model.dtype
        )

    def project(self, m):
        m = np.asarray(m, dtype=float)
        m_min = 1.0 / self.c_max**2
        m_max = 1.0 / self.c_min**2
        projected = np.clip(m, m_min, m_max).copy()
        projected[~self.mask] = m_from_c(self.c_background)
        return projected

    def apply_medium(self, m):
        m = self.project(m)
        self.solver.set_c(c_from_m(m))
        return m

    @staticmethod
    def _residual(predicted, observed, tx):
        residual = predicted - observed
        residual = residual.clone()
        residual[tx].zero_()
        return residual

    def _sample_sources(self):
        if self.n_shots == self.array.n_elements:
            selected = np.arange(self.array.n_elements, dtype=int)
        else:
            selected = np.sort(
                self._shot_rng.choice(
                    self.array.n_elements, size=self.n_shots, replace=False
                )
            )
        self.active_sources = selected
        self.source_history.append(selected.copy())
        return selected

    def misfit(self, m):
        m = self.apply_medium(m)
        value = 0.0
        sources = self.active_sources
        if sources is None:
            sources = self._sample_sources()
        for tx in sources:
            data = self.observed[int(tx)]
            row, col = self.array.inject_rows_cols(tx)
            predicted, _ = simulate_shot(
                self.solver,
                self.sampler,
                self.pulse,
                row,
                col,
                self.n_steps,
            )
            residual = self._residual(predicted, data, tx)
            value += 0.5 * float(torch.sum(residual.square()).item())
        regularization, _ = tikhonov(m, self.mask, self.alpha)
        return value + regularization

    def misfit_and_grad(self, m):
        m = self.apply_medium(m)
        sources = self._sample_sources()
        value = 0.0
        g_c2 = torch.zeros(
            self.model.shape, device=self.model.device, dtype=self.model.dtype
        )
        for tx in sources:
            data = self.observed[int(tx)]
            predicted, wave = forward_store(
                self.solver,
                self.sampler,
                self.array,
                self.pulse,
                tx,
                self.n_steps,
            )
            residual = self._residual(predicted, data, tx)
            value += 0.5 * float(torch.sum(residual.square()).item())
            self._adjoint_accumulate(wave, residual, g_c2)
        regularization, g_regularization = tikhonov(m, self.mask, self.alpha)
        gradient = -g_c2.detach().cpu().numpy().astype(float) / (m * m)
        gradient += g_regularization
        gradient[~self.mask] = 0.0
        return value + regularization, gradient

    def _adjoint_accumulate(self, wave, residual, g_c2):
        self.solver.reset()
        dt2 = self.solver.dt2
        source = self._adj_source
        with torch.inference_mode():
            for n in range(self.n_steps - 1, -1, -1):
                self.sampler.record_adjoint(residual[:, n], out=source)
                adjoint_field = self.solver.inject_field_and_step(
                    source, adjoint=True
                )
                if n == 0:
                    continue
                laplacian = self.model.laplacian(wave[n - 1])
                g_c2.addcmul_(adjoint_field, laplacian, value=dt2)


def run_torch_gradient_check(device, dtype, compile_step, seed=0):
    """Directional finite-difference check of the PyTorch adjoint mode."""
    n = 41
    spacing = 2.0e-3
    ring_radius = 0.030
    x_m = axis_centers(n, spacing)
    y_m = axis_centers(n, spacing)
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    c_true = np.full((n, n), C0)
    c_true[x * x + y * y <= 0.012**2] = PHANTOM_C
    dt = stable_dt(2000.0, spacing, spacing, cfl=CFL)
    n_steps = n_steps_for_crossing(ring_radius, float(c_true.min()), dt)
    pulse = linear_chirp(
        dt,
        CHIRP_DURATION,
        F_START_HZ,
        F_END_HZ,
        device=device,
        dtype=dtype,
    )
    n_steps = max(n_steps, pulse.numel() + 1)
    model = TorchScalarWave2D(
        c_true, spacing, spacing, device=device, dtype=dtype
    )
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    array = RingArray(2, ring_radius, x_m, y_m)
    sampler = TorchRingSampler(array, device=device, dtype=dtype)
    observed = []
    for tx in range(array.n_elements):
        row, col = array.inject_rows_cols(tx)
        traces, _ = simulate_shot(
            solver, sampler, pulse, row, col, n_steps
        )
        observed.append(traces.detach().clone())
    mask = interior_mask(
        x_m, y_m, ring_radius, margin_m=2.0 * spacing
    )
    problem = TorchLeastSquaresFWI(
        solver,
        sampler,
        array,
        pulse,
        n_steps,
        1,
        observed,
        mask,
        c_min=1400.0,
        c_max=2000.0,
        c_background=C0,
        shot_seed=seed,
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
    print("PyTorch adjoint gradient check")
    print(f"  Device: {device_summary(device)}")
    print(f"  J = {value:.6e}")
    print(f"  g · δm = {directional:.6e}")
    print(f"  {'eps':>10}  {'FD':>14}  {'rel. error':>12}")
    for epsilon in (1e-2, 1e-3, 1e-4):
        plus = problem.misfit(problem.project(m0 + epsilon * direction))
        minus = problem.misfit(problem.project(m0 - epsilon * direction))
        finite_difference = (plus - minus) / (2.0 * epsilon)
        relative_error = abs(finite_difference - directional) / max(
            abs(directional), 1e-30
        )
        print(
            f"  {epsilon:10.1e}  {finite_difference:14.6e}  "
            f"{relative_error:12.4e}"
        )


def fixed_step_gradient_descent(
    m0,
    misfit_and_grad,
    project,
    mask,
    step_size,
    max_iter=15,
    verbose=True,
):
    """Projected gradient descent with one fixed step for every iteration."""
    m = project(np.asarray(m0, dtype=float))
    value, gradient = misfit_and_grad(m)
    grad_norm = float(np.linalg.norm(gradient[mask]))
    history = {
        "misfit": [float(value)],
        "step_size": [0.0],
        "grad_norm": [grad_norm],
    }
    for iteration in range(max_iter):
        if grad_norm <= 0.0:
            break
        m = project(m - step_size * gradient)
        value, gradient = misfit_and_grad(m)
        grad_norm = float(np.linalg.norm(gradient[mask]))
        history["misfit"].append(float(value))
        history["step_size"].append(float(step_size))
        history["grad_norm"].append(grad_norm)
        if verbose:
            print(
                f"  GD iter {iteration + 1:3d}  J = {value:.6e}  "
                f"step = {step_size:.3e}  |g| = {grad_norm:.4e}"
            )
    return m, history


class IterationVisualizer:
    """Live speed estimate, per-iteration update, and misfit history."""

    def __init__(self, x_m, y_m, mask, c_min, c_max):
        dx = x_m[1] - x_m[0]
        dy = y_m[1] - y_m[0]
        extent = [
            1e3 * (x_m[0] - 0.5 * dx),
            1e3 * (x_m[-1] + 0.5 * dx),
            1e3 * (y_m[0] - 0.5 * dy),
            1e3 * (y_m[-1] + 0.5 * dy),
        ]
        self.mask = mask
        self.previous = None
        self.misfit = []
        plt.ion()
        self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 4.8))
        self.estimate_image = self.axes[0].imshow(
            np.full(mask.shape, C0),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=c_min,
            vmax=c_max,
        )
        self.update_image = self.axes[1].imshow(
            np.zeros(mask.shape),
            origin="lower",
            extent=extent,
            cmap="seismic",
            vmin=-1.0,
            vmax=1.0,
        )
        self.misfit_line, = self.axes[2].plot([], [], "o-", color="tab:blue")
        self.axes[0].set(title="Speed estimate", xlabel="x (mm)", ylabel="y (mm)")
        self.axes[1].set(title="Iteration update", xlabel="x (mm)", ylabel="y (mm)")
        self.axes[2].set(title="Waveform misfit", xlabel="Iteration", ylabel="J(m)")
        self.axes[2].grid(alpha=0.3)
        self.figure.colorbar(
            self.estimate_image, ax=self.axes[0], label="Speed (m/s)"
        )
        self.update_colorbar = self.figure.colorbar(
            self.update_image, ax=self.axes[1], label="Δ speed (m/s)"
        )
        self.figure.tight_layout()
        self.figure.show()

    def update(self, m, value):
        speed = c_from_m(m)
        if self.previous is None:
            change = np.zeros_like(speed)
        else:
            change = speed - self.previous
        change = np.where(self.mask, change, 0.0)
        limit = max(float(np.max(np.abs(change[self.mask]))), 1.0e-6)
        self.estimate_image.set_data(speed)
        self.update_image.set_data(change)
        self.update_image.set_clim(-limit, limit)
        self.update_colorbar.update_normal(self.update_image)
        self.misfit.append(float(value))
        iterations = np.arange(len(self.misfit))
        self.misfit_line.set_data(iterations, self.misfit)
        self.axes[2].relim()
        self.axes[2].autoscale_view()
        if np.all(np.asarray(self.misfit) > 0.0):
            self.axes[2].set_yscale("log")
        self.figure.suptitle(f"FWI iteration {len(self.misfit) - 1}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)
        self.previous = speed.copy()


def print_summary(
    c_true,
    c_est,
    mask,
    history,
    n_elements,
    n_shots,
    source_history,
    shot_seed,
    phantom,
    dt,
    n_steps,
    execution_device,
    dtype_name,
    compiled,
    optimizer,
    step_size,
):
    print("2D ring FWI results")
    print(f"  Phantom:                         {phantom}")
    print(f"  Precomputed observed shots:      all {n_elements} transmitters")
    print(f"  Gradient shots per evaluation:   {n_shots}")
    print(f"  Random shot seed:                {shot_seed}")
    if source_history:
        last_sources = [int(tx) for tx in source_history[-1]]
        print(f"  Last gradient transmitters:      {last_sources}")
    print(f"  Time steps:                      {n_steps}")
    print(f"  dt:                              {dt:.4g} s")
    print(f"  PyTorch device:                  {execution_device}")
    print(f"  Precision / compiled step:       {dtype_name} / {compiled}")
    print(f"  Optimizer:                       {optimizer}")
    if optimizer == "gradient-descent":
        print(f"  Fixed step size:                 {step_size:.4g}")
    print(f"  Optimization iterations:         {max(len(history['misfit']) - 1, 0)}")
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
    if args.n_elements < 2:
        raise SystemExit("Need at least 2 ring elements.")
    validate_shot_count(args.n_elements, args.n_shots)
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be at least 1.")
    if args.step_size <= 0.0:
        raise SystemExit("--step-size must be positive.")
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
        configure_runtime(
            device,
            cpu_threads=args.cpu_threads,
            allow_tf32=args.allow_tf32,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not configure PyTorch: {exc}") from exc
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise SystemExit("Apple MPS does not support this inversion in float64.")
    compile_step = args.compile == "on" or (
        args.compile == "auto" and device.type == "cuda"
    )
    if args.gradient_check:
        run_torch_gradient_check(device, dtype, compile_step)
        return

    x_m = axis_centers(NX, DX)
    y_m = axis_centers(NY, DY)
    c_true = build_medium(x_m, y_m, phantom=args.phantom)
    c_ceiling = max(float(c_true.max()), float(args.c_max), C0)
    dt = stable_dt(c_ceiling, DX, DY, cfl=CFL)
    n_steps = n_steps_for_crossing(RING_RADIUS, float(c_true.min()), dt)
    pulse = linear_chirp(
        dt,
        CHIRP_DURATION,
        F_START_HZ,
        F_END_HZ,
        device=device,
        dtype=dtype,
    )
    n_steps = max(n_steps, pulse.numel() + 1)

    model = TorchScalarWave2D(
        c_true, DX, DY, device=device, dtype=dtype
    )
    solver = TorchFDTD2D(model, dt, compile_step=compile_step)
    array = RingArray(args.n_elements, RING_RADIUS, x_m, y_m)
    sampler = TorchRingSampler(array, device=device, dtype=dtype)

    print("Precomputing observed traces for every transmitting element...")
    observed = []
    progress_stride = max(1, args.n_elements // 8)
    for tx in range(args.n_elements):
        row, col = array.inject_rows_cols(tx)
        traces, _ = simulate_shot(
            solver, sampler, pulse, row, col, n_steps
        )
        observed.append(traces.detach().clone())
        if (tx + 1) % progress_stride == 0 or tx + 1 == args.n_elements:
            print(
                f"  generated {tx + 1:4d}/{args.n_elements} shots; "
                f"latest peak |p| = {traces.abs().max().item():.4g}"
            )

    mask = interior_mask(
        x_m, y_m, RING_RADIUS, margin_m=args.margin_pixels * DX,
        center=array.center,
    )
    problem = TorchLeastSquaresFWI(
        solver, sampler, array, pulse, n_steps, args.n_shots, observed, mask,
        alpha=args.reg, c_min=args.c_min, c_max=args.c_max, c_background=C0,
        shot_seed=args.shot_seed,
    )
    m0 = np.full(c_true.shape, m_from_c(C0))
    visualizer = None
    if args.show_iterations:
        visualizer = IterationVisualizer(
            x_m, y_m, mask, args.c_min, args.c_max
        )

    def evaluated_objective(m):
        value, gradient = problem.misfit_and_grad(m)
        if visualizer is not None:
            visualizer.update(m, value)
        return value, gradient

    if args.optimizer == "gradient-descent":
        print(f"Running fixed-step gradient descent (step={args.step_size:.3e})...")
        m_est, history = fixed_step_gradient_descent(
            m0,
            evaluated_objective,
            problem.project,
            mask,
            step_size=args.step_size,
            max_iter=args.max_iter,
            verbose=True,
        )
    else:
        print("Running Polak–Ribière CG...")
        m_est, history = polak_ribiere(
            m0,
            problem.misfit,
            evaluated_objective,
            problem.project,
            mask,
            max_iter=args.max_iter,
            verbose=True,
        )
    if visualizer is not None:
        plt.ioff()
    c_est = c_from_m(m_est)
    print_summary(
        c_true,
        c_est,
        mask,
        history,
        args.n_elements,
        args.n_shots,
        problem.source_history,
        args.shot_seed,
        args.phantom,
        dt,
        n_steps,
        device_summary(device),
        args.dtype,
        compile_step,
        args.optimizer,
        args.step_size,
    )

    if not args.no_show or args.save_figure:
        figure = plot_inversion(
            c_true,
            c_est,
            history,
            x_m,
            y_m,
            array,
            mask=mask,
            optimizer=args.optimizer,
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
