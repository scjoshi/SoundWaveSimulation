"""Neural-network parameterization of sound speed for FWI.

Instead of one unknown per pixel, sound speed is a coordinate network

    c(x) = c_min + (c_max - c_min) * sigmoid(a_theta(x / L) + b0)

evaluated inside the interior mask, with the known background outside. The
unknowns are the network weights theta. The sigmoid keeps c strictly inside
[c_min, c_max] without the gradient-killing clip of pixel FWI, and b0 is chosen
so that a zero network output reproduces the background speed exactly.

The wave physics is unchanged. Any FWI problem exposing the adjoint gradient
``misfit_and_grad(m) -> (J, dJ/dm)`` on the grid (for example
invert_ring.TorchLeastSquaresFWI) is reused as is, and the chain rule

    dJ/dtheta = (dm/dtheta)^T dJ/dm

is applied with a single backward pass through the network only. Nothing is
backpropagated through the FDTD time loop.

Coordinates are normalized by a fixed physical length L, not by the grid, so
one network can be evaluated on any grid (multiscale continuation).
"""

import math

import numpy as np
import torch
from torch import nn


class SoundSpeedNet(nn.Module):
    """Fourier-feature MLP mapping normalized (x, y) to one scalar per point.

    Random Fourier features gamma(v) = [sin(2 pi B v), cos(2 pi B v)] with
    B ~ N(0, fourier_scale^2) set the finest resolvable detail: a larger
    ``fourier_scale`` allows sharper models (and more noise). B is fixed, not
    trained. The output layer starts at zero so the initial model is exactly
    the background.

    ``activation`` defaults to SiLU: it is smooth, so the loss is
    differentiable in theta (needed for finite-difference checks and helpful
    for L-BFGS). ReLU is available but has kinks that make J only piecewise
    smooth in theta.
    """

    ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(self, n_fourier=128, fourier_scale=4.0, width=256, depth=4, seed=0,
                 activation="silu"):
        super().__init__()
        if n_fourier < 1 or width < 1 or depth < 1:
            raise ValueError("n_fourier, width, and depth must be positive.")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(self.ACTIVATIONS)}.")
        if fourier_scale <= 0:
            raise ValueError("fourier_scale must be positive.")
        generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer(
            "frequencies", fourier_scale * torch.randn(2, n_fourier, generator=generator)
        )
        layers = []
        features = 2 * n_fourier
        for _ in range(depth):
            layers += [nn.Linear(features, width), self.ACTIVATIONS[activation]()]
            features = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(features, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, coords):
        """coords: (N, 2) normalized positions. Returns (N,) raw outputs."""
        projected = 2.0 * math.pi * coords @ self.frequencies
        features = torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)
        return self.output(self.hidden(features))[:, 0]


class NeuralSpeedModel(nn.Module):
    """Bounded, masked sound-speed map produced by a coordinate network."""

    def __init__(self, net, c_min, c_max, c_background, length_scale_m):
        super().__init__()
        if not 0.0 < c_min < c_background < c_max:
            raise ValueError("Need 0 < c_min < c_background < c_max.")
        if length_scale_m <= 0:
            raise ValueError("length_scale_m must be positive.")
        self.net = net
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.c_background = float(c_background)
        self.length_scale_m = float(length_scale_m)
        fraction = (self.c_background - self.c_min) / (self.c_max - self.c_min)
        self.output_offset = math.log(fraction / (1.0 - fraction))
        self._grid = None

    def set_grid(self, x_m, y_m, mask):
        """Bind the grid the model is evaluated on; call again per multiscale stage."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(y_m), len(x_m)):
            raise ValueError("mask shape must be (len(y_m), len(x_m)).")
        x, y = np.meshgrid(np.asarray(x_m, float), np.asarray(y_m, float), indexing="xy")
        parameter = next(self.net.parameters())
        coords = np.stack((x[mask], y[mask]), axis=-1) / self.length_scale_m
        self._grid = {
            "shape": mask.shape,
            "mask": torch.as_tensor(mask, device=parameter.device),
            "coords": torch.as_tensor(coords, device=parameter.device, dtype=parameter.dtype),
        }
        return self

    def _require_grid(self):
        if self._grid is None:
            raise RuntimeError("Call set_grid(x_m, y_m, mask) before evaluating the model.")
        return self._grid

    def speed(self):
        """Sound speed (m/s) on the bound grid as a differentiable (ny, nx) tensor."""
        grid = self._require_grid()
        raw = self.net(grid["coords"])
        inside = self.c_min + (self.c_max - self.c_min) * torch.sigmoid(raw + self.output_offset)
        c = torch.full(grid["shape"], self.c_background, device=raw.device, dtype=raw.dtype)
        # index_put keeps the scatter differentiable with respect to ``inside``.
        return c.index_put((grid["mask"],), inside)

    def slowness(self):
        """Slowness squared m = 1 / c^2, the FWI unknown on the grid."""
        return self.speed().square().reciprocal()

    def fit_to(self, target_speed, iterations=500, learning_rate=1e-3, verbose=False):
        """Pretrain the network to reproduce a given speed map (no wave solves).

        Useful for a starting model or to transfer an earlier result. Returns
        the final RMS speed error (m/s) inside the mask.
        """
        grid = self._require_grid()
        parameter = next(self.net.parameters())
        target = torch.as_tensor(target_speed, device=parameter.device, dtype=parameter.dtype)
        target = target.clamp(self.c_min + 1e-3, self.c_max - 1e-3)[grid["mask"]]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        for iteration in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            error = self.speed()[grid["mask"]] - target
            loss = error.square().mean()
            loss.backward()
            optimizer.step()
            if verbose and (iteration % 100 == 0 or iteration == iterations - 1):
                print(f"  fit {iteration:4d}  RMS {float(loss.sqrt()):.3f} m/s")
        with torch.no_grad():
            return float((self.speed()[grid["mask"]] - target).square().mean().sqrt())


class NeuralFWIObjective:
    """Chain-rule bridge between a grid FWI problem and a NeuralSpeedModel.

    ``problem`` needs ``misfit_and_grad(m)`` returning (J, dJ/dm) as NumPy
    arrays and ``misfit(m)`` returning J; m is slowness squared on the grid.
    With ``normalize`` the loss is J / J(theta_0), so network optimizers see
    O(1) values regardless of the data amplitude.
    """

    def __init__(self, problem, model, normalize=True):
        self.problem = problem
        self.model = model
        self.normalize = bool(normalize)
        self.reference = None

    def _scale(self, value):
        if not self.normalize:
            return 1.0
        if self.reference is None:
            self.reference = max(float(value), 1e-300)
        return 1.0 / self.reference

    def slowness_numpy(self):
        with torch.no_grad():
            return self.model.slowness().detach().cpu().numpy().astype(float)

    def loss_and_backward(self):
        """Evaluate the (scaled) loss and accumulate dLoss/dtheta in ``.grad``.

        Returns (scaled loss, raw J). One forward and one adjoint wave solve
        per shot, plus one network forward/backward pass.
        """
        m = self.model.slowness()
        value, gradient_m = self.problem.misfit_and_grad(
            m.detach().cpu().numpy().astype(float)
        )
        scale = self._scale(value)
        m.backward(gradient=torch.as_tensor(scale * gradient_m, device=m.device, dtype=m.dtype))
        return scale * float(value), float(value)

    def closure(self, optimizer):
        """Closure for torch optimizers: ``optimizer.step(objective.closure(optimizer))``."""
        def evaluate():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self.loss_and_backward()
            return loss
        return evaluate

    def loss(self):
        """Scaled loss only (no gradient), using the problem's cheaper misfit."""
        value = self.problem.misfit(self.slowness_numpy())
        return self._scale(value) * float(value)


def _parameters_vector(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def _assign_vector(parameters, vector):
    offset = 0
    with torch.no_grad():
        for p in parameters:
            count = p.numel()
            p.copy_(vector[offset:offset + count].view_as(p))
            offset += count


def theta_gradient_check(objective, epsilons=(1e-2, 1e-3, 1e-4), seed=0, verbose=True):
    """Directional finite-difference test of dLoss/dtheta.

    Compares grad . v with (L(theta + eps v) - L(theta - eps v)) / (2 eps)
    for a random unit direction v over all network parameters. The problem
    must be deterministic (e.g. every shot active) for the test to be exact.
    Returns a list of (eps, finite difference, relative error).
    """
    parameters = [p for p in objective.model.net.parameters() if p.requires_grad]
    for p in parameters:
        p.grad = None
    loss, value = objective.loss_and_backward()
    gradient = torch.cat([p.grad.reshape(-1) for p in parameters])
    theta = _parameters_vector(parameters)
    generator = torch.Generator().manual_seed(int(seed))
    direction = torch.randn(theta.numel(), generator=generator, dtype=theta.dtype)
    direction = (direction / direction.norm()).to(theta.device)
    directional = float(gradient @ direction)
    results = []
    if verbose:
        print("Neural-parameterization gradient check (theta space)")
        print(f"  parameters = {theta.numel()}")
        print(f"  J = {value:.6e};  scaled loss = {loss:.6e}")
        print(f"  grad . v = {directional:.6e}")
        print(f"  {'eps':>10}  {'FD':>14}  {'rel. error':>12}")
    try:
        for epsilon in epsilons:
            _assign_vector(parameters, theta + epsilon * direction)
            plus = objective.loss()
            _assign_vector(parameters, theta - epsilon * direction)
            minus = objective.loss()
            finite_difference = (plus - minus) / (2.0 * epsilon)
            error = abs(finite_difference - directional) / max(abs(directional), 1e-300)
            results.append((epsilon, finite_difference, error))
            if verbose:
                print(f"  {epsilon:10.1e}  {finite_difference:14.6e}  {error:12.4e}")
    finally:
        _assign_vector(parameters, theta)
    return results


# --------------------------------------------------------------------------
# Command-line integration shared by invert_ring.py and invert_arc.py
# --------------------------------------------------------------------------

NEURAL_OPTIMIZERS = ("adam", "gradient-descent")


def add_neural_arguments(parser):
    """Add ``--model`` and the network options to an FWI argument parser."""
    parser.add_argument(
        "--model", choices=("pixel", "neural"), default="pixel",
        help="Unknowns: one slowness per pixel, or the weights of a coordinate "
             "network c(x) (default: pixel)",
    )
    group = parser.add_argument_group("neural model (--model neural)")
    group.add_argument("--nn-width", type=int, default=256, metavar="N",
                       help="Hidden units per layer (default: 256)")
    group.add_argument("--nn-depth", type=int, default=4, metavar="N",
                       help="Hidden layers (default: 4)")
    group.add_argument("--nn-features", type=int, default=128, metavar="N",
                       help="Random Fourier features (default: 128)")
    group.add_argument("--fourier-scale", type=float, default=2.0, metavar="SIGMA",
                       help="Fourier-feature frequency scale in cycles per length "
                            "scale; larger allows sharper models (default: 2)")
    group.add_argument("--nn-activation", choices=sorted(SoundSpeedNet.ACTIVATIONS),
                       default="silu", help="Hidden activation (default: silu)")
    group.add_argument("--nn-lr", type=float, default=3e-3, metavar="ALPHA",
                       help="Learning rate on network weights (default: 3e-3)")
    group.add_argument("--nn-seed", type=int, default=0, metavar="SEED",
                       help="Seed for Fourier features and weights (default: 0)")
    return parser


def validate_neural_arguments(args, c_background):
    """Raise SystemExit for inconsistent neural-model options."""
    if args.model != "neural":
        return
    if args.optimizer not in NEURAL_OPTIMIZERS:
        raise SystemExit(
            f"--model neural supports --optimizer {' or '.join(NEURAL_OPTIMIZERS)}."
        )
    if min(args.nn_width, args.nn_depth, args.nn_features) < 1:
        raise SystemExit("--nn-width, --nn-depth, and --nn-features must be positive.")
    if args.fourier_scale <= 0 or args.nn_lr <= 0:
        raise SystemExit("--fourier-scale and --nn-lr must be positive.")
    if not args.c_min < c_background < args.c_max:
        raise SystemExit(
            f"--model neural needs --c-min < {c_background:g} m/s (background) < --c-max."
        )


def neural_model_from_args(args, c_background, length_scale_m, device, dtype):
    """Build the network and bounded speed model described by ``args``."""
    net = SoundSpeedNet(
        n_fourier=args.nn_features, fourier_scale=args.fourier_scale,
        width=args.nn_width, depth=args.nn_depth, seed=args.nn_seed,
        activation=args.nn_activation,
    ).to(device=device, dtype=dtype)
    return NeuralSpeedModel(net, args.c_min, args.c_max, c_background, length_scale_m)


def describe_neural_model(model):
    net = model.net
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    layers = [m for m in net.hidden if isinstance(m, nn.Linear)]
    return (f"Fourier-feature MLP, {net.frequencies.shape[1]} features, "
            f"{len(layers)} x {layers[0].out_features}, {count} weights")


def optimize_neural(objective, optimizer_name, max_iter, learning_rate,
                    on_evaluate=None, verbose=True):
    """Optimize network weights; returns (m on the grid, history).

    ``history`` matches the pixel optimizers in invert_ring.py: ``misfit``
    holds the raw J (not the normalized loss) for the initial model and after
    every update, and ``grad_norm`` holds |dLoss/dtheta|. ``on_evaluate(m, J)``
    is called after each evaluation (e.g. the live IterationVisualizer).
    """
    parameters = [p for p in objective.model.net.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        label = "Adam"
    elif optimizer_name == "gradient-descent":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        label = "GD"
    else:
        raise ValueError(f"Unsupported neural optimizer {optimizer_name!r}.")
    history = {"misfit": [], "step_size": [], "grad_norm": []}
    for iteration in range(int(max_iter) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, value = objective.loss_and_backward()
        grad_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)))
        history["misfit"].append(value)
        history["step_size"].append(0.0 if iteration == 0 else float(learning_rate))
        history["grad_norm"].append(grad_norm)
        if on_evaluate is not None:
            on_evaluate(objective.slowness_numpy(), value)
        if verbose:
            print(f"  NN {label} iter {iteration:3d}  J = {value:.6e}  "
                  f"J/J0 = {loss:.4e}  |g_theta| = {grad_norm:.4e}")
        if iteration == max_iter:
            break
        optimizer.step()
    return objective.slowness_numpy(), history
