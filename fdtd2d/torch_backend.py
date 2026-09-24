"""PyTorch backend for the 2D scalar-wave ring simulation.

The time recurrence is sequential, while each spatial update and receiver
gather is issued as a parallel PyTorch kernel. CUDA decides the block layout
from the selected device, so the implementation automatically uses the GPU's
available streaming multiprocessors without hard-coding a core count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda[:index]/mps and validate availability."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Unknown device {requested!r}; use auto, cpu, cuda, cuda:N, or mps."
        ) from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {index} is unavailable; found {torch.cuda.device_count()}."
            )
        return torch.device("cuda", index)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested, but this PyTorch build cannot access it.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("The FDTD backend supports only CPU, CUDA, and MPS devices.")
    return device


def configure_runtime(
    device: torch.device,
    cpu_threads: int = 0,
    allow_tf32: bool = False,
) -> None:
    """Configure host threading and optional CUDA TensorFloat-32 execution."""
    if cpu_threads < 0:
        raise ValueError("cpu_threads must be zero (automatic) or positive.")
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.benchmark = True


def device_summary(device: torch.device) -> str:
    """Human-readable execution-device and parallel-hardware description."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        memory_gib = props.total_memory / 1024**3
        return (
            f"CUDA {device.index or 0}: {props.name}; "
            f"{props.multi_processor_count} SMs; {memory_gib:.1f} GiB"
        )
    if device.type == "mps":
        return "Apple Metal (MPS); GPU core scheduling managed by PyTorch/Metal"
    return f"CPU; {torch.get_num_threads()} PyTorch threads"


def linear_chirp(
    dt: float,
    duration_s: float,
    f_start_hz: float,
    f_end_hz: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a Hann-windowed linear chirp directly on the target device."""
    if duration_s <= 0.0:
        raise ValueError("Chirp duration must be positive.")
    if f_start_hz <= 0.0 or f_end_hz <= 0.0:
        raise ValueError("Chirp frequencies must be positive.")
    n_pulse = max(int(round(duration_s / dt)), 2)
    t = torch.arange(n_pulse, device=device, dtype=dtype) * dt
    sweep = (f_end_hz - f_start_hz) / float((n_pulse - 1) * dt)
    phase = 2.0 * math.pi * (f_start_hz * t + 0.5 * sweep * t.square())
    window = 0.5 - 0.5 * torch.cos(
        2.0 * math.pi * torch.arange(n_pulse, device=device, dtype=dtype)
        / (n_pulse - 1)
    )
    return window * torch.sin(phase)


class TorchScalarWave2D:
    """Scalar wave model whose fields and coefficients remain on one device."""

    def __init__(self, c, dx: float, dy: float | None = None, *, device, dtype):
        self.device = torch.device(device)
        self.dtype = dtype
        self._c = torch.as_tensor(c, device=self.device, dtype=dtype).contiguous()
        if self._c.ndim != 2:
            raise ValueError("Sound speed c must be a 2D array (ny, nx).")
        if not bool(torch.isfinite(self._c).all()) or not bool((self._c > 0).all()):
            raise ValueError("Sound speed must contain finite positive values.")
        self._dx = float(dx)
        self._dy = float(dx if dy is None else dy)
        if self._dx <= 0.0 or self._dy <= 0.0:
            raise ValueError("Grid spacing must be positive.")
        self._c2 = self._c.square()

        inv_dx2 = 1.0 / self._dx**2
        inv_dy2 = 1.0 / self._dy**2
        self.laplacian_kernel = torch.tensor(
            [
                [0.0, inv_dy2, 0.0],
                [inv_dx2, -2.0 * (inv_dx2 + inv_dy2), inv_dx2],
                [0.0, inv_dy2, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        ).reshape(1, 1, 3, 3)

    @property
    def shape(self):
        return tuple(self._c.shape)

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def c_max(self):
        return float(self._c.max().item())

    @property
    def c(self):
        return self._c

    @property
    def c2(self):
        return self._c2


def _advance_field(
    u_old,
    u,
    c2,
    kernel,
    source_value,
    source_row: int,
    source_col: int,
    add_source: bool,
    k_left,
    k_right,
    k_bottom,
    k_top,
    corner_rows,
    corner_cols,
    corner_rows_in,
    corner_cols_in,
    corner_kx,
    corner_ky,
    dt2: float,
):
    """One vectorized leapfrog update, including source and Mur boundaries."""
    laplacian = functional.conv2d(
        u[None, None], kernel, padding=1
    )[0, 0]
    u_next = 2.0 * u - u_old + dt2 * c2 * laplacian
    if add_source:
        u_next[source_row, source_col] += source_value

    u_next[1:-1, 0] = u[1:-1, 1] + k_left * (
        u_next[1:-1, 1] - u[1:-1, 0]
    )
    u_next[1:-1, -1] = u[1:-1, -2] + k_right * (
        u_next[1:-1, -2] - u[1:-1, -1]
    )
    u_next[0, 1:-1] = u[1, 1:-1] + k_bottom * (
        u_next[1, 1:-1] - u[0, 1:-1]
    )
    u_next[-1, 1:-1] = u[-2, 1:-1] + k_top * (
        u_next[-2, 1:-1] - u[-1, 1:-1]
    )

    corner_u = u[corner_rows, corner_cols]
    from_x = u[corner_rows, corner_cols_in] + corner_kx * (
        u_next[corner_rows, corner_cols_in] - corner_u
    )
    from_y = u[corner_rows_in, corner_cols] + corner_ky * (
        u_next[corner_rows_in, corner_cols] - corner_u
    )
    u_next[corner_rows, corner_cols] = 0.5 * (from_x + from_y)
    return u, u_next


class TorchFDTD2D:
    """Second-order leapfrog solver using parallel PyTorch tensor kernels."""

    def __init__(self, model: TorchScalarWave2D, dt: float, compile_step=False):
        self.model = model
        self.dt = float(dt)
        self.dt2 = self.dt * self.dt
        self.u_old = torch.zeros(model.shape, device=model.device, dtype=model.dtype)
        self.u = torch.zeros_like(self.u_old)
        self.source_row = 0
        self.source_col = 0
        c = model.c
        dt = self.dt
        dx = model.dx
        dy = model.dy
        self.k_left = (c[1:-1, 0] * dt - dx) / (c[1:-1, 0] * dt + dx)
        self.k_right = (c[1:-1, -1] * dt - dx) / (c[1:-1, -1] * dt + dx)
        self.k_bottom = (c[0, 1:-1] * dt - dy) / (c[0, 1:-1] * dt + dy)
        self.k_top = (c[-1, 1:-1] * dt - dy) / (c[-1, 1:-1] * dt + dy)
        self.corner_rows = torch.tensor([0, 0, -1, -1], device=model.device)
        self.corner_cols = torch.tensor([0, -1, 0, -1], device=model.device)
        self.corner_rows_in = torch.tensor([1, 1, -2, -2], device=model.device)
        self.corner_cols_in = torch.tensor([1, -2, 1, -2], device=model.device)
        corner_c = c[self.corner_rows, self.corner_cols]
        self.corner_kx = (corner_c * dt - dx) / (corner_c * dt + dx)
        self.corner_ky = (corner_c * dt - dy) / (corner_c * dt + dy)
        if self.cfl >= 1.0:
            raise ValueError(f"Unstable time step: CFL = {self.cfl:.3f} >= 1.")
        self.compiled = bool(compile_step)
        self._advance = _advance_field
        if self.compiled:
            if not hasattr(torch, "compile"):
                raise RuntimeError("This PyTorch version does not support torch.compile.")
            self._advance = torch.compile(
                _advance_field, mode="reduce-overhead", fullgraph=True
            )

    @property
    def cfl(self):
        return float(
            self.model.c_max
            * self.dt
            * math.sqrt(1.0 / self.model.dx**2 + 1.0 / self.model.dy**2)
        )

    def set_source(self, row: int, col: int):
        self.source_row = int(row)
        self.source_col = int(col)

    def _step(self, value, add_source):
        self.u_old, self.u = self._advance(
            self.u_old,
            self.u,
            self.model.c2,
            self.model.laplacian_kernel,
            value,
            self.source_row,
            self.source_col,
            add_source,
            self.k_left,
            self.k_right,
            self.k_bottom,
            self.k_top,
            self.corner_rows,
            self.corner_cols,
            self.corner_rows_in,
            self.corner_cols_in,
            self.corner_kx,
            self.corner_ky,
            self.dt2,
        )
        return self.u

    def inject_and_step(self, value):
        return self._step(value, True)

    def step(self, zero):
        return self._step(zero, False)


class TorchRingSampler:
    """Bilinear receiver gather with all indices and weights on the device."""

    def __init__(self, ring_array, *, device, dtype):
        self.n_elements = ring_array.n_elements
        index_names = ("_row0", "_row1", "_col0", "_col1")
        for name in index_names:
            value = torch.as_tensor(
                getattr(ring_array, name), device=device, dtype=torch.long
            )
            setattr(self, name, value)
        for name in ("_w00", "_w10", "_w01", "_w11"):
            value = torch.as_tensor(getattr(ring_array, name), device=device, dtype=dtype)
            setattr(self, name, value)

    def record(self, field):
        return (
            field[self._row0, self._col0] * self._w00
            + field[self._row0, self._col1] * self._w10
            + field[self._row1, self._col0] * self._w01
            + field[self._row1, self._col1] * self._w11
        )


def simulate_shot(
    solver: TorchFDTD2D,
    sampler: TorchRingSampler,
    pulse: torch.Tensor,
    source_row: int,
    source_col: int,
    n_steps: int,
    snapshot_stride: int = 0,
):
    """Run one shot while retaining fields and traces on the target device."""
    solver.set_source(source_row, source_col)
    traces = torch.empty(
        (sampler.n_elements, n_steps),
        device=solver.model.device,
        dtype=solver.model.dtype,
    )
    snapshots = []
    zero = torch.zeros((), device=solver.model.device, dtype=solver.model.dtype)
    with torch.inference_mode():
        for step in range(n_steps):
            if step < pulse.numel():
                field = solver.inject_and_step(pulse[step])
            else:
                field = solver.step(zero)
            traces[:, step] = sampler.record(field)
            if snapshot_stride and step % snapshot_stride == 0:
                snapshots.append((field.detach().cpu().numpy().copy(), step))
    return traces, snapshots
