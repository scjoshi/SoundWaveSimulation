"""Adjoint-state gradient of the ring-array waveform misfit.

Forward leapfrog (existing FDTD2D):

    u^{n+1} = 2 u^n - u^{n-1} + dt^2 c^2 Δ u^n + s^n

with c^2 = 1/m. The discrete adjoint runs the same leapfrog backward with
L*[λ] = Δ(c^2 λ) and injects the bilinear scatter of the residual
r = P u - d (transmitter muted). Mur ABC on the adjoint is the same
first-order operator as the forward run: that is an approximation, but the
ring sits well inside the domain.

The gradient with respect to c^2 is the zero-lag correlation

    g_{c^2} += λ^{n+1} ⊙ (dt^2 Δ u^n)

and the chain rule m = 1/c^2 gives g_m = -g_{c^2} / m^2, then the interior
mask and optional Tikhonov term.
"""

import numpy as np

from ..experiment import simulate_shot
from .medium import c_from_m, clip_m, m_from_c
from .misfit import mute_transmitter, tikhonov, waveform_misfit


def forward_store(solver, array, pulse, tx, n_steps):
    """Forward shot that also keeps u after every time step (float32)."""
    tx = array.transmitter_index(tx)
    rows, cols = array.inject_rows_cols(tx)
    ny, nx = solver.u.shape
    traces = np.zeros((array.n_elements, n_steps), dtype=float)
    wave = np.empty((n_steps, ny, nx), dtype=np.float32)
    n_pulse = pulse.size
    solver.reset()
    for step in range(n_steps):
        value = float(pulse[step]) if step < n_pulse else 0.0
        field = solver.inject_and_step(rows, cols, value)
        traces[:, step] = array.record(field)
        wave[step] = field
    return traces, wave


class LeastSquaresFWI:
    """J(m) and ∇J for one or more ring-array shots."""

    def __init__(
        self,
        solver,
        array,
        pulse,
        n_steps,
        sources,
        observed,
        mask,
        alpha=0.0,
        c_min=1400.0,
        c_max=2000.0,
        c_background=1500.0,
    ):
        self.solver = solver
        self.model = solver.model
        self.array = array
        self.pulse = np.asarray(pulse, dtype=float)
        self.n_steps = int(n_steps)
        self.sources = [array.transmitter_index(tx) for tx in sources]
        self.observed = [np.asarray(d, dtype=float) for d in observed]
        if len(self.observed) != len(self.sources):
            raise ValueError("Need one observed gather per source.")
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.shape != solver.u.shape:
            raise ValueError("Mask shape must match the FDTD grid.")
        self.alpha = float(alpha)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.c_background = float(c_background)
        self._adj_src = np.zeros(solver.u.shape, dtype=float)
        self._lap = np.zeros(solver.u.shape, dtype=float)
        self._u_n = np.zeros(solver.u.shape, dtype=float)
        self._g_c2 = np.zeros(solver.u.shape, dtype=float)

    def project(self, m):
        m = clip_m(np.asarray(m, dtype=float), self.c_min, self.c_max)
        m = np.array(m, dtype=float, copy=True)
        m[~self.mask] = m_from_c(self.c_background)
        return m

    def apply_medium(self, m):
        m = self.project(m)
        self.model.set_c(c_from_m(m))
        return m

    def misfit(self, m):
        """Forward-only J(m), used by the CG line search."""
        m = self.apply_medium(m)
        J = 0.0
        for tx, data in zip(self.sources, self.observed):
            traces, _ = simulate_shot(
                self.solver, self.array, self.pulse, tx, self.n_steps,
            )
            J += waveform_misfit(traces, data, tx)
        Jt, _ = tikhonov(m, self.mask, self.alpha)
        return J + Jt

    def misfit_and_grad(self, m):
        """Forward + adjoint: J(m) and g_m on the interior mask."""
        m = self.apply_medium(m)
        J = 0.0
        g_c2 = self._g_c2
        g_c2.fill(0.0)
        for tx, data in zip(self.sources, self.observed):
            traces, wave = forward_store(
                self.solver, self.array, self.pulse, tx, self.n_steps,
            )
            J += waveform_misfit(traces, data, tx)
            self._adjoint_accumulate(wave, traces, data, tx, g_c2)
        Jt, g_t = tikhonov(m, self.mask, self.alpha)
        g_m = -g_c2 / (m * m)
        g_m += g_t
        g_m[~self.mask] = 0.0
        return J + Jt, g_m

    def _adjoint_accumulate(self, wave, predicted, observed, tx, g_c2):
        residual = mute_transmitter(predicted - observed, tx)
        solver = self.solver
        model = self.model
        dt2 = solver.dt2
        lap = self._lap
        u_n = self._u_n
        src = self._adj_src
        solver.reset()
        n_steps = self.n_steps
        for n in range(n_steps - 1, -1, -1):
            self.array.record_adjoint(residual[:, n], out=src)
            lam = solver.inject_field_and_step(src, adjoint=True)
            if n == 0:
                continue
            np.copyto(u_n, wave[n - 1])
            model.laplacian(u_n, lap)
            g_c2 += lam * (dt2 * lap)
