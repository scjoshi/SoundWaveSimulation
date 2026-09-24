"""Wave-equation models for 2D FDTD.

The time stepper calls WaveModel.acceleration(u), which is the right-hand
side of

    d^2 u / dt^2 = L[u]

The default model is the scalar wave equation with spatially varying speed:

    L[u] = c(x, y)^2 (u_xx + u_yy)

Replace ScalarWave2D with another WaveModel subclass to add density,
attenuation, elasticity, or other physics without changing the FDTD loop.
"""

from abc import ABC, abstractmethod

import numpy as np


class WaveModel(ABC):
    """Spatial operator L such that u_tt = L[u]."""

    @property
    @abstractmethod
    def shape(self):
        """(ny, nx) of the field."""

    @property
    @abstractmethod
    def dx(self):
        """Grid spacing in x (m)."""

    @property
    @abstractmethod
    def dy(self):
        """Grid spacing in y (m)."""

    @property
    @abstractmethod
    def c_max(self):
        """Maximum sound speed (m/s), used for the CFL check."""

    @property
    @abstractmethod
    def c(self):
        """Sound-speed map on the grid (m/s). Used by first-order Mur ABC."""

    @abstractmethod
    def acceleration(self, u, out):
        """Write L[u] into out. Boundary nodes may be left as 0."""

    def acceleration_adjoint(self, u, out):
        """Write L*[u] into out. Defaults to L for self-adjoint models."""
        return self.acceleration(u, out)


class ScalarWave2D(WaveModel):
    """Second-order scalar wave equation with variable sound speed.

    Density is taken as constant, so interfaces reflect from c jumps only.
    A later VariableDensityWave2D can use the conservative form
    L[u] = rho c^2 div((1/rho) grad u) without touching the stepper.
    """

    def __init__(self, c, dx, dy=None):
        self._c = np.asarray(c, dtype=float)
        if self._c.ndim != 2:
            raise ValueError("Sound speed c must be a 2D array (ny, nx).")
        self._dx = float(dx)
        self._dy = float(dx if dy is None else dy)
        self._c2 = self._c * self._c
        self._inv_dx2 = 1.0 / self._dx**2
        self._inv_dy2 = 1.0 / self._dy**2
        self._work = np.zeros_like(self._c)

    @property
    def shape(self):
        return self._c.shape

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def c_max(self):
        return float(self._c.max())

    @property
    def c(self):
        return self._c

    def set_c(self, c):
        """Replace the sound-speed map and rebuild c^2. Shape must match."""
        c = np.asarray(c, dtype=float)
        if c.shape != self._c.shape:
            raise ValueError(
                f"Sound speed shape {c.shape} does not match {self._c.shape}."
            )
        self._c = np.array(c, dtype=float, copy=True)
        self._c2 = self._c * self._c
        if self._work.shape != self._c.shape:
            self._work = np.zeros_like(self._c)

    def laplacian(self, u, out):
        """Second-order centered Laplacian, interior only."""
        out.fill(0.0)
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        out[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) * inv_dx2
            + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * inv_dy2
        )
        return out

    def acceleration(self, u, out):
        """Second-order centered Laplacian scaled by c^2, interior only."""
        self.laplacian(u, out)
        out *= self._c2
        return out

    def acceleration_adjoint(self, u, out):
        """Adjoint of L[u] = c^2 Δu, which is Δ(c^2 u), interior only."""
        np.multiply(self._c2, u, out=self._work)
        return self.laplacian(self._work, out)
