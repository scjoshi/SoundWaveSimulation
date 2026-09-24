"""2D FDTD time stepper with first-order Mur absorbing boundaries.

The stepper is independent of the interior physics. It advances

    u_next = 2 u - u_old + dt^2 L[u]

where L comes from a WaveModel, then applies Mur ABC on the domain edges.

Stability (2D, second-order Laplacian) requires

    c_max * dt * sqrt(1/dx^2 + 1/dy^2) < 1
"""

import numpy as np


class FDTD2D:
    """Leapfrog FDTD on a rectangular grid."""

    def __init__(self, model, dt):
        self.model = model
        self.dt = float(dt)
        self.dt2 = self.dt * self.dt
        ny, nx = model.shape
        self.u_old = np.zeros((ny, nx), dtype=float)
        self.u = np.zeros((ny, nx), dtype=float)
        self.u_next = np.zeros((ny, nx), dtype=float)
        cfl = self.cfl
        if cfl >= 1.0:
            raise ValueError(
                f"Unstable time step: CFL = {cfl:.3f} >= 1. "
                "Reduce dt or increase dx."
            )

    @property
    def cfl(self):
        return float(
            self.model.c_max * self.dt * np.sqrt(
                1.0 / self.model.dx**2 + 1.0 / self.model.dy**2
            )
        )

    def reset(self):
        self.u_old.fill(0.0)
        self.u.fill(0.0)
        self.u_next.fill(0.0)

    def add_source(self, rows, cols, value):
        """Additive (soft) source injected into the upcoming field."""
        self.u_next[rows, cols] += value

    def _interior_update(self, adjoint=False):
        if adjoint:
            self.model.acceleration_adjoint(self.u, out=self.u_next)
        else:
            self.model.acceleration(self.u, out=self.u_next)
        self.u_next *= self.dt2
        self.u_next += 2.0 * self.u - self.u_old

    def _commit(self):
        self._apply_mur()
        self.u_old, self.u, self.u_next = self.u, self.u_next, self.u_old
        return self.u

    def step(self):
        """Advance one time step. Source terms must be added after this call
        returns if they should affect the *next* update; prefer inject_and_step.
        """
        self._interior_update()
        return self._commit()

    def inject_and_step(self, rows, cols, value):
        """Interior update, add a soft source, apply ABC, then rotate buffers."""
        self._interior_update()
        if value != 0.0:
            self.u_next[rows, cols] += value
        return self._commit()

    def inject_field_and_step(self, source_field, adjoint=False):
        """Interior update, add a dense source field, apply ABC, rotate.

        Used by the adjoint solver to inject the bilinear scatter of the
        receiver residual. If adjoint is True, L* is used in place of L.
        """
        self._interior_update(adjoint=adjoint)
        if source_field is not None:
            self.u_next += source_field
        return self._commit()

    def _apply_mur(self):
        """First-order Mur ABC on the four edges; corners are averaged 1D Murs."""
        c = self.model.c
        dt = self.dt
        dx = self.model.dx
        dy = self.model.dy
        u = self.u
        un = self.u_next

        k_left = (c[1:-1, 0] * dt - dx) / (c[1:-1, 0] * dt + dx)
        un[1:-1, 0] = u[1:-1, 1] + k_left * (un[1:-1, 1] - u[1:-1, 0])

        k_right = (c[1:-1, -1] * dt - dx) / (c[1:-1, -1] * dt + dx)
        un[1:-1, -1] = u[1:-1, -2] + k_right * (un[1:-1, -2] - u[1:-1, -1])

        k_bottom = (c[0, 1:-1] * dt - dy) / (c[0, 1:-1] * dt + dy)
        un[0, 1:-1] = u[1, 1:-1] + k_bottom * (un[1, 1:-1] - u[0, 1:-1])

        k_top = (c[-1, 1:-1] * dt - dy) / (c[-1, 1:-1] * dt + dy)
        un[-1, 1:-1] = u[-2, 1:-1] + k_top * (un[-2, 1:-1] - u[-1, 1:-1])

        self._mur_corner(0, 0, 1, 1, dt, dx, dy)
        self._mur_corner(0, -1, 1, -2, dt, dx, dy)
        self._mur_corner(-1, 0, -2, 1, dt, dx, dy)
        self._mur_corner(-1, -1, -2, -2, dt, dx, dy)

    def _mur_corner(self, row, col, row_in, col_in, dt, dx, dy):
        c = self.model.c[row, col]
        kx = (c * dt - dx) / (c * dt + dx)
        ky = (c * dt - dy) / (c * dt + dy)
        u = self.u
        un = self.u_next
        from_x = u[row, col_in] + kx * (un[row, col_in] - u[row, col])
        from_y = u[row_in, col] + ky * (un[row_in, col] - u[row, col])
        un[row, col] = 0.5 * (from_x + from_y)
