"""Circular ring transducer: each element is a transmitter and a receiver."""

import numpy as np


class RingArray:
    """N point elements equally spaced on a circle.

    Elements are nearest-grid-point sources. Receivers use bilinear sampling
    of the pressure field so recorded traces are not locked to pixel centers.
    """

    def __init__(self, n_elements, radius_m, x_m, y_m, center=(0.0, 0.0)):
        if n_elements < 2:
            raise ValueError("Need at least 2 ring elements.")
        self.n_elements = int(n_elements)
        self.radius_m = float(radius_m)
        self.center = (float(center[0]), float(center[1]))
        self.theta = 2.0 * np.pi * np.arange(self.n_elements) / self.n_elements
        self.x = self.center[0] + self.radius_m * np.cos(self.theta)
        self.y = self.center[1] + self.radius_m * np.sin(self.theta)
        self._x_axis = np.asarray(x_m, dtype=float)
        self._y_axis = np.asarray(y_m, dtype=float)
        self.cols, self.rows = self._nearest_indices()
        self._check_inside()
        self._init_bilinear()

    def _nearest_indices(self):
        dx = self._x_axis[1] - self._x_axis[0]
        dy = self._y_axis[1] - self._y_axis[0]
        cols = np.rint((self.x - self._x_axis[0]) / dx).astype(np.intp)
        rows = np.rint((self.y - self._y_axis[0]) / dy).astype(np.intp)
        return cols, rows

    def _check_inside(self):
        ny = self._y_axis.size
        nx = self._x_axis.size
        if np.any(self.cols < 1) or np.any(self.cols > nx - 2):
            raise ValueError("Ring radius extends too close to the x boundaries.")
        if np.any(self.rows < 1) or np.any(self.rows > ny - 2):
            raise ValueError("Ring radius extends too close to the y boundaries.")
        spacing = 2.0 * np.pi * self.radius_m / self.n_elements
        dx = abs(self._x_axis[1] - self._x_axis[0])
        if spacing < dx:
            raise ValueError(
                f"Element spacing {spacing:.4g} m is smaller than dx = {dx:.4g} m; "
                "two elements would share a pixel. Reduce N or refine the grid."
            )

    def transmitter_index(self, tx):
        tx = int(tx)
        if tx < 0 or tx >= self.n_elements:
            raise ValueError(f"Transmitter index {tx} is out of range.")
        return tx

    def inject_rows_cols(self, tx):
        tx = self.transmitter_index(tx)
        return self.rows[tx], self.cols[tx]

    def _init_bilinear(self):
        x_axis = self._x_axis
        y_axis = self._y_axis
        dx = x_axis[1] - x_axis[0]
        dy = y_axis[1] - y_axis[0]
        col = (self.x - x_axis[0]) / dx
        row = (self.y - y_axis[0]) / dy
        self._col0 = np.floor(col).astype(np.intp)
        self._row0 = np.floor(row).astype(np.intp)
        self._col1 = self._col0 + 1
        self._row1 = self._row0 + 1
        wx = col - self._col0
        wy = row - self._row0
        self._w00 = (1.0 - wx) * (1.0 - wy)
        self._w10 = wx * (1.0 - wy)
        self._w01 = (1.0 - wx) * wy
        self._w11 = wx * wy

    def record(self, field):
        """Bilinear sample of field at every element location."""
        return (
            field[self._row0, self._col0] * self._w00
            + field[self._row0, self._col1] * self._w10
            + field[self._row1, self._col0] * self._w01
            + field[self._row1, self._col1] * self._w11
        )

    def record_adjoint(self, residual, out=None):
        """Scatter a per-element residual onto the grid (adjoint of record)."""
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.n_elements,):
            raise ValueError(
                f"Residual shape {residual.shape} does not match "
                f"({self.n_elements},)."
            )
        ny = self._y_axis.size
        nx = self._x_axis.size
        if out is None:
            out = np.zeros((ny, nx), dtype=float)
        else:
            out.fill(0.0)
        np.add.at(out, (self._row0, self._col0), residual * self._w00)
        np.add.at(out, (self._row0, self._col1), residual * self._w10)
        np.add.at(out, (self._row1, self._col0), residual * self._w01)
        np.add.at(out, (self._row1, self._col1), residual * self._w11)
        return out
