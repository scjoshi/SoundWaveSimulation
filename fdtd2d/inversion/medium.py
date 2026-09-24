"""Slowness-squared parameterization and the interior inversion mask.

The FDTD stepper stores sound speed c. Inversion updates

    m = 1 / c^2

and converts back with c = 1 / sqrt(m) before each modeling call.
Only pixels strictly inside the ring (minus a few-cell margin) are free;
the background outside the mask stays at the known water speed.
"""

import numpy as np


def m_from_c(c):
    c = np.asarray(c, dtype=float)
    return 1.0 / (c * c)


def c_from_m(m):
    m = np.asarray(m, dtype=float)
    return 1.0 / np.sqrt(m)


def clip_m(m, c_min, c_max):
    """Clip slowness-squared so c stays in [c_min, c_max]."""
    m_min = 1.0 / (float(c_max) ** 2)
    m_max = 1.0 / (float(c_min) ** 2)
    return np.clip(m, m_min, m_max)


def interior_mask(x_m, y_m, radius_m, margin_m, center=(0.0, 0.0)):
    """True for pixels with r < radius_m - margin_m."""
    x, y = np.meshgrid(x_m, y_m, indexing="xy")
    dx = x - float(center[0])
    dy = y - float(center[1])
    radius = float(radius_m) - float(margin_m)
    if radius <= 0.0:
        raise ValueError("Interior radius must be positive; reduce the margin.")
    return dx * dx + dy * dy < radius * radius
