"""Sound-speed phantoms for the 2D ring-array FDTD."""

import numpy as np

# Modified Shepp–Logan ellipses (Toft 1996 / MATLAB phantom).
# Columns: additive amplitude, a, b, x0, y0, rotation (deg), in the unit square.
MODIFIED_SHEPP_LOGAN = (
    (1.00, 0.6900, 0.9200, 0.0000, 0.0000, 0.0),
    (-0.80, 0.6624, 0.8740, 0.0000, -0.0184, 0.0),
    (-0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18.0),
    (-0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18.0),
    (0.10, 0.2100, 0.2500, 0.0000, 0.3500, 0.0),
    (0.10, 0.0460, 0.0460, 0.0000, 0.1000, 0.0),
    (0.10, 0.0460, 0.0460, 0.0000, -0.1000, 0.0),
    (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0.0),
    (0.10, 0.0230, 0.0230, 0.0000, -0.6050, 0.0),
    (0.10, 0.0230, 0.0460, 0.0600, -0.6050, 0.0),
)

# Outer ellipse semi-axis in normalized coordinates; used to fit inside the ring.
_SHEPP_OUTER_B = 0.92
_SHEPP_FIT = 0.88
# Sound-speed contrast (m/s) per unit phantom amplitude. Rim ~ c0 + this value.
SHEPP_DC = 400.0


def _ellipse_mask(x, y, x0, y0, a, b, phi_deg):
    phi = np.deg2rad(phi_deg)
    dx = x - x0
    dy = y - y0
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    xr = dx * cos_p + dy * sin_p
    yr = -dx * sin_p + dy * cos_p
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def shepp_logan_mu(x, y, scale_m):
    """Modified Shepp–Logan additive phantom, scaled from the unit square."""
    mu = np.zeros_like(x, dtype=float)
    for amp, a, b, x0, y0, phi in MODIFIED_SHEPP_LOGAN:
        inside = _ellipse_mask(
            x, y, scale_m * x0, scale_m * y0, scale_m * a, scale_m * b, phi,
        )
        mu[inside] += amp
    return np.clip(mu, 0.0, None)


def shepp_logan_speed(x, y, ring_radius_m, c0, dc=SHEPP_DC):
    """Map the Shepp–Logan phantom onto a sound-speed field (m/s).

    The head is scaled to sit inside the ring. Background (mu = 0) is c0;
    the outer rim is about c0 + dc.
    """
    scale_m = _SHEPP_FIT * ring_radius_m / _SHEPP_OUTER_B
    mu = shepp_logan_mu(x, y, scale_m)
    return c0 + dc * mu
