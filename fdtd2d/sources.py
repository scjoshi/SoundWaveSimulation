"""Band-limited source waveforms."""

import numpy as np


def hann_window(n):
    if n <= 1:
        return np.ones(max(n, 0), dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def linear_chirp(dt, duration_s, f_start_hz, f_end_hz):
    """Hann-windowed linear FM pulse, starting at t = 0.

    s(t) = w(t) sin(2 pi (f0 t + 1/2 K t^2)),  K = (f1 - f0) / T
    """
    if duration_s <= 0:
        raise ValueError("Chirp duration must be positive.")
    if f_start_hz <= 0 or f_end_hz <= 0:
        raise ValueError("Chirp frequencies must be positive.")
    n_pulse = max(int(round(duration_s / dt)), 2)
    t = np.arange(n_pulse) * dt
    sweep = (f_end_hz - f_start_hz) / t[-1]
    phase = 2.0 * np.pi * (f_start_hz * t + 0.5 * sweep * t**2)
    pulse = hann_window(n_pulse) * np.sin(phase)
    return pulse
