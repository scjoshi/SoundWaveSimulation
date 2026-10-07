"""Tests for the measurement helpers in src/phase2_numerics.py (CPU only; the GPU runs are in the doc)."""

import numpy as np
import pytest

from src.phase2_numerics import F0, T_PULSE, WATER, Medium, analytic_trace, chirp, first_arrival, xcorr_shift


def test_chirp_is_windowed_and_band_limited():
    t = np.arange(-5e-6, T_PULSE + 5e-6, 1e-8)
    s = chirp(t)
    assert np.all(s[t < 0] == 0) and np.all(s[t > T_PULSE] == 0)
    assert abs(s[0]) == 0 and np.abs(s).max() <= 1.0
    spec = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(t.size, 1e-8)
    assert 0.8 * F0 < f[spec.argmax()] < 3e5  # energy in the chirp band


def test_refined_medium_is_identical_piecewise():
    c = np.array([[1500.0, 1600.0], [1450.0, 3000.0]])
    m = Medium(c, 1e-3, 0.0)
    r = m.refined(2)
    assert r.c.shape == (4, 4) and r.h == pytest.approx(0.5e-3)
    assert np.all(r.c[:2, :2] == 1500.0) and np.all(r.c[2:, 2:] == 3000.0)
    # fine cell centres sit symmetrically inside the coarse cells
    assert r.x[0] == pytest.approx(m.x[0] - m.h / 4) and r.x[1] == pytest.approx(m.x[0] + m.h / 4)


def test_xcorr_shift_recovers_known_delay():
    t = np.arange(0, 60e-6, 2e-8)
    a = chirp(t - 10.0e-6)
    b = chirp(t - 10.0e-6 - 0.37e-6)  # b is later by 0.37 us
    assert xcorr_shift(b, a, t, (5e-6, 40e-6)) == pytest.approx(0.37e-6, abs=5e-9)


def test_first_arrival_interpolates_between_samples():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 0.0, 1.0, 2.0])
    assert first_arrival(x, t, 0.5) == pytest.approx(1.5)
    assert np.isnan(first_arrival(x, t, 5.0))


def test_analytic_trace_onset_and_linearity():
    r, h, dt = 0.1, 0.5e-3, 5e-8
    t = np.arange(1, 4000) * dt
    u = analytic_trace(r, t, h, dt)
    assert np.abs(u[t < r / WATER - dt]).max() < 1e-9 * np.abs(u).max()  # causal (FFT round-off only)
    onset = t[np.argmax(np.abs(u) > 1e-3 * np.abs(u).max())]
    assert r / WATER <= onset < r / WATER + 1e-6  # the Hann window ramps up gently
    assert np.allclose(analytic_trace(r, t, 2 * h, dt), 4 * u)  # source strength scales with h^2
    far = analytic_trace(2 * r, t, h, dt)
    assert np.abs(far).max() / np.abs(u).max() == pytest.approx(1 / np.sqrt(2), rel=0.1)  # 2D spreading
