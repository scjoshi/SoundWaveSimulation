"""Waveform least-squares misfit, Tikhonov smoothness and total variation on m."""

import numpy as np


def mute_transmitter(traces, tx):
    """Copy traces with the transmitter channel zeroed."""
    muted = np.array(traces, dtype=float, copy=True)
    muted[tx] = 0.0
    return muted


def waveform_misfit(predicted, observed, tx):
    """1/2 ||P u - d||^2 with the transmitter row omitted."""
    residual = mute_transmitter(predicted - observed, tx)
    return 0.5 * float(np.sum(residual * residual))


def tikhonov(m, mask, alpha):
    """Discrete 1/2 α ||∇m||^2 on interior-mask edges, and its gradient.

    Forward differences count only when both pixels lie in the mask, so the
    known exterior does not leak a jump penalty into the ring interior.
    """
    g = np.zeros_like(m, dtype=float)
    if alpha <= 0.0:
        return 0.0, g

    both_x = mask[:, 1:] & mask[:, :-1]
    dxm = m[:, 1:] - m[:, :-1]
    val_x = 0.5 * alpha * float(np.sum(dxm[both_x] ** 2))
    contrib_x = np.where(both_x, alpha * dxm, 0.0)
    g[:, 1:] += contrib_x
    g[:, :-1] -= contrib_x

    both_y = mask[1:, :] & mask[:-1, :]
    dym = m[1:, :] - m[:-1, :]
    val_y = 0.5 * alpha * float(np.sum(dym[both_y] ** 2))
    contrib_y = np.where(both_y, alpha * dym, 0.0)
    g[1:, :] += contrib_y
    g[:-1, :] -= contrib_y

    g[~mask] = 0.0
    return val_x + val_y, g


def total_variation(m, mask, alpha, eps):
    """Smoothed isotropic α Σ (sqrt(|∇m|² + ε²) − ε) on the interior mask, and its gradient.

    Forward differences count only when both pixels lie in the mask, as in
    ``tikhonov``. ``eps`` (slowness-squared units) rounds the corner of |∇m| so
    the gradient stays finite where the model is flat.
    """
    g = np.zeros_like(m, dtype=float)
    if alpha <= 0.0:
        return 0.0, g

    dx = np.zeros_like(m, dtype=float)
    dy = np.zeros_like(m, dtype=float)
    dx[:, :-1] = np.where(mask[:, 1:] & mask[:, :-1], m[:, 1:] - m[:, :-1], 0.0)
    dy[:-1, :] = np.where(mask[1:, :] & mask[:-1, :], m[1:, :] - m[:-1, :], 0.0)
    norm = np.sqrt(dx * dx + dy * dy + eps * eps)
    value = alpha * float(np.sum(norm - eps))

    gx = alpha * dx / norm
    gy = alpha * dy / norm
    g -= gx + gy
    g[:, 1:] += gx[:, :-1]
    g[1:, :] += gy[:-1, :]
    g[~mask] = 0.0
    return value, g
