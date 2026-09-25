"""Visualization for a 2D ring-array FDTD shot."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def set_window_title(fig, title):
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(title)


def _extent_mm(x_m, y_m):
    dx = x_m[1] - x_m[0]
    dy = y_m[1] - y_m[0]
    return [
        1e3 * (x_m[0] - 0.5 * dx),
        1e3 * (x_m[-1] + 0.5 * dx),
        1e3 * (y_m[0] - 0.5 * dy),
        1e3 * (y_m[-1] + 0.5 * dy),
    ]


def _overlay_geometry(ax, array, tx, x_m=None, y_m=None, c_map=None):
    ax.plot(1e3 * array.x, 1e3 * array.y, "k.", ms=3, label="Elements")
    if tx is not None:
        ax.plot(
            1e3 * array.x[tx], 1e3 * array.y[tx], "o", color="tab:red", ms=8,
            label="Transmitter",
        )
    ax.add_patch(Circle(
        (1e3 * array.center[0], 1e3 * array.center[1]), 1e3 * array.radius_m,
        fill=False, linestyle="--", color="0.4", lw=0.8,
    ))
    if c_map is None or x_m is None or y_m is None:
        return
    if float(c_map.max() - c_map.min()) <= 0:
        return
    levels = np.unique(np.round(c_map))
    interior = levels[levels > levels.min() + 0.5]
    if interior.size:
        ax.contour(
            1e3 * x_m, 1e3 * y_m, c_map, levels=interior,
            colors="tab:green", linewidths=1.0,
        )
        ax.plot([], [], color="tab:green", lw=1.6, label="Phantom")


def plot_dashboard(field, traces, time_s, array, pulse, dt, tx, x_m, y_m,
                   c_map=None):
    extent = _extent_mm(x_m, y_m)
    time_us = 1e6 * time_s
    pulse_us = 1e6 * np.arange(pulse.size) * dt
    vabs = max(float(np.max(np.abs(field))), 1e-12)
    gather = traces.copy()
    gather[tx] = np.nan
    gabs = max(float(np.nanmax(np.abs(gather))), 1e-12)
    opposite = (tx + array.n_elements // 2) % array.n_elements
    neighbor = (tx + 1) % array.n_elements

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    set_window_title(fig, "Ring FDTD traces")
    fig.suptitle(
        f"2D ring-array FDTD  (TX element {tx} of {array.n_elements})",
        fontsize=14,
    )

    ax = axes[0, 0]
    image = ax.imshow(
        field, origin="lower", extent=extent, cmap="seismic",
        vmin=-vabs, vmax=vabs, aspect="equal",
    )
    _overlay_geometry(ax, array, tx, x_m=x_m, y_m=y_m, c_map=c_map)
    ax.set(title="Pressure field (final time)", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="pressure")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    image = ax.imshow(
        gather, origin="lower", aspect="auto", cmap="seismic",
        vmin=-gabs, vmax=gabs,
        extent=[time_us[0], time_us[-1], -0.5, array.n_elements - 0.5],
    )
    ax.axhline(tx, color="0.3", ls=":", lw=0.8)
    ax.set(
        title="Received traces (TX row omitted)",
        xlabel="Time (µs)",
        ylabel="Element index",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="pressure")

    ax = axes[1, 0]
    ax.plot(pulse_us, pulse, color="tab:red", lw=1.5)
    ax.set(title="Transmit chirp", xlabel="Time (µs)", ylabel="Amplitude")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time_us, traces[neighbor], color="#1f77b4", lw=1.2,
            label=f"Neighbor (el. {neighbor})")
    ax.plot(time_us, traces[opposite], color="tab:orange", lw=1.2,
            label=f"Opposite (el. {opposite})")
    ax.set(title="Selected received waveforms", xlabel="Time (µs)",
           ylabel="Pressure")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def animate_field(snapshots, array, tx, x_m, y_m, dt, interval=40, c_map=None):
    """Wavefield animation in its own window."""
    extent = _extent_mm(x_m, y_m)
    peak = max(max(float(np.max(np.abs(field))) for field, _ in snapshots), 1e-12)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    set_window_title(fig, "Ring FDTD acquisition")
    image = ax.imshow(
        snapshots[0][0], origin="lower", extent=extent, cmap="seismic",
        vmin=-peak, vmax=peak, aspect="equal",
    )
    _overlay_geometry(ax, array, tx, x_m=x_m, y_m=y_m, c_map=c_map)
    ax.set(xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="pressure")

    def update(frame):
        field, step = snapshots[frame]
        image.set_data(field)
        ax.set_title(f"Pressure  t = {1e6 * step * dt:.1f} µs")
        return image,

    update(0)
    fig.tight_layout()
    anim = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=interval,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    fig._animation = anim
    return fig


def plot_inversion(
    c_true,
    c_est,
    history,
    x_m,
    y_m,
    array,
    mask=None,
    optimizer="cg",
):
    """True / reconstructed / difference sound speed and misfit history."""
    optimizer_labels = {
        "cg": ("adjoint Polak–Ribière CG", "CG iteration"),
        "gradient-descent": (
            "adjoint fixed-step gradient descent",
            "Gradient-descent iteration",
        ),
    }
    method_title, iteration_label = optimizer_labels.get(
        optimizer,
        (str(optimizer).replace("-", " "), "Optimization iteration"),
    )
    extent = _extent_mm(x_m, y_m)
    vmin = float(min(np.min(c_true), np.min(c_est)))
    vmax = float(max(np.max(c_true), np.max(c_est)))
    diff = c_est - c_true
    if mask is not None:
        shown = np.array(diff, copy=True)
        shown[~mask] = 0.0
        diff = shown
    vabs = max(float(np.max(np.abs(diff))), 1e-12)
    misfit = np.asarray(history["misfit"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    set_window_title(fig, "Ring FWI")
    fig.suptitle(
        f"Nonlinear FWI  (slowness-squared, {method_title})", fontsize=14
    )

    panels = (
        (axes[0, 0], c_true, "True $c$ (m/s)", "viridis", vmin, vmax),
        (axes[0, 1], c_est, "Reconstructed $c$ (m/s)", "viridis", vmin, vmax),
        (axes[1, 0], diff, "Difference (est. $-$ true)", "seismic", -vabs, vabs),
    )
    for ax, data, title, cmap, lo, hi in panels:
        image = ax.imshow(
            data, origin="lower", extent=extent, cmap=cmap,
            vmin=lo, vmax=hi, aspect="equal",
        )
        _overlay_geometry(ax, array, tx=None, x_m=x_m, y_m=y_m)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    iters = np.arange(misfit.size)
    ax.plot(iters, misfit, "o-", color="tab:blue", lw=1.5)
    ax.set(title="Waveform misfit", xlabel=iteration_label, ylabel="$J(m)$")
    if misfit.size and np.all(misfit > 0.0):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
