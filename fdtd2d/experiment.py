"""One-transmitter ring-array experiment and shared grid helpers."""

import numpy as np


def axis_centers(n, spacing):
    """Cell-center coordinates for n cells of the given spacing, origin at 0."""
    return (np.arange(n) - 0.5 * (n - 1)) * spacing


def stable_dt(c_max, dx, dy, cfl=0.45):
    """Largest dt that satisfies the 2D second-order CFL bound at c_max."""
    return cfl / (c_max * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))


def n_steps_for_crossing(radius_m, c_min, dt, extras=1.6):
    """Time to cross the ring diameter, plus margin for the chirp tail."""
    return int(np.ceil(extras * (2.0 * radius_m / c_min) / dt))


def simulate_shot(solver, array, pulse, tx, n_steps, snapshot_stride=0):
    """Fire one element and record every element, including the transmitter.

    Returns
    -------
    traces : (n_elements, n_steps)
        Pressure time series at each ring element.
    snapshots : list of (field, step) or empty
    """
    tx = array.transmitter_index(tx)
    rows, cols = array.inject_rows_cols(tx)
    traces = np.zeros((array.n_elements, n_steps), dtype=float)
    snapshots = []
    n_pulse = pulse.size
    solver.reset()

    for step in range(n_steps):
        value = float(pulse[step]) if step < n_pulse else 0.0
        field = solver.inject_and_step(rows, cols, value)
        traces[:, step] = array.record(field)
        if snapshot_stride and step % snapshot_stride == 0:
            snapshots.append((field.copy(), step))

    return traces, snapshots
