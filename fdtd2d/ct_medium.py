"""Prepare a CT-derived sound-speed map for the ring FDTD grid."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .experiment import axis_centers


@dataclass(frozen=True)
class CTRingGrid:
    """A centered CT medium and ring geometry on a square FDTD grid."""

    c: np.ndarray
    body_mask: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    spacing_m: float
    ring_radius_m: float
    source_shape: tuple[int, int]
    source_spacing_mm: tuple[float, float]


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"speed_m_s", "body_mask", "spacing_mm"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(
                f"{path} is missing {', '.join(missing)}. "
                "Use an .npz produced by ct_to_speed.py."
            )
        speed = np.asarray(archive["speed_m_s"], dtype=float)
        body = np.asarray(archive["body_mask"], dtype=bool)
        spacing = np.asarray(archive["spacing_mm"], dtype=float).ravel()
    if speed.ndim != 2 or body.shape != speed.shape:
        raise ValueError("speed_m_s and body_mask must be matching 2D arrays.")
    if spacing.size != 2 or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_mm must contain positive row and column spacing.")
    if not np.all(np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("speed_m_s must contain finite positive values.")
    if not np.any(body):
        raise ValueError("body_mask contains no foreground pixels.")
    return speed, body, (float(spacing[0]), float(spacing[1]))


def _body_center_and_radius(body, source_spacing_mm):
    rows, cols = np.nonzero(body)
    center_row = 0.5 * (float(rows.min()) + float(rows.max()))
    center_col = 0.5 * (float(cols.min()) + float(cols.max()))
    dy_mm, dx_mm = source_spacing_mm
    radius_mm = float(
        np.max(
            np.hypot(
                (rows - center_row) * dy_mm,
                (cols - center_col) * dx_mm,
            )
        )
    )
    return center_row, center_col, radius_mm


def _sample_centered(
    array,
    output_shape,
    output_spacing_mm,
    source_center,
    source_spacing_mm,
    order,
    cval,
):
    ny, nx = output_shape
    center_row, center_col = source_center
    source_dy_mm, source_dx_mm = source_spacing_mm
    output_rows = (
        center_row
        + (np.arange(ny) - 0.5 * (ny - 1))
        * output_spacing_mm
        / source_dy_mm
    )
    output_cols = (
        center_col
        + (np.arange(nx) - 0.5 * (nx - 1))
        * output_spacing_mm
        / source_dx_mm
    )
    row_grid, col_grid = np.meshgrid(output_rows, output_cols, indexing="ij")
    return ndimage.map_coordinates(
        array,
        (row_grid, col_grid),
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )


def load_ct_ring_grid(
    path,
    grid_shape=(512, 512),
    padding_speed_m_s=1480.0,
    ring_clearance_mm=10.0,
    edge_margin_pixels=8,
):
    """Load, downsample, center, and pad a CT medium for a ring acquisition.

    The output spacing is isotropic and never finer than either source-pixel
    spacing, so this operation only downsamples. The complete transformed body
    fits inside a circular ring with the requested physical clearance. Pixels
    outside the transformed body are coupling water (or another requested
    padding speed), while internal gas/tissue values are retained.
    """
    path = Path(path)
    speed, body, source_spacing_mm = _load_npz(path)
    ny, nx = map(int, grid_shape)
    if ny != nx or ny < 16:
        raise ValueError("The CT FDTD grid must be square and at least 16x16.")
    if not np.isfinite(padding_speed_m_s) or padding_speed_m_s <= 0:
        raise ValueError("padding_speed_m_s must be positive.")
    if ring_clearance_mm <= 0:
        raise ValueError("ring_clearance_mm must be positive.")
    if edge_margin_pixels < 3 or edge_margin_pixels >= nx // 4:
        raise ValueError("edge_margin_pixels must be at least 3 and below N/4.")

    center_row, center_col, source_radius_mm = _body_center_and_radius(
        body, source_spacing_mm
    )
    available_ring_radius_px = 0.5 * (nx - 1) - float(edge_margin_pixels)
    fit_spacing_mm = (
        source_radius_mm + float(ring_clearance_mm)
    ) / available_ring_radius_px
    output_spacing_mm = max(*source_spacing_mm, fit_spacing_mm)

    source_center = (center_row, center_col)
    output_shape = (ny, nx)
    output_body = _sample_centered(
        body.astype(np.uint8),
        output_shape,
        output_spacing_mm,
        source_center,
        source_spacing_mm,
        order=0,
        cval=0,
    ).astype(bool)
    output_speed = _sample_centered(
        speed,
        output_shape,
        output_spacing_mm,
        source_center,
        source_spacing_mm,
        order=0,
        cval=float(padding_speed_m_s),
    )
    output_speed[~output_body] = float(padding_speed_m_s)

    out_rows, out_cols = np.nonzero(output_body)
    output_center = 0.5 * (nx - 1)
    body_radius_px = float(
        np.max(np.hypot(out_rows - output_center, out_cols - output_center))
    )
    ring_radius_mm = body_radius_px * output_spacing_mm + float(ring_clearance_mm)
    max_ring_radius_mm = available_ring_radius_px * output_spacing_mm
    if ring_radius_mm > max_ring_radius_mm:
        # Nearest-neighbor sampling can expand the body edge by a fraction of a
        # pixel. Increase the physical grid spacing just enough and resample.
        output_spacing_mm *= ring_radius_mm / max_ring_radius_mm
        output_body = _sample_centered(
            body.astype(np.uint8), output_shape, output_spacing_mm,
            source_center, source_spacing_mm, order=0, cval=0,
        ).astype(bool)
        output_speed = _sample_centered(
            speed, output_shape, output_spacing_mm, source_center,
            source_spacing_mm, order=0, cval=float(padding_speed_m_s),
        )
        output_speed[~output_body] = float(padding_speed_m_s)
        out_rows, out_cols = np.nonzero(output_body)
        body_radius_px = float(
            np.max(np.hypot(out_rows - output_center, out_cols - output_center))
        )
        ring_radius_mm = body_radius_px * output_spacing_mm + float(
            ring_clearance_mm
        )

    spacing_m = 1.0e-3 * output_spacing_mm
    return CTRingGrid(
        c=output_speed,
        body_mask=output_body,
        x_m=axis_centers(nx, spacing_m),
        y_m=axis_centers(ny, spacing_m),
        spacing_m=spacing_m,
        ring_radius_m=1.0e-3 * ring_radius_mm,
        source_shape=tuple(map(int, speed.shape)),
        source_spacing_mm=source_spacing_mm,
    )
