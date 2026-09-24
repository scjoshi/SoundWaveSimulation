"""Convert one abdominal CT slice into tissue labels and sound speed.

This is a simulation-preprocessing tool, not a clinical segmentation system.
HU alone identifies air, adipose tissue, and high-density bone reasonably well,
but muscle overlaps abdominal organs and SAT has the same HU range as VAT.
Consequently, this script uses body depth as a transparent anatomical proxy:

* fat within ``--boundary-depth-mm`` of the body surface is SAT; deeper fat is
  VAT;
* non-fat soft tissue near the surface is muscle; deeper soft tissue is
  represented as visceral organs.

Default HU ranges:
    air                  HU < -190
    adipose tissue       -190 <= HU <= -30
    soft tissue          -29 <= HU < 200
    bone                 HU >= 200

The commonly published skeletal-muscle band is -29 to +150 HU. This script
keeps the 151 to 199 HU transition band with soft tissue rather than calling it
bone; use ``--bone-hu`` to change that modeling choice.

The adipose and skeletal-muscle ranges follow commonly used abdominal CT body
composition thresholds. The representative sound speeds are from Table 1 of
Culjat et al., Ultrasound Med Biol. 2010;36:861-873:
air 330, fat 1478, muscle 1547, liver 1595, and cortical bone 3476 m/s. Liver
is used as the representative visceral-organ value. SAT and VAT share the fat
sound speed because location, not composition, distinguishes them here.

Three HU-to-sound-speed mappings are available:
* ``categorical`` assigns the representative tissue values above;
* ``regression`` converts conventional HU to density using the four-part
  k-Wave/Schneider fit, then applies the Mast soft-tissue relationship
  ``c = (density + 349) / 0.893``;
* ``hybrid`` uses that regression for fat and soft tissue, but keeps separate
  air/water and bone rules. This is the recommended abdominal simulation mode.

The published k-Wave coefficients use a CT-number scale on which water is near
1000. DICOM HU are therefore shifted by +1000 before applying the fit.

Supported input:
* DICOM CT (requires pydicom; applies RescaleSlope and RescaleIntercept)
* .npy arrays containing HU values
* PNG/TIFF/JPEG images, with ``--slope`` and ``--intercept`` if pixels are not
  already HU. Windowed screenshots cannot be converted back to quantitative HU.

Examples
--------
    python ct_to_speed.py abdomen.dcm --output abdomen_acoustic.npz
    python ct_to_speed.py abdomen.dcm --air-as-water
    python ct_to_speed.py abdomen.dcm --mapping hybrid --air-as-water
    python ct_to_speed.py abdomen_hu.npy --pixel-spacing-mm 0.8
    python ct_to_speed.py ct.tiff --slope 1 --intercept -1024
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


AIR = 0
BONE = 1
MUSCLE = 2
VISCERAL_ORGANS = 3
SAT = 4
VAT = 5

TISSUE_NAMES = np.array(
    [
        "Air",
        "Bone",
        "Muscle",
        "Visceral organs",
        "Subcutaneous adipose tissue",
        "Visceral adipose tissue",
    ]
)

# Representative longitudinal sound speed in m/s.
SOUND_SPEED_M_S = np.array([330.0, 3476.0, 1547.0, 1595.0, 1478.0, 1478.0])
WATER_SPEED_M_S = 1480.0
AIR_DENSITY_KG_M3 = 1.2
WATER_DENSITY_KG_M3 = 1000.0

MAPPING_CATEGORICAL = "categorical"
MAPPING_REGRESSION = "regression"
MAPPING_HYBRID = "hybrid"
MAPPING_CHOICES = (MAPPING_CATEGORICAL, MAPPING_REGRESSION, MAPPING_HYBRID)


@dataclass(frozen=True)
class CTImage:
    hu: np.ndarray
    spacing_mm: tuple[float, float]


def _as_2d(array: np.ndarray, slice_index: int) -> np.ndarray:
    """Select a 2D slice from a 2D or multi-frame image."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if not -array.shape[0] <= slice_index < array.shape[0]:
            raise IndexError(
                f"slice index {slice_index} is outside 0..{array.shape[0] - 1}"
            )
        return array[slice_index]
    raise ValueError(f"Expected a 2D image or 3D stack, got shape {array.shape}.")


def load_ct(
    path: str | Path,
    slice_index: int = 0,
    slope: float = 1.0,
    intercept: float = 0.0,
    pixel_spacing_mm: tuple[float, float] | None = None,
) -> CTImage:
    """Load a CT slice and return calibrated HU and row/column spacing."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError(
                "DICOM input requires pydicom. Install it with "
                "`python -m pip install pydicom`, or provide a HU-valued .npy file."
            ) from exc

        dataset = pydicom.dcmread(path)
        raw = _as_2d(dataset.pixel_array, slice_index).astype(np.float64)
        dicom_slope = float(getattr(dataset, "RescaleSlope", 1.0))
        dicom_intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        hu = raw * dicom_slope + dicom_intercept
        if pixel_spacing_mm is None:
            spacing = getattr(dataset, "PixelSpacing", (1.0, 1.0))
            pixel_spacing_mm = (float(spacing[0]), float(spacing[1]))
    elif suffix == ".npy":
        hu = _as_2d(np.load(path), slice_index).astype(np.float64)
        hu = hu * float(slope) + float(intercept)
    else:
        raw = np.asarray(Image.open(path))
        if raw.ndim == 3:
            if raw.shape[-1] in (3, 4):
                if not np.all(raw[..., :3] == raw[..., :1]):
                    raise ValueError(
                        "Color input is not a quantitative CT image. Export the "
                        "original DICOM or a single-channel HU-valued image."
                    )
                raw = raw[..., 0]
            else:
                raw = _as_2d(raw, slice_index)
        hu = raw.astype(np.float64) * float(slope) + float(intercept)

    if pixel_spacing_mm is None:
        pixel_spacing_mm = (1.0, 1.0)
    if len(pixel_spacing_mm) != 2 or min(pixel_spacing_mm) <= 0.0:
        raise ValueError("Pixel spacing must contain two positive values in mm.")
    if hu.ndim != 2 or not np.all(np.isfinite(hu)):
        raise ValueError("The calibrated CT slice must be a finite 2D array.")

    return CTImage(hu=hu, spacing_mm=tuple(map(float, pixel_spacing_mm)))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected foreground component."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def body_mask_from_hu(hu: np.ndarray, body_threshold_hu: float = -500.0) -> np.ndarray:
    """Estimate the patient's filled 2D body envelope."""
    body = largest_component(hu > body_threshold_hu)
    body = ndimage.binary_closing(body, structure=np.ones((5, 5), dtype=bool))
    return ndimage.binary_fill_holes(body)


def segment_tissues(
    hu: np.ndarray,
    spacing_mm: tuple[float, float] = (1.0, 1.0),
    boundary_depth_mm: float = 25.0,
    fat_hu: tuple[float, float] = (-190.0, -30.0),
    bone_hu: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, body mask, and the estimated visceral-cavity mask.

    Every pixel receives one of the six requested labels. Background and
    internal gas are Air. The depth-based split is deliberately simple and is
    intended for constructing an acoustic phantom, not medical measurement.
    """
    if boundary_depth_mm <= 0.0:
        raise ValueError("boundary_depth_mm must be positive.")
    fat_low, fat_high = map(float, fat_hu)
    if fat_low >= fat_high or fat_high >= bone_hu:
        raise ValueError("Expected fat_low < fat_high < bone_hu.")

    body = body_mask_from_hu(hu)
    if not np.any(body):
        raise ValueError("No body-sized foreground component was found in the CT.")

    depth_mm = ndimage.distance_transform_edt(body, sampling=spacing_mm)
    cavity = body & (depth_mm > float(boundary_depth_mm))

    labels = np.full(hu.shape, AIR, dtype=np.uint8)
    fat = body & (hu >= fat_low) & (hu <= fat_high)
    bone = body & (hu >= float(bone_hu))
    # Do not let gas pockets enclosed by the filled body contour fall through
    # to soft tissue. Values between fat and bone are the soft-tissue band.
    soft = body & (hu > fat_high) & (hu < float(bone_hu))

    labels[soft & ~cavity] = MUSCLE
    labels[soft & cavity] = VISCERAL_ORGANS
    labels[fat & ~cavity] = SAT
    labels[fat & cavity] = VAT
    labels[bone] = BONE
    return labels, body, cavity


def sound_speed_lookup(air_as_water: bool = False) -> np.ndarray:
    """Return the tissue lookup, optionally replacing air with water."""
    lookup = SOUND_SPEED_M_S.copy()
    if air_as_water:
        lookup[AIR] = WATER_SPEED_M_S
    return lookup


def labels_to_speed(
    labels: np.ndarray,
    air_as_water: bool = False,
) -> np.ndarray:
    """Map tissue labels to representative sound speed in m/s."""
    labels = np.asarray(labels)
    if labels.size and (labels.min() < 0 or labels.max() >= len(SOUND_SPEED_M_S)):
        raise ValueError("Tissue label is outside the sound-speed lookup table.")
    return sound_speed_lookup(air_as_water=air_as_water)[labels]


def hu_to_density_kwave(hu: np.ndarray) -> np.ndarray:
    """Convert conventional HU to density using the k-Wave piecewise fit.

    k-Wave reproduces a four-part fit to the Schneider et al. CT calibration.
    Its independent variable is a CT-number scale with water near 1000, so
    conventional DICOM HU must first be shifted by +1000.
    """
    ct_number = np.asarray(hu, dtype=float) + 1000.0
    density = np.empty_like(ct_number)

    part1 = ct_number < 930.0
    part2 = (ct_number >= 930.0) & (ct_number <= 1098.0)
    part3 = (ct_number > 1098.0) & (ct_number < 1260.0)
    part4 = ct_number >= 1260.0

    density[part1] = (
        1.025793065681423 * ct_number[part1] - 5.680404011488714
    )
    density[part2] = (
        0.9082709691264 * ct_number[part2] + 103.6151457847139
    )
    density[part3] = (
        0.5108369316599 * ct_number[part3] + 539.9977189228704
    )
    density[part4] = (
        0.6625370912451 * ct_number[part4] + 348.8555178455294
    )
    return density


def density_to_speed_mast(density_kg_m3: np.ndarray) -> np.ndarray:
    """Approximate soft-tissue sound speed from density (Mast, 2000)."""
    return (np.asarray(density_kg_m3, dtype=float) + 349.0) / 0.893


def representative_speeds(labels: np.ndarray, speed_m_s: np.ndarray) -> np.ndarray:
    """Median speed in each label, with categorical fallback if absent."""
    representative = SOUND_SPEED_M_S.copy()
    for index in range(len(TISSUE_NAMES)):
        values = speed_m_s[labels == index]
        if values.size:
            representative[index] = float(np.median(values))
    return representative


def map_hu_to_speed(
    hu: np.ndarray,
    labels: np.ndarray,
    mapping: str = MAPPING_CATEGORICAL,
    air_as_water: bool = False,
    bone_speed_m_s: float = SOUND_SPEED_M_S[BONE],
    speed_clip_m_s: tuple[float, float] = (1200.0, 4000.0),
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Create a sound-speed map using categorical, regression, or hybrid rules.

    Returns ``(speed, density, representative_speeds)``. Density is ``None``
    for categorical mapping. The Mast relation is an empirical soft-tissue
    approximation; hybrid mode avoids applying it to bone.
    """
    if mapping not in MAPPING_CHOICES:
        raise ValueError(f"Unknown mapping {mapping!r}; choose from {MAPPING_CHOICES}.")
    if not np.isfinite(bone_speed_m_s) or bone_speed_m_s <= 0.0:
        raise ValueError("bone_speed_m_s must be positive.")
    clip_low, clip_high = map(float, speed_clip_m_s)
    if clip_low <= 0.0 or clip_low >= clip_high:
        raise ValueError("Expected 0 < speed clip LOW < HIGH.")

    labels = np.asarray(labels)
    if labels.shape != np.asarray(hu).shape:
        raise ValueError("HU and tissue labels must have matching shapes.")
    if mapping == MAPPING_CATEGORICAL:
        speed = labels_to_speed(labels, air_as_water=air_as_water)
        return speed, None, sound_speed_lookup(air_as_water=air_as_water)

    density = hu_to_density_kwave(hu)
    speed = density_to_speed_mast(density)
    non_air = labels != AIR
    speed[non_air] = np.clip(speed[non_air], clip_low, clip_high)

    air_speed = WATER_SPEED_M_S if air_as_water else SOUND_SPEED_M_S[AIR]
    speed[~non_air] = air_speed
    density[~non_air] = (
        WATER_DENSITY_KG_M3 if air_as_water else AIR_DENSITY_KG_M3
    )

    if mapping == MAPPING_HYBRID:
        speed[labels == BONE] = float(bone_speed_m_s)

    return speed, density, representative_speeds(labels, speed)


def save_results(
    path: str | Path,
    ct: CTImage,
    labels: np.ndarray,
    speed_m_s: np.ndarray,
    body_mask: np.ndarray,
    cavity_mask: np.ndarray,
    speed_lookup_m_s: np.ndarray | None = None,
    density_kg_m3: np.ndarray | None = None,
    mapping: str = MAPPING_CATEGORICAL,
    bone_speed_m_s: float = SOUND_SPEED_M_S[BONE],
    speed_clip_m_s: tuple[float, float] = (1200.0, 4000.0),
) -> Path:
    """Save calibrated input, segmentation, speed map, and metadata as NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if speed_lookup_m_s is None:
        speed_lookup_m_s = SOUND_SPEED_M_S
    arrays = dict(
        hu=ct.hu.astype(np.float32),
        labels=labels,
        speed_m_s=speed_m_s.astype(np.float32),
        body_mask=body_mask,
        cavity_mask=cavity_mask,
        spacing_mm=np.asarray(ct.spacing_mm),
        tissue_names=TISSUE_NAMES,
        tissue_sound_speed_m_s=np.asarray(speed_lookup_m_s, dtype=float),
        mapping=np.asarray(mapping),
        mapping_bone_speed_m_s=np.asarray(float(bone_speed_m_s)),
        mapping_speed_clip_m_s=np.asarray(speed_clip_m_s, dtype=float),
        regression_name=np.asarray("k-Wave/Schneider density + Mast sound speed"),
    )
    if density_kg_m3 is not None:
        arrays["density_kg_m3"] = np.asarray(density_kg_m3, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return path


def plot_results(
    hu: np.ndarray,
    labels: np.ndarray,
    speed_m_s: np.ndarray,
    output: str | Path | None = None,
    mapping: str = MAPPING_CATEGORICAL,
):
    """Create a three-panel quality-control image."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = ["#101010", "#f2f2f2", "#d95f02", "#1b9e77", "#e6ab02", "#7570b3"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(hu, cmap="gray", vmin=-200, vmax=300)
    axes[0].set_title("CT (−200 to 300 HU)")
    axes[1].imshow(labels, cmap=cmap, norm=norm)
    axes[1].set_title("Tissue classes")
    image = axes[2].imshow(
        speed_m_s,
        cmap="turbo",
        vmin=float(np.min(speed_m_s)),
        vmax=float(np.max(speed_m_s)),
    )
    axes[2].set_title(f"Speed of sound (m/s): {mapping}")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_axis_off()
    axes[1].legend(
        handles=[Patch(color=colors[i], label=name) for i, name in enumerate(TISSUE_NAMES)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        fontsize=8,
        frameon=False,
    )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def _spacing_arg(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) == 2:
        return (values[0], values[1])
    raise ValueError("--pixel-spacing-mm accepts one value or ROW COL.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment an abdominal CT slice and create a sound-speed map."
    )
    parser.add_argument("input", type=Path, help="DICOM, .npy, or grayscale image")
    parser.add_argument("--output", type=Path, help="Output .npz path")
    parser.add_argument("--figure", type=Path, help="QC figure path (PNG recommended)")
    parser.add_argument("--slice-index", type=int, default=0, help="Frame for a 3D input")
    parser.add_argument("--slope", type=float, default=1.0, help="Non-DICOM HU slope")
    parser.add_argument("--intercept", type=float, default=0.0, help="Non-DICOM HU intercept")
    parser.add_argument(
        "--pixel-spacing-mm",
        type=float,
        nargs="+",
        metavar=("ROW", "COL"),
        help="One isotropic value or row and column spacing; DICOM reads metadata",
    )
    parser.add_argument(
        "--boundary-depth-mm",
        type=float,
        default=25.0,
        help="Depth proxy separating body wall/SAT from viscera/VAT (default: 25)",
    )
    parser.add_argument(
        "--fat-hu",
        type=float,
        nargs=2,
        default=(-190.0, -30.0),
        metavar=("LOW", "HIGH"),
        help="Inclusive adipose-tissue HU range (default: -190 -30)",
    )
    parser.add_argument(
        "--bone-hu",
        type=float,
        default=200.0,
        help="Lower HU threshold for bone (default: 200)",
    )
    parser.add_argument(
        "--air-as-water",
        action="store_true",
        help="Assign Air-labeled pixels the water speed, 1480 m/s, instead of 330",
    )
    parser.add_argument(
        "--mapping",
        choices=MAPPING_CHOICES,
        default=MAPPING_CATEGORICAL,
        help=(
            "HU-to-speed mapping: categorical tissue constants, continuous "
            "regression, or regression with separate air/bone rules "
            "(default: categorical)"
        ),
    )
    parser.add_argument(
        "--bone-speed",
        type=float,
        default=float(SOUND_SPEED_M_S[BONE]),
        metavar="M_S",
        help="Bone speed used by hybrid mapping (default: 3476 m/s)",
    )
    parser.add_argument(
        "--speed-clip",
        type=float,
        nargs=2,
        default=(1200.0, 4000.0),
        metavar=("LOW", "HIGH"),
        help="Non-air regression speed limits in m/s (default: 1200 4000)",
    )
    parser.add_argument("--show", action="store_true", help="Show the QC figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spacing = _spacing_arg(args.pixel_spacing_mm)
    ct = load_ct(
        args.input,
        slice_index=args.slice_index,
        slope=args.slope,
        intercept=args.intercept,
        pixel_spacing_mm=spacing,
    )
    labels, body, cavity = segment_tissues(
        ct.hu,
        spacing_mm=ct.spacing_mm,
        boundary_depth_mm=args.boundary_depth_mm,
        fat_hu=tuple(args.fat_hu),
        bone_hu=args.bone_hu,
    )
    speed, density, speed_lookup = map_hu_to_speed(
        ct.hu,
        labels,
        mapping=args.mapping,
        air_as_water=args.air_as_water,
        bone_speed_m_s=args.bone_speed,
        speed_clip_m_s=tuple(args.speed_clip),
    )

    output = args.output or args.input.with_name(f"{args.input.stem}_acoustic.npz")
    figure = args.figure or output.with_suffix(".png")
    save_results(
        output,
        ct,
        labels,
        speed,
        body,
        cavity,
        speed_lookup_m_s=speed_lookup,
        density_kg_m3=density,
        mapping=args.mapping,
        bone_speed_m_s=args.bone_speed,
        speed_clip_m_s=tuple(args.speed_clip),
    )
    fig = plot_results(ct.hu, labels, speed, output=figure, mapping=args.mapping)

    print(f"Loaded:  {args.input}")
    print(f"Shape:   {ct.hu.shape}; spacing: {ct.spacing_mm} mm")
    print(f"Mapping: {args.mapping}")
    print(f"Saved:   {output}")
    print(f"Figure:  {figure}")
    for index, name in enumerate(TISSUE_NAMES):
        count = int(np.count_nonzero(labels == index))
        fraction = 100.0 * count / labels.size
        values = speed[labels == index]
        if values.size:
            speed_text = (
                f"c={values.min():.0f}/{np.median(values):.0f}/"
                f"{values.max():.0f} m/s (min/median/max)"
            )
        else:
            speed_text = f"c={speed_lookup[index]:.0f} m/s"
        print(f"  {name:30s} {count:9d} px ({fraction:6.2f}%)  {speed_text}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
