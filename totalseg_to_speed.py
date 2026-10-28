"""Build an abdominal sound-speed slice from an open TotalSegmentator segmentation.

The TotalSegmentator CT dataset (Wasserthal et al., Radiology: AI 2023;
CC BY 4.0) provides CT volumes with expert-reviewed masks of 117 structures.
This script takes one axial slice of one subject, merges the anatomical masks
into tissue classes, and writes the same .npz format as ct_to_speed.py, so the
result loads directly with ``--ct-speed`` in 2DRingFDTD.py, invert_ring.py,
invert_arc.py and fdfd_ring/.

Compared with ct_to_speed.py, organs, vessels, bowel, bone and the major trunk
muscles come from the segmentation instead of HU thresholds and a skin-depth
rule. Only body pixels that no mask covers are classified by HU: fat
(-190 to -30 HU), gas (< -190 HU, replaced by water) or other soft tissue.

Acoustic properties come from abdominal_acoustic_parameters.csv (speed,
density, power-law attenuation and B/A per tissue; values compiled from Duck,
Physical Properties of Tissue, 1990; Szabo, Diagnostic Ultrasound Imaging:
Inside Out; and Mast, ARLO 2000). TISSUE_CLASSES names the table row each
class uses. The solvers are lossless and constant-density, so only speed
enters them; the density map and attenuation parameters are stored for later
models. Gas is modelled as water, not 343 m/s air: without a density contrast a
scalar model cannot reproduce the near-total reflection at a gas boundary.

Fetching a subject reads only that subject's files out of the 3.2 GB
"Small subset of TotalSegmentator Dataset" zip on Zenodo with HTTP range
requests (about 44 MB for s1366).

Examples
--------
    python totalseg_to_speed.py --fetch s1366 --data-dir data/totalseg
    python totalseg_to_speed.py --subject-dir data/totalseg/s1366 \
        --output totalseg_s1366_acoustic.npz --figure totalseg_s1366_acoustic.png
    python invert_ring.py --ct-speed totalseg_s1366_acoustic.npz --multiscale
"""

from __future__ import annotations

import argparse
import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ct_to_speed import body_mask_from_hu

SMALL_SUBSET_URL = (
    "https://zenodo.org/records/10047263/files/"
    "Totalsegmentator_dataset_small_v201.zip?download=1"
)

FAT_HU = (-190.0, -30.0)
# Median filter (pixels) applied to HU before classifying unlabelled tissue;
# low-dose scans otherwise scatter fat/soft-tissue/gas speckle.
HU_MEDIAN_PIXELS = 5
ACOUSTIC_TABLE = Path(__file__).with_name("abdominal_acoustic_parameters.csv")

# (class name, row of ACOUSTIC_TABLE, TotalSegmentator mask-name prefixes).
# Mask prefixes must be unique. Order sets the label index; masks are applied
# in file order and overwrite unlabelled (HU-classified) pixels.
TISSUE_CLASSES = (
    ("Water (outside body, gas)", "Water / Bile / Urine", ()),
    ("Fat", "Fat (Visceral/Subcutaneous)", ()),
    ("Other soft tissue", "Soft Tissue (Machine Average)", ()),
    ("Muscle", "Muscle (Abdominal Wall)", ("autochthon", "iliopsoas", "gluteus")),
    ("Digestive tract", "Soft Tissue (Machine Average)",
     ("stomach", "duodenum", "small_bowel", "colon", "esophagus")),
    ("Liver", "Liver", ("liver",)),
    ("Spleen", "Spleen", ("spleen",)),
    ("Kidney", "Kidney", ("kidney",)),
    ("Other organs", "Soft Tissue (Machine Average)", ("pancreas", "adrenal_gland", "prostate")),
    ("Fluid (bile, urine)", "Water / Bile / Urine", ("gallbladder", "urinary_bladder")),
    ("Blood vessels", "Blood", ("aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
                                "iliac_artery", "iliac_vena")),
    ("Bone", "Bone / Cortical Rib", ("vertebrae", "rib", "sacrum", "hip", "femur", "sternum")),
)
WATER, FAT, OTHER_SOFT = 0, 1, 2
DIGESTIVE = 4
PROPERTY_COLUMNS = {
    "speed": "Speed of Sound (m/s)",
    "density": "Density (kg/m3)",
    "alpha0": "Attenuation Alpha0 (dB/MHz^y-cm)",
    "power": "Power Law Exponent y",
    "b_over_a": "Non-linearity Parameter B/A",
}


def load_acoustic_table(path=ACOUSTIC_TABLE):
    """Per-class property arrays (speed, density, ...) read from the CSV table."""
    import csv
    with open(path, newline="") as handle:
        rows = {row["Tissue Type"].strip(): row for row in csv.DictReader(handle)}
    missing = sorted({row for _, row, _ in TISSUE_CLASSES} - set(rows))
    if missing:
        raise SystemExit(f"{path} has no row for: {', '.join(missing)}")
    return {key: np.array([float(rows[row][column]) for _, row, _ in TISSUE_CLASSES])
            for key, column in PROPERTY_COLUMNS.items()}


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests."""

    def __init__(self, url, block=4 << 20):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Length"])
        self.pos, self.block, self.cache, self.fetched = 0, block, {}, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, index):
        if index not in self.cache:
            start = index * self.block
            end = min(start + self.block, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request) as response:
                if response.status != 206:
                    raise RuntimeError("The server ignored the HTTP range request.")
                self.cache[index] = response.read()
            self.fetched += end - start + 1
        return self.cache[index]

    def readinto(self, buffer):
        count = min(len(buffer), max(self.size - self.pos, 0))
        view, done = memoryview(buffer), 0
        while done < count:
            index, offset = divmod(self.pos, self.block)
            chunk = self._block(index)[offset:offset + count - done]
            view[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done


def fetch_subject(subject, data_dir, url=SMALL_SUBSET_URL):
    """Extract one subject folder (CT + masks) from the remote dataset zip."""
    remote = HTTPRangeFile(url)
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=4 << 20))
    members = [m for m in archive.infolist()
               if m.filename.startswith(f"{subject}/") and not m.is_dir()]
    if not members:
        raise SystemExit(f"Subject {subject!r} is not in the dataset archive.")
    for member in members:
        archive.extract(member, data_dir)
    print(f"Fetched {len(members)} files for {subject} "
          f"({remote.fetched / 1e6:.1f} MB transferred)")
    return Path(data_dir) / subject


NIFTI_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def read_nifti(path):
    """Minimal NIfTI-1 reader: returns (data in index order x, y, z; voxel size mm)."""
    raw = gzip.open(path).read() if str(path).endswith(".gz") else Path(path).read_bytes()
    endian = "<" if np.frombuffer(raw, "<i4", 1, 0)[0] == 348 else ">"
    dim = np.frombuffer(raw, endian + "i2", 8, 40)
    datatype = int(np.frombuffer(raw, endian + "i2", 1, 70)[0])
    pixdim = np.frombuffer(raw, endian + "f4", 8, 76)
    offset = int(np.frombuffer(raw, endian + "f4", 1, 108)[0])
    slope, intercept = np.frombuffer(raw, endian + "f4", 2, 112)
    shape = tuple(int(d) for d in dim[1:1 + dim[0]])
    data = np.frombuffer(raw, endian + NIFTI_DTYPES[datatype], int(np.prod(shape)), offset)
    data = data.reshape(shape, order="F")
    if slope not in (0.0, 1.0) or intercept != 0.0:
        data = data * slope + intercept
    srow = np.vstack([np.frombuffer(raw, endian + "f4", 4, o) for o in (280, 296, 312)])
    if np.any(np.diag(srow[:, :3]) < 0) or np.any(srow[:, :3] - np.diag(np.diag(srow[:, :3]))):
        raise ValueError(f"{path}: only axis-aligned volumes with positive spacing are supported.")
    return np.asarray(data), tuple(float(v) for v in pixdim[1:4])


def to_radiological(axial):
    """(x, y) RAS-indexed slice -> image rows anterior-to-posterior, columns right-to-left.

    This matches the DICOM-derived maps written by ct_to_speed.py.
    """
    return np.ascontiguousarray(axial.T[::-1, ::-1])


# --------------------------------------------------------------------------
# Slice selection and tissue classes
# --------------------------------------------------------------------------

KEY_STRUCTURES = ("liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach",
                  "aorta", "inferior_vena_cava", "autochthon_left", "autochthon_right")


def class_of(mask_name):
    for index, (_, _, prefixes) in enumerate(TISSUE_CLASSES):
        if any(mask_name.startswith(prefix) for prefix in prefixes):
            return index
    return None


def gas_fraction(hu):
    body = body_mask_from_hu(hu)
    smooth = ndimage.median_filter(hu, size=HU_MEDIAN_PIXELS)
    return float((body & (smooth < FAT_HU[0])).sum() / max(body.sum(), 1))


def choose_slice(subject_dir, min_pixels=50):
    """Slice showing the most key structures, with the least gas among those.

    Gas (bowel, stomach) becomes water in the scalar model, so slices
    through a gas-filled colon make poor sound-speed demos.
    Returns (slice index, key structures present, gas fraction of body).
    """
    counts = None
    for name in KEY_STRUCTURES:
        mask, _ = read_nifti(subject_dir / "segmentations" / f"{name}.nii.gz")
        present = (mask > 0).sum(axis=(0, 1)) >= min_pixels
        counts = present.astype(int) if counts is None else counts + present
    ct, _ = read_nifti(subject_dir / "ct.nii.gz")
    candidates = np.flatnonzero(counts == counts.max())
    gas = [gas_fraction(to_radiological(ct[:, :, z].astype(np.float32))) for z in candidates]
    best = int(np.argmin(gas))
    return int(candidates[best]), int(counts.max()), gas[best]


def build_slice(subject_dir, z):
    ct, voxel_mm = read_nifti(subject_dir / "ct.nii.gz")
    hu = to_radiological(ct[:, :, z].astype(np.float32))
    body = body_mask_from_hu(hu)

    smooth = ndimage.median_filter(hu, size=HU_MEDIAN_PIXELS)
    labels = np.full(hu.shape, WATER, dtype=np.uint8)
    labels[body & (smooth >= FAT_HU[0]) & (smooth <= FAT_HU[1])] = FAT
    labels[body & (smooth > FAT_HU[1])] = OTHER_SOFT
    structures = {}
    for path in sorted((subject_dir / "segmentations").glob("*.nii.gz")):
        name = path.name[:-len(".nii.gz")]
        tissue = class_of(name)
        if tissue is None:
            continue
        mask, _ = read_nifti(path)
        mask = to_radiological(mask[:, :, z] > 0) & body
        if mask.any():
            structures[name] = int(mask.sum())
            labels[mask] = tissue
    # Gas inside the bowel or stomach cannot be modelled by the scalar
    # fluid solver; treat it like the body's other gas pockets. Segmented
    # solid tissue keeps its class (low HU there is noise, not gas).
    gas_allowed = np.isin(labels, [WATER, FAT, OTHER_SOFT, DIGESTIVE])
    labels[body & gas_allowed & (smooth < FAT_HU[0])] = WATER

    spacing_mm = (voxel_mm[1], voxel_mm[0])  # (row, column)
    return hu, labels, body, spacing_mm, structures


# Speed colour window for figures: soft tissue spans ~1450-1580 m/s while bone
# is 3750 m/s, so a linear full-range scale hides all soft-tissue contrast.
DISPLAY_RANGE_M_S = (1420.0, 1620.0)


def save_figure(path, hu, labels, speed, spacing_mm, title, speed_range=DISPLAY_RANGE_M_S):
    rows, cols = hu.shape
    extent = [0, cols * spacing_mm[1], rows * spacing_mm[0], 0]
    names = [name for name, _, _ in TISSUE_CLASSES]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    axes[0].imshow(hu, cmap="gray", vmin=-200, vmax=300, extent=extent)
    axes[0].set_title("CT (-200 to 300 HU)")
    cmap = plt.get_cmap("tab20", len(names))
    image = axes[1].imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5,
                           interpolation="nearest", extent=extent)
    bar = fig.colorbar(image, ax=axes[1], ticks=range(len(names)), fraction=0.046)
    bar.ax.set_yticklabels(names, fontsize=8)
    axes[1].set_title("Tissue classes (TotalSegmentator + HU)")
    low, high = speed_range
    image = axes[2].imshow(speed, cmap="viridis", vmin=low, vmax=high, extent=extent)
    extend = ("min" if speed.min() < low else "neither") if speed.max() <= high else \
             ("both" if speed.min() < low else "max")
    fig.colorbar(image, ax=axes[2], label="m/s", fraction=0.046, extend=extend)
    axes[2].set_title(f"Speed of sound ({low:.0f}-{high:.0f} m/s window; bone saturates)")
    for axis in axes:
        axis.set(xlabel="mm", ylabel="mm")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch", metavar="SUBJECT",
                        help="Download one subject (e.g. s1366) from the Zenodo small subset")
    source.add_argument("--subject-dir", type=Path,
                        help="Local TotalSegmentator subject folder (ct.nii.gz + segmentations/)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/totalseg"),
                        help="Where --fetch stores the subject (default: data/totalseg)")
    parser.add_argument("--slice", type=int, metavar="Z",
                        help="Axial slice index (default: the slice showing most key organs)")
    parser.add_argument("--output", type=Path, help="Output .npz (default: totalseg_<subject>_acoustic.npz)")
    parser.add_argument("--figure", type=Path, help="Optional PNG of CT, classes and speed")
    parser.add_argument("--acoustic-table", type=Path, default=ACOUSTIC_TABLE,
                        help="CSV of tissue acoustic properties (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    subject_dir = (fetch_subject(args.fetch, args.data_dir) if args.fetch
                   else args.subject_dir)
    subject = subject_dir.name
    if args.slice is None:
        z, count, gas = choose_slice(subject_dir)
        print(f"Slice {z}: {count} of {len(KEY_STRUCTURES)} key structures present, "
              f"gas {100 * gas:.1f}% of body")
    else:
        z = args.slice
    table = load_acoustic_table(args.acoustic_table)
    hu, labels, body, spacing_mm, structures = build_slice(subject_dir, z)
    speed = table["speed"][labels].astype(np.float32)
    output = args.output or Path(f"totalseg_{subject}_acoustic.npz")
    names = np.array([name for name, _, _ in TISSUE_CLASSES])
    np.savez_compressed(
        output,
        hu=hu, labels=labels, speed_m_s=speed, body_mask=body,
        density_kg_m3=table["density"][labels].astype(np.float32),
        spacing_mm=np.asarray(spacing_mm, dtype=float),
        tissue_names=names,
        tissue_table_rows=np.array([row for _, row, _ in TISSUE_CLASSES]),
        tissue_sound_speed_m_s=table["speed"],
        tissue_density_kg_m3=table["density"],
        tissue_alpha0_db_mhz_y_cm=table["alpha0"],
        tissue_power_law_y=table["power"],
        tissue_b_over_a=table["b_over_a"],
        structure_names=np.array(sorted(structures)),
        source=np.asarray(f"TotalSegmentator small subset v2.0.1 (CC BY 4.0), "
                          f"subject {subject}, axial slice {z}"),
    )
    shares = np.bincount(labels[body], minlength=len(names)) / body.sum()
    print(f"Saved {output}: {hu.shape[0]} x {hu.shape[1]} at "
          f"{spacing_mm[0]:g} x {spacing_mm[1]:g} mm, c = {speed.min():.0f}-{speed.max():.0f} m/s")
    for name, share, c in zip(names, shares, table["speed"]):
        print(f"  {name:28s} {c:6.0f} m/s  {100 * share:5.1f}% of body")
    print(f"  Segmented structures on this slice: {', '.join(sorted(structures))}")
    if args.figure:
        save_figure(args.figure, hu, labels, speed, spacing_mm,
                    f"TotalSegmentator {subject}, axial slice {z}")
        print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
