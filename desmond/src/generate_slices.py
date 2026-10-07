"""Generate synthetic abdominal CT landmark slices with MAISI rflow-ct.

MAISI (NV-Generate-CTMR) only generates 3D volumes, so each sample is a full
paired volume: a 132-class anatomy mask from the mask diffusion model (DDPM),
then a CT image conditioned on that mask by the ControlNet + rectified-flow
image model. The mask labels individual vertebrae, so each landmark slice
(default: mid-L3) is located from the labels rather than guessed.

The model repo is imported read-only (no bytecode is written to it) and all
outputs go under this repo. ``scripts.inference`` is not used because it
filters the saved mask down to the size-controlled organs, which drops the
vertebrae, and it writes into the model repo through relative paths.

Each sample folder contains, per landmark slice:

* ``<name>_hu.npy``     int16 HU slice, radiological orientation (rows run
                        anterior -> posterior, columns run patient right -> left,
                        like a DICOM axial image); input for ``ct_to_speed.py``
* ``<name>_label.npy``  uint8 MAISI labels for the same slice
* ``meta.json``         seed, model commit, slice index, spacing, ...

and optionally the full RAS volumes (``volume_image.nii.gz``,
``volume_label.nii.gz``) and a QC ``preview.png``.

Examples
--------
    python desmond/src/generate_slices.py --config desmond/configs/slices.yaml
    python desmond/src/generate_slices.py --num-volumes 3 --run-name l3_try
    python ct_to_speed.py desmond/data/generated/<run>/sample_0000/L3_mid_hu.npy \\
        --pixel-spacing-mm 0.781 --mapping hybrid --air-as-water
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# desmond/ (this student workspace): configs, data and docs live here;
# the advisor's code is in the repository root one level up.
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "slices.yaml"

HF_REPO_ID = "nvidia/NV-Generate-CT"
ANATOMY_SIZE_FILE = "all_anatomy_size_conditions.json"
# Path B (retrieved masks): MAISI's database of real training masks, under <aux_cache>/datasets/.
MASK_DB_JSON = "candidate_masks_flexible_size_and_spacing_4000.json"
MASK_DB_ZIP = "all_masks_flexible_size_and_spacing_4000.zip"
MASK_SOURCES = ("generated", "retrieved")
# The mask diffusion model always generates 256^3 voxels at 1.5 mm (sample.py TODO).
MASK_DM_FOV_MM = (256 * 1.5, 256 * 1.5)
CHECKPOINTS = {
    "autoencoder": "models/autoencoder_v1.pt",
    "diffusion_unet": "models/diff_unet_3d_rflow-ct.pt",
    "controlnet": "models/controlnet_3d_rflow-ct.pt",
    "mask_autoencoder": "models/mask_generation_autoencoder.pt",
    "mask_diffusion_unet": "models/mask_generation_diffusion_unet.pt",
}
# Slots of the 10-d anatomy-size condition (scripts/sample.py); -1 means absent.
N_ORGAN_SLOTS = 5  # gallbladder, liver, stomach, pancreas, colon; slots 5..9 are tumors
TUMOR_LABEL_NAMES = (
    "lung tumor",
    "pancreatic tumor",
    "hepatic tumor",
    "colon cancer primaries",
    "bone lesion",
)
HU_RANGE = (-1000, 1000)
SEED_ATTEMPT_STRIDE = 100_000
ORIENTATION_NOTE = (
    "Slices: rows anterior->posterior, columns patient right->left "
    "(radiological, DICOM-like); spacing_mm is (row, col). "
    "Volumes: RAS voxel axes (x->right, y->anterior, z->superior), "
    "slice = volume[:, :, slice_index]."
)

logger = logging.getLogger("generate_slices")


class SampleRejected(RuntimeError):
    """A generated volume is unusable and should be re-drawn with a new seed."""


class LandmarkError(SampleRejected):
    """A landmark label is missing or truncated in a generated volume."""


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path=DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    """Load the YAML config, apply dotted-key overrides, and resolve paths."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = key.split(".")
        for name in parents:
            node = node[name]
        node[leaf] = value

    cfg["model"]["repo"] = str(_resolve(cfg["model"]["repo"]))
    cfg["model"]["aux_cache"] = str(_resolve(cfg["model"]["aux_cache"]))
    cfg["output"]["dir"] = str(_resolve(cfg["output"]["dir"]))

    gen = cfg["generation"]
    if gen["num_volumes"] < 1 or gen["num_volumes"] >= SEED_ATTEMPT_STRIDE:
        raise ValueError(f"num_volumes must be in [1, {SEED_ATTEMPT_STRIDE}).")
    if gen["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least 1.")
    if gen.get("mask_source", "generated") not in MASK_SOURCES:
        raise ValueError(f"generation.mask_source must be one of {MASK_SOURCES}.")
    names = [spec["name"] for spec in cfg["slices"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("slices must be a non-empty list with unique names.")
    for spec in cfg["slices"]:
        if spec.get("method", "centroid") != "centroid":
            raise ValueError(f"Unsupported slice method {spec['method']!r}.")
    return cfg


def load_label_dict(model_repo) -> dict[str, int]:
    with open(Path(model_repo) / "configs" / "label_dict.json") as f:
        return json.load(f)


def resolve_label_ids(slice_specs, label_dict) -> list[dict]:
    """Attach the integer MAISI label id to every slice spec."""
    resolved = []
    for spec in slice_specs:
        if spec["label"] not in label_dict:
            raise KeyError(f"Unknown MAISI label {spec['label']!r}.")
        resolved.append({**spec, "label_id": int(label_dict[spec["label"]])})
    return resolved


# ----------------------------------------------------------------------------
# Landmarks and slices (pure numpy)
# ----------------------------------------------------------------------------


def find_landmark_index(label_vol: np.ndarray, label_id: int) -> dict:
    """Locate the axial (z) index at the centroid of ``label_id``.

    ``label_vol`` is an (X, Y, Z) array. Returns the rounded centroid index,
    the z-extent of the label, and whether it touches the first/last slice of
    the labelled field of view (in which case the structure may be truncated).
    The labelled FOV, not the array, is the reference: resampling a mask onto a
    larger FOV pads empty slices, which would hide a cut-off structure.
    """
    z_counts = np.count_nonzero(label_vol == label_id, axis=(0, 1))
    present = np.flatnonzero(z_counts)
    if present.size == 0:
        raise LandmarkError(f"label {label_id} is not present in the volume")
    fov = np.flatnonzero(np.any(label_vol > 0, axis=(0, 1)))
    z_centroid = float(np.sum(np.arange(z_counts.size) * z_counts) / z_counts.sum())
    z_min, z_max = int(present[0]), int(present[-1])
    return {
        "index": int(round(z_centroid)),
        "z_centroid": z_centroid,
        "z_min": z_min,
        "z_max": z_max,
        "fov_z_min": int(fov[0]),
        "fov_z_max": int(fov[-1]),
        "touches_boundary": z_min <= fov[0] or z_max >= fov[-1],
    }


def to_radiological(slice_xy: np.ndarray) -> np.ndarray:
    """Convert an RAS (x, y) axial slice to radiological (row, col) display.

    Rows run anterior -> posterior and columns run patient right -> left,
    matching DICOM axial images such as the KiTS slice used by ct_to_speed.py.
    """
    return np.ascontiguousarray(slice_xy[::-1, ::-1].T)


def fov_bounds_px(source_fov_mm, out_size, spacing) -> list[tuple[int, int]]:
    """In-plane index range covered by a source FOV after MAISI's resampling.

    ensure_output_size_and_spacing resamples to ``spacing`` and then centre
    pads/crops to ``out_size`` (ResizeWithPadOrCrop), so a source FOV smaller
    than the output occupies a centred band and the rest is zero padding.
    """
    bounds = []
    for fov_mm, n_out, sp in zip(source_fov_mm, out_size, spacing):
        n = fov_mm / sp
        if n >= n_out - 0.5:
            bounds.append((0, n_out - 1))
        else:
            lo = int(np.floor((n_out - n) / 2))
            bounds.append((lo, int(np.ceil(lo + n)) - 1))
    return bounds


def body_cut_sides(slice_label: np.ndarray, fov_px, tol: int = 1) -> list[str]:
    """RAS sides where the labelled body reaches the source FOV edge (i.e. is cut)."""
    xs = np.flatnonzero(np.any(slice_label > 0, axis=1))
    ys = np.flatnonzero(np.any(slice_label > 0, axis=0))
    if xs.size == 0:
        return []
    (x_lo, x_hi), (y_lo, y_hi) = fov_px
    checks = {
        "left": xs[0] <= x_lo + tol,
        "right": xs[-1] >= x_hi - tol,
        "posterior": ys[0] <= y_lo + tol,
        "anterior": ys[-1] >= y_hi - tol,
    }
    return [side for side, cut in checks.items() if cut]


def locate_landmarks(label, slice_specs, tumor_ids=(), fov_px=None) -> list[dict]:
    """Locate every configured landmark slice in a label volume.

    Depends on the labels only, so it can reject a sample right after mask
    generation, before the (expensive) image stage. With ``fov_px`` (from
    fov_bounds_px) it also rejects slices whose body is cut by the in-plane
    field of view of the source mask. Raises LandmarkError.
    """
    locations = []
    for spec in slice_specs:
        loc = find_landmark_index(label, spec["label_id"])
        if spec.get("require_full_extent", True) and loc["touches_boundary"]:
            raise LandmarkError(
                f"{spec['label']} touches the volume boundary "
                f"(z {loc['z_min']}..{loc['z_max']}, labelled FOV z {loc['fov_z_min']}..{loc['fov_z_max']} "
                f"of {label.shape[2]})"
            )
        if fov_px is not None:
            sides = body_cut_sides(label[:, :, loc["index"]], fov_px)
            if sides:
                raise LandmarkError(f"body cut by the in-plane field of view ({', '.join(sides)}) in {spec['name']}")
        tumors = np.intersect1d(np.unique(label[:, :, loc["index"]]), tumor_ids)
        if tumors.size:
            raise LandmarkError(f"tumor labels {tumors.tolist()} present in the {spec['name']} slice")
        locations.append(loc)
    return locations


def extract_slices(image, label, slice_specs, tumor_ids=()) -> list[dict]:
    """Extract every configured landmark slice, or raise LandmarkError."""
    return [
        {
            "spec": spec,
            "hu": to_radiological(image[:, :, loc["index"]]).astype(np.int16),
            "label": to_radiological(label[:, :, loc["index"]]).astype(np.uint8),
            "location": loc,
        }
        for spec, loc in zip(slice_specs, locate_landmarks(label, slice_specs, tumor_ids))
    ]


# ----------------------------------------------------------------------------
# Anatomy-size conditions
# ----------------------------------------------------------------------------


def ensure_anatomy_size_db(cache_dir) -> Path:
    """Return the cached anatomy-size DB, fetching it from HuggingFace if needed."""
    path = Path(cache_dir) / ANATOMY_SIZE_FILE
    if not path.exists():
        from huggingface_hub import hf_hub_download

        logger.info("Fetching %s from %s", ANATOMY_SIZE_FILE, HF_REPO_ID)
        cached = hf_hub_download(HF_REPO_ID, f"datasets/{ANATOMY_SIZE_FILE}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, path)
    return path


def load_anatomy_conditions(path) -> np.ndarray:
    with open(path) as f:
        return np.array([entry["organ_size"] for entry in json.load(f)], dtype=float)


def load_mask_db(cache_dir) -> dict[str, dict]:
    """Path B mask DB keyed by mask file name; the archive is unzipped by find_masks."""
    db_dir = Path(cache_dir) / "datasets"
    for name in (MASK_DB_JSON, MASK_DB_ZIP):
        if not (db_dir / name).exists() and not (db_dir / name.removesuffix(".zip")).exists():
            raise FileNotFoundError(
                f"{db_dir / name} is missing. Download it with huggingface_hub.hf_hub_download("
                f"'{HF_REPO_ID}', 'datasets/{name}', local_dir='{cache_dir}')."
            )
    with open(db_dir / MASK_DB_JSON) as f:
        return {item["pseudo_label_filename"]: item for item in json.load(f)}


def retrieved_mask_info(candidate, mask_db, pool_index, pool_size, need_resample, cfg) -> dict:
    """Metadata for a Path B mask: the DB entry's original size/spacing and pool position."""
    name = Path(candidate["pseudo_label"]).name
    entry = next((v for k, v in mask_db.items() if Path(k).name == name), {})
    return {
        "mask_source": "retrieved",
        "retrieved_mask": {
            "file": name,
            "source_dim": entry.get("dim"),
            "source_spacing": entry.get("spacing"),
            "pool_index": pool_index,
            "pool_size": pool_size,
            "resampled": need_resample,
            "body_region": cfg.get("retrieval", {}).get("body_region", []),
        },
    }


def sample_anatomy_condition(conditions: np.ndarray, rng, exclude_tumors=True):
    """Pick a training-set anatomy-size vector with every abdominal organ present.

    With ``exclude_tumors`` the tumor slots are forced to -1 (absent), the same
    way ``LDMSampler.prepare_anatomy_size_condition`` overrides user slots.
    Returns (db_index, condition list).
    """
    candidates = np.flatnonzero(np.all(conditions[:, :N_ORGAN_SLOTS] >= 0, axis=1))
    index = int(rng.choice(candidates))
    condition = conditions[index].copy()
    if exclude_tumors:
        condition[N_ORGAN_SLOTS:] = -1.0
    return index, [float(v) for v in condition]


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------


def git_state(repo) -> dict:
    """Commit hash and whether tracked files differ from it."""

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


# ----------------------------------------------------------------------------
# MAISI wrapper
# ----------------------------------------------------------------------------


@dataclass
class GeneratedVolume:
    image: np.ndarray  # (X, Y, Z) int16 HU, RAS
    label: np.ndarray  # (X, Y, Z) uint8 MAISI labels, RAS, unfiltered
    qc_passed: bool


class MaisiGenerator:
    """Loads the MAISI rflow-ct paired pipeline once and samples volumes."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        import torch

        gen = cfg["generation"]
        repo = Path(cfg["model"]["repo"])
        self.output_size = tuple(int(v) for v in gen["output_size"])
        self.spacing = tuple(float(v) for v in gen["spacing"])
        self.device = torch.device(device)

        self.mask_source = gen.get("mask_source", "generated")
        body_region = cfg.get("retrieval", {}).get("body_region", []) if self.mask_source == "retrieved" else []
        landmark_names = [spec["label"] for spec in cfg["slices"]]
        label_dict_json = repo / "configs" / "label_dict.json"
        db_dir = Path(cfg["model"]["aux_cache"]) / "datasets"

        scripts = self._import_model_scripts(repo)
        if self.mask_source == "retrieved":
            scripts.sample_mask.check_input_ct(
                body_region, landmark_names, str(label_dict_json), self.output_size, self.spacing, []
            )
        else:
            scripts.sample_mask.check_input_ct(None, None, None, self.output_size, self.spacing, None)

        with open(repo / cfg["model"]["network_config"]) as f:
            args = Namespace(**json.load(f))
        args.autoencoder_def["num_splits"] = gen["autoencoder_tp_num_splits"]
        args.mask_generation_autoencoder_def["num_splits"] = gen["autoencoder_tp_num_splits"]
        define = scripts.utils.define_instance

        def load(name, **kwargs):
            return torch.load(repo / CHECKPOINTS[name], **kwargs)

        logger.info("Loading MAISI %s checkpoints from %s", cfg["model"]["version"], repo)
        autoencoder = define(args, "autoencoder_def").to(self.device)
        ckpt = load("autoencoder")
        autoencoder.load_state_dict(ckpt.get("unet_state_dict", ckpt))

        diffusion_unet = define(args, "diffusion_unet_def").to(self.device)
        ckpt = load("diffusion_unet", weights_only=False)
        diffusion_unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        scale_factor = ckpt["scale_factor"].to(self.device)

        controlnet = define(args, "controlnet_def").to(self.device)
        scripts.monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
        ckpt = load("controlnet", weights_only=False)
        controlnet.load_state_dict(ckpt["controlnet_state_dict"], strict=False)

        mask_autoencoder = define(args, "mask_generation_autoencoder_def").to(self.device)
        mask_autoencoder.load_state_dict(load("mask_autoencoder", weights_only=True))

        mask_unet = define(args, "mask_generation_diffusion_def").to(self.device)
        ckpt = load("mask_diffusion_unet", weights_only=False)
        mask_unet.load_state_dict(ckpt["unet_state_dict"])
        mask_scale_factor = ckpt["scale_factor"]
        del ckpt

        latent_shape = [args.latent_channels, *(s // 4 for s in self.output_size)]
        # controllable_anatomy_size=[] keeps anatomy_list = the landmarks, so
        # ensure_output_size_and_spacing raises if a landmark is cropped away and
        # find_masks only returns masks containing the landmarks. Path A
        # (generated) drives prepare_one_mask_and_meta_info directly; Path B
        # (retrieved) uses body_region and the mask DB, as in sample_multiple_images.
        retrieved = self.mask_source == "retrieved"
        self.sampler = scripts.sample.LDMSampler(
            body_region=body_region,
            anatomy_list=landmark_names,
            all_mask_files_json=str(db_dir / MASK_DB_JSON) if retrieved else None,
            all_anatomy_size_conditions_json=None,
            all_mask_files_base_dir=str(db_dir / MASK_DB_ZIP.removesuffix(".zip")) if retrieved else None,
            label_dict_json=str(label_dict_json),
            label_dict_remap_json=str(repo / "configs" / "label_dict_124_to_132.json"),
            autoencoder=autoencoder,
            diffusion_unet=diffusion_unet,
            controlnet=controlnet,
            noise_scheduler=define(args, "noise_scheduler"),
            scale_factor=scale_factor,
            mask_generation_autoencoder=mask_autoencoder,
            mask_generation_diffusion_unet=mask_unet,
            mask_generation_scale_factor=mask_scale_factor,
            mask_generation_noise_scheduler=define(args, "mask_generation_noise_scheduler"),
            device=self.device,
            latent_shape=latent_shape,
            mask_generation_latent_shape=args.mask_generation_latent_shape,
            output_size=list(self.output_size),
            output_dir=None,
            controllable_anatomy_size=[],
            real_img_median_statistics=str(repo / "configs" / "image_median_statistics_ct.json"),
            spacing=list(self.spacing),
            modality=1,
            num_inference_steps=gen["num_inference_steps"],
            mask_generation_num_inference_steps=gen["mask_generation_num_inference_steps"],
            random_seed=None,
            autoencoder_sliding_window_infer_size=gen["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=gen["autoencoder_sliding_window_infer_overlap"],
            cfg_guidance_scale=0.0,
        )
        self.mask_db = load_mask_db(cfg["model"]["aux_cache"]) if retrieved else {}
        self._set_determinism = scripts.monai.utils.set_determinism
        self._find_masks = scripts.find_masks.find_masks
        self._augmentation = scripts.sample.augmentation
        self._torch = torch

    @staticmethod
    def _import_model_scripts(repo: Path) -> Namespace:
        """Import the model repo's ``scripts`` package without writing to it."""
        sys.dont_write_bytecode = True
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import monai.networks.utils
        import monai.utils
        import scripts.find_masks
        import scripts.sample
        import scripts.sample_mask
        import scripts.utils

        if Path(scripts.__file__).resolve().parent != (repo / "scripts").resolve():
            raise ImportError(f"'scripts' resolved to {scripts.__file__}, not the model repo.")
        return Namespace(
            monai=monai,
            find_masks=scripts.find_masks,
            sample=scripts.sample,
            sample_mask=scripts.sample_mask,
            utils=scripts.utils,
        )

    def source_fov_mm(self, mask_spec: dict) -> tuple[float, float]:
        """In-plane FOV (mm) of the mask before it was resampled to the output grid."""
        if mask_spec["source"] == "generated":
            return MASK_DM_FOV_MM
        name = Path(mask_spec["candidate"]["pseudo_label"]).name
        entry = next(v for k, v in self.mask_db.items() if Path(k).name == name)
        return tuple(float(d) * float(sp) for d, sp in zip(entry["dim"][:2], entry["spacing"][:2]))

    def _load_candidate(self, candidate, need_resample):
        """Load a retrieved mask (Path B) the way sample_multiple_images does."""
        mask, top, bottom, spacing_t = self.sampler.read_mask_information(candidate)
        if need_resample:
            mask = self.sampler.ensure_output_size_and_spacing(mask)
        return mask, top, bottom, spacing_t

    def retrieval_pool(self, n_needed: int, base_seed: int, mask_check) -> tuple[list[dict], bool]:
        """Build the Path B mask pool: valid, shuffled training masks.

        Mirrors LDMSampler.sample_multiple_images: exact (size, spacing)
        matches if there are enough, otherwise find_closest_masks (closest FOV,
        resampled). The candidates are shuffled with ``base_seed`` and every
        mask is checked with ``mask_check`` up front, so each sample gets a
        distinct mask whose landmarks are fully inside the volume (when the
        pool is large enough).
        """
        s = self.sampler
        rng = np.random.default_rng(base_seed)
        exact = self._find_masks(
            s.body_region, s.anatomy_list, s.spacing, s.output_size, True, s.all_mask_files_json, s.data_root
        )
        need_resample = len(exact) < n_needed
        k, previous = n_needed, -1
        while True:
            candidates = s.find_closest_masks(k) if need_resample else exact
            order = rng.permutation(len(candidates))
            pool = []
            with self._torch.inference_mode():
                for idx in order:
                    candidate = candidates[idx]
                    try:
                        mask = self._load_candidate(candidate, need_resample)[0]
                        fov = self.source_fov_mm({"source": "retrieved", "candidate": candidate})
                        mask_check(mask.cpu().numpy().squeeze(), fov)
                    except (ValueError, SampleRejected) as exc:
                        logger.info("mask %s rejected: %s", Path(candidate["pseudo_label"]).name, exc)
                        continue
                    pool.append(candidate)
                    if not need_resample and len(pool) >= n_needed:
                        break
            if not need_resample or len(pool) >= n_needed or len(candidates) == previous:
                break
            previous, k = len(candidates), 2 * k
        if not pool:
            raise RuntimeError("No retrieved mask contains the full landmark(s).")
        if len(pool) < n_needed:
            logger.warning("Only %d valid retrieved masks for %d samples; masks will repeat.", len(pool), n_needed)
        logger.info("Retrieved-mask pool: %d masks (%s match, resample=%s)",
                    len(pool), "closest" if need_resample else "exact", need_resample)
        return pool, need_resample

    def generate(self, seed: int, mask_spec: dict, mask_check=None) -> GeneratedVolume:
        """Sample one mask + CT pair.

        ``mask_spec`` is ``{"source": "generated", "condition": [...]}`` (Path A,
        mask diffusion conditioned on an anatomy-size vector) or
        ``{"source": "retrieved", "candidate": {...}, "need_resample": bool}``
        (Path B, a training mask from the DB with MAISI's augmentation).
        ``mask_check(label_xyz, source_fov_mm)`` runs on the mask before the
        image stage and may raise SampleRejected to skip it. Raises LandmarkError if a landmark
        was cropped while resampling the mask.
        """
        torch = self._torch
        self._set_determinism(seed=seed)
        with torch.inference_mode():
            try:
                if mask_spec["source"] == "generated":
                    mask, top, bottom, spacing_t = self.sampler.prepare_one_mask_and_meta_info(
                        mask_spec["condition"]
                    )
                else:
                    mask, top, bottom, spacing_t = self._load_candidate(
                        mask_spec["candidate"], mask_spec["need_resample"]
                    )
                    mask = self._augmentation(mask, list(self.output_size), seed)
            except ValueError as exc:  # ensure_output_size_and_spacing: landmark cropped
                raise LandmarkError(str(exc)) from exc
            torch.cuda.empty_cache()
            if mask_check is not None:
                mask_check(mask.cpu().numpy().squeeze(), self.source_fov_mm(mask_spec))
            image, label = self.sampler.sample_one_pair(
                mask, top, bottom, spacing_t, self.sampler.modality_tensor
            )
            image = image.cpu().numpy().squeeze()
            label = label.cpu().numpy().squeeze()
        qc_passed = bool(self.sampler.quality_check_ct(image[None, None], label[None, None]))
        torch.cuda.empty_cache()
        image = np.clip(np.rint(image), *HU_RANGE).astype(np.int16)
        return GeneratedVolume(image=image, label=label.astype(np.uint8), qc_passed=qc_passed)


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------


def save_nifti(array, spacing, path):
    import nibabel as nib

    nib.save(nib.Nifti1Image(array, affine=np.diag([*spacing, 1.0])), str(path))


def save_preview(volume: GeneratedVolume, slices, spacing, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(slices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)
    sx, sy, sz = spacing
    for row, item in zip(axes, slices):
        loc = item["location"]
        name = item["spec"]["name"]
        row[0].imshow(item["hu"], cmap="gray", vmin=-200, vmax=300)
        row[0].set_title(f"{name}: z={loc['index']} (HU, W500/L50)")
        row[1].imshow(item["hu"], cmap="gray", vmin=-200, vmax=300)
        row[1].imshow(np.ma.masked_equal(item["label"], 0), cmap="tab20", alpha=0.45, interpolation="nearest")
        row[1].set_title(f"{name} with MAISI labels")
        cols = np.flatnonzero(np.any(volume.label == item["spec"]["label_id"], axis=(1, 2)))
        x_mid = int(np.median(cols)) if cols.size else volume.image.shape[0] // 2
        row[2].imshow(volume.image[x_mid].T, cmap="gray", vmin=-200, vmax=300, origin="lower", aspect=sz / sy)
        row[2].axhline(loc["index"], color="tab:red", lw=1)
        row[2].axhspan(loc["z_min"], loc["z_max"], color="tab:red", alpha=0.15)
        row[2].set_title(f"sagittal x={x_mid} (anterior right, superior up)")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def save_sample(sample_dir: Path, volume, slices, meta, cfg):
    sample_dir.mkdir(parents=True, exist_ok=True)
    spacing = tuple(cfg["generation"]["spacing"])
    for item in slices:
        name = item["spec"]["name"]
        np.save(sample_dir / f"{name}_hu.npy", item["hu"])
        np.save(sample_dir / f"{name}_label.npy", item["label"])
    if cfg["output"]["save_volumes"]:
        save_nifti(volume.image, spacing, sample_dir / "volume_image.nii.gz")
        save_nifti(volume.label, spacing, sample_dir / "volume_label.nii.gz")
    if cfg["output"]["save_preview"]:
        save_preview(volume, slices, spacing, sample_dir / "preview.png")
    # meta.json last: its presence marks the sample as complete (used for resume).
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def write_manifest(run_dir: Path, records: list[dict]):
    fields = ["sample_index", "status", "seed", "attempt", "qc_passed", "slices", "slice_indices", "dir", "reason"]
    with open(run_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda r: r["sample_index"]):
            writer.writerow(record)


def _record_from_meta(meta: dict, sample_dir: Path) -> dict:
    return {
        "sample_index": meta["sample_index"],
        "status": "ok",
        "seed": meta["seed"],
        "attempt": meta["attempt"],
        "qc_passed": meta["qc_passed"],
        "slices": ";".join(s["name"] for s in meta["slices"]),
        "slice_indices": ";".join(str(s["slice_index"]) for s in meta["slices"]),
        "dir": sample_dir.name,
        "reason": "",
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def build_meta(cfg, sample_index, seed, attempt, mask_info, volume, slices, provenance, seconds):
    gen = cfg["generation"]
    sx, sy, sz = gen["spacing"]
    return {
        "sample_index": sample_index,
        "seed": seed,
        "base_seed": gen["base_seed"],
        "attempt": attempt,
        "model_version": cfg["model"]["version"],
        "model_repo": cfg["model"]["repo"],
        "model_commit": provenance["model"]["commit"],
        "model_repo_dirty": provenance["model"]["dirty"],
        "checkpoints": provenance["checkpoints"],
        "this_repo_commit": provenance["this_repo"]["commit"],
        "this_repo_dirty": provenance["this_repo"]["dirty"],
        "slices": [
            {
                "name": item["spec"]["name"],
                "landmark": item["spec"]["label"],
                "label_id": item["spec"]["label_id"],
                "method": item["spec"].get("method", "centroid"),
                "slice_index": item["location"]["index"],
                "slice_z_mm": item["location"]["index"] * sz,
                "label_z_range": [item["location"]["z_min"], item["location"]["z_max"]],
                "labelled_fov_z_range": [item["location"]["fov_z_min"], item["location"]["fov_z_max"]],
                "label_z_extent_mm": (item["location"]["z_max"] - item["location"]["z_min"] + 1) * sz,
                "hu_file": f"{item['spec']['name']}_hu.npy",
                "label_file": f"{item['spec']['name']}_label.npy",
                "shape": list(item["hu"].shape),
                "spacing_mm": [sy, sx],
            }
        for item in slices],
        "volume_shape": list(volume.image.shape),
        "volume_spacing_xyz_mm": [sx, sy, sz],
        "orientation": ORIENTATION_NOTE,
        "hu_clip": list(HU_RANGE),
        "qc_passed": volume.qc_passed,
        **mask_info,
        "num_inference_steps": gen["num_inference_steps"],
        "mask_generation_num_inference_steps": gen["mask_generation_num_inference_steps"],
        "versions": provenance["versions"],
        "generation_seconds": round(seconds, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def collect_provenance(cfg) -> dict:
    import monai
    import torch

    repo = Path(cfg["model"]["repo"])
    return {
        "model": git_state(repo),
        "this_repo": git_state(REPO_ROOT),
        "checkpoints": {
            Path(rel).name: (repo / rel).stat().st_size for rel in CHECKPOINTS.values()
        },
        "versions": {"torch": torch.__version__, "monai": monai.__version__, "numpy": np.__version__},
    }


def run(cfg: dict, generator: MaisiGenerator | None = None) -> list[dict]:
    """Generate ``num_volumes`` samples into ``output.dir/output.run_name``."""
    gen = cfg["generation"]
    run_dir = Path(cfg["output"]["dir"]) / cfg["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    label_dict = load_label_dict(cfg["model"]["repo"])
    slice_specs = resolve_label_ids(cfg["slices"], label_dict)
    tumor_ids = [label_dict[name] for name in TUMOR_LABEL_NAMES] if gen["exclude_tumors"] else []
    mask_source = gen.get("mask_source", "generated")
    if mask_source == "generated":
        conditions = load_anatomy_conditions(ensure_anatomy_size_db(cfg["model"]["aux_cache"]))
    else:
        mask_db = load_mask_db(cfg["model"]["aux_cache"])
        pool, need_resample = None, None

    def mask_check(label_xyz, source_fov_mm):
        fov_px = fov_bounds_px(source_fov_mm, gen["output_size"][:2], gen["spacing"][:2])
        return locate_landmarks(label_xyz, slice_specs, tumor_ids, fov_px)
    provenance = collect_provenance(cfg)
    if provenance["model"]["dirty"]:
        logger.warning("Model repo has local modifications to tracked files.")

    records = []
    for i in range(gen["num_volumes"]):
        sample_dir = run_dir / f"sample_{i:04d}"
        if (sample_dir / "meta.json").exists():
            with open(sample_dir / "meta.json") as f:
                records.append(_record_from_meta(json.load(f), sample_dir))
            logger.info("sample %d: already done, skipping", i)
            continue
        if generator is None:
            generator = MaisiGenerator(cfg)
        if mask_source == "retrieved" and pool is None:
            pool, need_resample = generator.retrieval_pool(gen["num_volumes"], gen["base_seed"], mask_check)

        reasons = []
        for attempt in range(gen["max_attempts"]):
            seed = gen["base_seed"] + i + attempt * SEED_ATTEMPT_STRIDE
            if mask_source == "generated":
                db_index, condition = sample_anatomy_condition(
                    conditions, np.random.default_rng(seed), gen["exclude_tumors"]
                )
                mask_spec = {"source": "generated", "condition": condition}
                mask_info = {"mask_source": "generated", "anatomy_size_db_index": db_index,
                             "anatomy_size_condition": condition}
                logger.info("sample %d attempt %d: seed %d, anatomy DB entry %d", i, attempt, seed, db_index)
            else:
                # Each sample keeps its pool mask; retries re-draw augmentation + image noise.
                candidate = pool[i % len(pool)]
                mask_spec = {"source": "retrieved", "candidate": candidate, "need_resample": need_resample}
                mask_info = retrieved_mask_info(candidate, mask_db, i % len(pool), len(pool), need_resample, cfg)
                logger.info("sample %d attempt %d: seed %d, mask %s", i, attempt, seed, mask_info["retrieved_mask"]["file"])
            start = time.time()
            try:
                volume = generator.generate(seed, mask_spec, mask_check=mask_check)
                if not volume.qc_passed:
                    raise SampleRejected("MAISI organ-HU quality check failed")
                slices = extract_slices(volume.image, volume.label, slice_specs, tumor_ids)
            except SampleRejected as exc:
                reasons.append(f"seed {seed}: {exc}")
                logger.warning("sample %d attempt %d rejected: %s", i, attempt, exc)
                continue
            meta = build_meta(cfg, i, seed, attempt, mask_info, volume, slices, provenance, time.time() - start)
            meta["rejected_attempts"] = reasons
            save_sample(sample_dir, volume, slices, meta, cfg)
            records.append(_record_from_meta(meta, sample_dir))
            logger.info(
                "sample %d: saved %s in %.0f s",
                i,
                ", ".join(f"{s['spec']['name']}@z{s['location']['index']}" for s in slices),
                meta["generation_seconds"],
            )
            break
        else:
            records.append({"sample_index": i, "status": "failed", "dir": sample_dir.name, "reason": " | ".join(reasons)})
            logger.error("sample %d: no valid volume after %d attempts", i, gen["max_attempts"])
        write_manifest(run_dir, records)

    write_manifest(run_dir, records)
    n_ok = sum(r["status"] == "ok" for r in records)
    logger.info("Done: %d/%d samples in %s", n_ok, len(records), run_dir)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    parser.add_argument("--num-volumes", type=int, help="Override generation.num_volumes")
    parser.add_argument("--base-seed", type=int, help="Override generation.base_seed")
    parser.add_argument("--run-name", help="Override output.run_name")
    parser.add_argument("--mask-source", choices=MASK_SOURCES, help="Override generation.mask_source")
    parser.add_argument("--max-attempts", type=int, help="Override generation.max_attempts")
    parser.add_argument("--output-dir", help="Override output.dir")
    parser.add_argument("--no-volumes", action="store_true", help="Do not save full NIfTI volumes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    cfg = load_config(
        args.config,
        {
            "generation.num_volumes": args.num_volumes,
            "generation.base_seed": args.base_seed,
            "generation.mask_source": args.mask_source,
            "generation.max_attempts": args.max_attempts,
            "output.run_name": args.run_name,
            "output.dir": args.output_dir,
            "output.save_volumes": False if args.no_volumes else None,
        },
    )
    records = run(copy.deepcopy(cfg))
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
