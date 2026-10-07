"""Tests for src/generate_slices.py (MAISI landmark-slice generation).

The unit tests are CPU-only. The smoke tests run the real MAISI pipeline for one
volume per mask source (Path A generated, Path B retrieved; about 20 GB GPU, a
few minutes) and are marked ``slow``:

    python -m pytest desmond/tests/test_generate_slices.py -m "not slow"
    python -m pytest desmond/tests/test_generate_slices.py -m slow
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.generate_slices import (
    CHECKPOINTS,
    MASK_DM_FOV_MM,
    body_cut_sides,
    fov_bounds_px,
    MASK_DB_JSON,
    DEFAULT_CONFIG,
    HU_RANGE,
    LandmarkError,
    extract_slices,
    find_landmark_index,
    load_config,
    load_label_dict,
    locate_landmarks,
    resolve_label_ids,
    run,
    sample_anatomy_condition,
    to_radiological,
)

L3 = 35


def make_label(shape=(8, 6, 10), z_range=(3, 6), label_id=L3, fov=None):
    """Body (200) over the labelled FOV (default: all slices) with a landmark block."""
    label = np.zeros(shape, dtype=np.uint8)
    lo, hi = fov if fov is not None else (0, shape[2] - 1)
    label[1:7, 0:5, lo : hi + 1] = 200
    label[2:5, 1:4, z_range[0] : z_range[1] + 1] = label_id
    return label


# ---------------------------------------------------------------- landmarks


def test_landmark_centroid_and_extent():
    loc = find_landmark_index(make_label(z_range=(3, 6)), L3)
    assert loc["z_min"] == 3 and loc["z_max"] == 6
    assert loc["z_centroid"] == pytest.approx(4.5)
    assert loc["index"] in (4, 5)
    assert not loc["touches_boundary"]


def test_landmark_centroid_is_weighted_by_voxels():
    label = make_label(z_range=(2, 3))
    label[:, :, 3] = L3  # slice 3 now dominates
    label[0, 0, :] = 200  # keep the labelled FOV spanning every slice
    loc = find_landmark_index(label, L3)
    assert loc["index"] == 3


def test_landmark_missing_raises():
    with pytest.raises(LandmarkError):
        find_landmark_index(np.zeros((4, 4, 4), dtype=np.uint8), L3)


@pytest.mark.parametrize("z_range", [(0, 3), (5, 9)])
def test_landmark_touching_boundary_is_flagged(z_range):
    assert find_landmark_index(make_label(z_range=z_range), L3)["touches_boundary"]


def test_landmark_truncated_by_padded_fov_is_flagged():
    # A scan whose FOV ends inside L3, padded with empty slices after resampling:
    # L3 does not touch the array edge but does touch the labelled FOV.
    label = make_label(z_range=(2, 3), fov=(2, 9))
    loc = find_landmark_index(label, L3)
    assert loc["z_min"] > 0 and loc["fov_z_min"] == 2
    assert loc["touches_boundary"]
    assert not find_landmark_index(make_label(z_range=(4, 6), fov=(2, 9)), L3)["touches_boundary"]


def test_extract_slices_rejects_truncated_landmark_and_tumors():
    spec = {"name": "L3_mid", "label": "vertebrae L3", "label_id": L3, "require_full_extent": True}
    image = np.zeros((8, 6, 10), dtype=np.int16)

    with pytest.raises(LandmarkError, match="boundary"):
        extract_slices(image, make_label(z_range=(0, 4)), [spec])

    label = make_label(z_range=(3, 6))
    label[0, 0, :] = 26  # hepatic tumor on every slice
    with pytest.raises(LandmarkError, match="tumor"):
        extract_slices(image, label, [spec], tumor_ids=[26])

    (item,) = extract_slices(image, make_label(z_range=(3, 6)), [spec])
    assert item["hu"].shape == (6, 8) and item["hu"].dtype == np.int16
    assert item["label"].dtype == np.uint8 and L3 in item["label"]


def test_fov_bounds_for_padded_and_cropped_sources():
    # Mask diffusion output (384 mm) on a 512 x 0.781 mm grid: centred band, ~10 px padding.
    (x_lo, x_hi), (y_lo, y_hi) = fov_bounds_px(MASK_DM_FOV_MM, (512, 512), (0.781, 0.781))
    assert (x_lo, x_hi) == (10, 501) and (y_lo, y_hi) == (10, 501)
    # A source at least as large as the output fills it (centre crop).
    assert fov_bounds_px((420.0, 400.0), (512, 512), (0.781, 0.781)) == [(0, 511), (0, 511)]


def test_body_cut_sides():
    fov = [(10, 501), (10, 501)]
    s = np.zeros((512, 512), dtype=np.uint8)
    s[100:400, 150:350] = 200
    assert body_cut_sides(s, fov) == []
    s[10:502, 200:300] = 200  # body spans the full source FOV left-right
    assert body_cut_sides(s, fov) == ["left", "right"]


def test_locate_landmarks_rejects_body_cut_in_plane():
    spec = {"name": "L3_mid", "label": "vertebrae L3", "label_id": L3, "require_full_extent": True}
    label = np.zeros((12, 12, 10), dtype=np.uint8)
    label[3:9, 3:9, :] = 200  # body x, y 3..8 on every slice
    label[5:7, 5:7, 3:7] = L3
    with pytest.raises(LandmarkError, match=r"in-plane field of view \(left, right\)"):
        locate_landmarks(label, [spec], fov_px=[(3, 8), (0, 11)])  # source FOV ends at the body sides
    locate_landmarks(label, [spec], fov_px=[(0, 11), (0, 11)])  # body well inside the FOV


# -------------------------------------------------------------- orientation


def test_to_radiological_orientation():
    # RAS slice: x -> patient right, y -> anterior.
    nx, ny = 5, 3
    slice_xy = np.zeros((nx, ny))
    slice_xy[nx - 1, ny - 1] = 1  # most right, most anterior
    slice_xy[0, 0] = 2  # most left, most posterior
    out = to_radiological(slice_xy)
    assert out.shape == (ny, nx)  # (rows, cols)
    assert out[0, 0] == 1  # anterior-right -> top-left (radiological display)
    assert out[-1, -1] == 2  # posterior-left -> bottom-right


# ------------------------------------------------------------ config & DB


def test_default_config_resolves():
    cfg = load_config(DEFAULT_CONFIG, {"generation.num_volumes": 2, "output.run_name": "x"})
    assert cfg["generation"]["num_volumes"] == 2
    assert cfg["output"]["run_name"] == "x"
    assert Path(cfg["output"]["dir"]).is_absolute()
    assert Path(cfg["model"]["aux_cache"]).is_absolute()


def test_label_names_resolve_to_maisi_ids():
    cfg = load_config(DEFAULT_CONFIG)
    if not Path(cfg["model"]["repo"]).exists():
        pytest.skip("model repo not available")
    specs = resolve_label_ids(cfg["slices"], load_label_dict(cfg["model"]["repo"]))
    assert specs[0]["label"] == "vertebrae L3" and specs[0]["label_id"] == L3


def test_anatomy_condition_has_organs_and_no_tumors():
    conditions = np.array(
        [
            [-1, 0.2, 0.3, 0.4, 0.5, 0.9, -1, -1, -1, -1],  # gallbladder absent -> never picked
            [0.1, 0.2, 0.3, 0.4, 0.5, -1, 0.7, -1, -1, -1],
            [0.1, 0.2, 0.3, 0.4, 0.5, -1, -1, -1, -1, 0.2],
        ]
    )
    for seed in range(10):
        index, cond = sample_anatomy_condition(conditions, np.random.default_rng(seed), True)
        assert index in (1, 2)
        assert all(v >= 0 for v in cond[:5]) and all(v == -1 for v in cond[5:])
    _, a = sample_anatomy_condition(conditions, np.random.default_rng(3), True)
    _, b = sample_anatomy_condition(conditions, np.random.default_rng(3), True)
    assert a == b


# --------------------------------------------------------------- GPU smoke


def _smoke_prerequisites(cfg, mask_source):
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    repo = Path(cfg["model"]["repo"])
    missing = [rel for rel in CHECKPOINTS.values() if not (repo / rel).exists()]
    if missing:
        pytest.skip(f"MAISI checkpoints missing: {missing}")
    if mask_source == "retrieved" and not (Path(cfg["model"]["aux_cache"]) / "datasets" / MASK_DB_JSON).exists():
        pytest.skip("Path B mask DB not downloaded (see docs/model_notes.md)")


@pytest.mark.slow
@pytest.mark.parametrize("mask_source", ["generated", "retrieved"])
def test_smoke_generate_one_volume(tmp_path, mask_source):
    cfg = load_config(
        DEFAULT_CONFIG,
        {
            "generation.mask_source": mask_source,
            "generation.num_volumes": 1,
            "output.dir": str(tmp_path),
            "output.run_name": "smoke",
            "output.save_volumes": False,
        },
    )
    _smoke_prerequisites(cfg, mask_source)

    records = run(cfg)

    assert len(records) == 1 and records[0]["status"] == "ok", records
    sample_dir = tmp_path / "smoke" / "sample_0000"
    meta = json.loads((sample_dir / "meta.json").read_text())
    nx, ny, _ = cfg["generation"]["output_size"]
    (slice_meta,) = meta["slices"]

    hu = np.load(sample_dir / slice_meta["hu_file"])
    label = np.load(sample_dir / slice_meta["label_file"])
    assert hu.shape == (ny, nx) == tuple(slice_meta["shape"])
    assert hu.dtype == np.int16
    assert HU_RANGE[0] <= hu.min() and hu.max() <= HU_RANGE[1]
    assert np.mean(hu < -900) > 0.05  # background air around the body
    assert np.mean((hu > -100) & (hu < 100)) > 0.05  # soft tissue
    assert L3 in label

    for key in ("seed", "model_commit", "slices", "volume_spacing_xyz_mm", "qc_passed", "rejected_attempts"):
        assert key in meta
    assert meta["mask_source"] == mask_source
    assert meta["seed"] == cfg["generation"]["base_seed"] + meta["attempt"] * 100_000
    assert len(meta["model_commit"]) == 40
    assert slice_meta["label_id"] == L3
    assert 0 < slice_meta["slice_index"] < cfg["generation"]["output_size"][2] - 1
    sx, sy, _ = cfg["generation"]["spacing"]
    assert slice_meta["spacing_mm"] == [sy, sx]
    assert (tmp_path / "smoke" / "manifest.csv").exists()
