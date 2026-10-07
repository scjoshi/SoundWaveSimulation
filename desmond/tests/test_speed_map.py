"""Tests for src/speed_map.py (Phase 1 IT'IS-anchored sound-speed maps). CPU only."""

import numpy as np
import pytest

from src.speed_map import build_maps, load_config, output_name, save_npz

LIVER, KIDNEY, L3 = 1, 5, 35


def phantom():
    """A 128x128 abdomen at 1 mm: fat rim, lean interior, liver with a gas pocket,
    a contrast-enhanced kidney, and a vertebra (MAISI label 35)."""
    n = 128
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    hu = np.full((n, n), -1000.0)
    hu[r < 55] = -100.0  # subcutaneous fat rim
    hu[r < 45] = 40.0  # lean interior
    label = np.zeros((n, n), dtype=np.uint8)
    label[r < 55] = 200  # MAISI body envelope
    label[40:60, 40:60] = LIVER
    hu[40:60, 40:60] = 60.0
    hu[48:53, 48:53] = -900.0  # gas pocket inside the liver label
    label[70:85, 40:55] = KIDNEY
    hu[70:85, 40:55] = 150.0  # contrast-enhanced kidney (non-contrast is about 30 HU)
    hu[76:79, 46:49] = 320.0  # a very bright contrast spot inside the kidney
    label[60:75, 70:85] = L3
    hu[60:75, 70:85] = 600.0
    return hu, label, (1.0, 1.0)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tissue_at(maps, r, c):
    return str(maps["tissue_names"][maps["tissue_index"][r, c]])


def test_config_resolves_all_tissues(cfg):
    assert L3 in cfg["bone_ids"] and 127 in cfg["bone_ids"] and LIVER not in cfg["bone_ids"]
    assert cfg["itis"]["Fat"]["speed"] == pytest.approx(1440.19, abs=0.01)
    assert cfg["itis"]["Bone (Cortical)"]["speed"] == pytest.approx(3514.86, abs=0.01)


def test_priority_rules(cfg):
    hu, label, sp = phantom()
    maps = build_maps(hu, label, sp, cfg)
    assert tissue_at(maps, 64, 10) == "Fat"  # fat rim
    assert tissue_at(maps, 45, 45) == "Liver"
    assert tissue_at(maps, 50, 50) == "Gas as Water"  # gas beats the organ label
    assert tissue_at(maps, 77, 47) == "Kidney"  # contrast spot is not bone
    assert tissue_at(maps, 67, 77) == "Bone (graded)"
    assert tissue_at(maps, 2, 2) == "Coupling water"
    assert not maps["body_mask"][2, 2] and maps["body_mask"][64, 64]


def test_contrast_offset_removed_and_texture_bounded(cfg):
    hu, label, sp = phantom()
    maps = build_maps(hu, label, sp, cfg)
    kid = maps["speed_m_s"][label == KIDNEY]
    mean, sd = cfg["itis"]["Kidney"]["speed"], cfg["itis"]["Kidney"]["speed_sd"]
    assert abs(np.median(kid) - mean) < 1.0  # the 150 HU contrast offset is absorbed
    assert kid.min() >= mean - sd - 1e-6 and kid.max() <= mean + sd + 1e-6


def test_bone_is_graded_between_anchors(cfg):
    hu, label, sp = phantom()
    maps = build_maps(hu, label, sp, cfg)
    c = maps["speed_m_s"][67, 77]
    canc, cort = cfg["itis"]["Bone (Cancellous)"]["speed"], cfg["itis"]["Bone (Cortical)"]["speed"]
    expected = canc + (600 - 200) / (1000 - 200) * (cort - canc)
    assert c == pytest.approx(expected, rel=0.02)
    assert maps["density_kg_m3"][67, 77] > cfg["itis"]["Bone (Cancellous)"]["density"]


@pytest.mark.parametrize("gas, tissue", [("water", "Water"), ("lumen", "Small Intestine Lumen"), ("air", "Air")])
def test_gas_options(gas, tissue):
    cfg = load_config(gas=gas)
    hu, label, sp = phantom()
    maps = build_maps(hu, label, sp, cfg)
    assert maps["speed_m_s"][50, 50] == pytest.approx(cfg["itis"][tissue]["speed"])
    assert output_name(cfg) == ("acoustic_itis.npz" if gas == "water" else f"acoustic_itis_gas-{gas}.npz")


def test_texture_off_gives_constant_tissues():
    cfg = load_config(texture=False)
    hu, label, sp = phantom()
    hu = hu + np.random.default_rng(0).normal(0, 20, hu.shape)  # CT-like noise
    maps = build_maps(hu, label, sp, cfg)
    liver = maps["speed_m_s"][(label == LIVER) & (maps["tissue_index"] == maps["tissue_index"][45, 45])]
    assert np.ptp(liver) == 0 and liver[0] == pytest.approx(cfg["itis"]["Liver"]["speed"])


def test_output_loads_in_advisor_simulator(cfg, tmp_path):
    from fdtd2d import load_ct_ring_grid

    hu, label, sp = phantom()
    maps = build_maps(hu, label, sp, cfg)
    path = tmp_path / "acoustic_itis.npz"
    save_npz(path, maps, sp, cfg)
    grid = load_ct_ring_grid(path, grid_shape=(64, 64))
    assert np.isfinite(grid.c).all() and grid.c.max() > 2000  # bone survives resampling
