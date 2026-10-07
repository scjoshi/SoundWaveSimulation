"""Phase 1 sound-speed maps: CT slice + MAISI labels -> IT'IS-anchored speed and density.

The advisor's ``ct_to_speed.py --mapping hybrid`` maps every pixel through one
HU -> density -> speed regression. On contrast-enhanced synthetic CT that gives
predictable biases (desmond/docs/critical_review.md, section 4):

* fat comes out too slow (about 1418 m/s at -100 HU; IT'IS fat is 1440 +- 22);
* contrast-enhanced organs come out too fast (kidney about 1620-1680 m/s;
  IT'IS kidney is 1554 +- 18);
* spongy vertebral bone above 200 HU gets cortical speed (3476 m/s);
* CT noise (8-27 HU) becomes 8-31 m/s of random per-pixel speed texture.

This module keeps the advisor's body mask and tissue classes
(``ct_to_speed.body_mask_from_hu`` and ``segment_tissues``) and adds the MAISI
organ labels. It then:

1. denoises HU with a Gaussian that never crosses tissue boundaries;
2. anchors every tissue to its IT'IS mean speed and density;
3. adds within-tissue texture from the denoised HU relative to that tissue's
   median HU in the slice. The median absorbs contrast offsets, and the
   texture is clipped to the tissue's IT'IS standard deviation;
4. grades bone between red marrow, cancellous and cortical values by HU;
5. handles gas as water (default), bowel contents, or air.

The output .npz is a drop-in replacement for ct_to_speed.py's: it has the
``speed_m_s``, ``body_mask`` and ``spacing_mm`` keys the advisor's
simulators read (``--ct-speed``), plus density and tissue maps. The rules and
values live in desmond/configs/tissue_map.yaml and itis_tissue_properties.csv.

    python desmond/src/speed_map.py --run desmond/data/generated/pathB_retrieved_n10
    python desmond/src/speed_map.py --run ... --gas lumen
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]  # desmond/
REPO_ROOT = ROOT.parent  # advisor's code: ct_to_speed.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ct_to_speed  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "tissue_map.yaml"
MAPPING_NAME = "desmond-itis-v1"
CLASS_CODES = {
    "AIR": ct_to_speed.AIR, "BONE": ct_to_speed.BONE, "MUSCLE": ct_to_speed.MUSCLE,
    "VISCERAL": ct_to_speed.VISCERAL_ORGANS, "SAT": ct_to_speed.SAT, "VAT": ct_to_speed.VAT,
}
WATER, GAS, BONE = "__water__", "__gas__", "__bone__"


def load_itis(path) -> dict[str, dict]:
    with open(path) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        table = {}
        for r in rows:
            sd = float(r["speed_sd"]) if r["speed_sd"] not in ("", None) else -1.0
            table[r["tissue"]] = {
                "speed": float(r["speed_m_s"]), "speed_sd": sd, "speed_n": int(float(r["speed_n"] or 0)),
                "density": float(r["density_kg_m3"]),
            }
    return table


def load_config(path=DEFAULT_CONFIG, gas=None, texture=None) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if gas is not None:
        cfg["gas"] = gas
    if texture is not None:
        cfg["texture"]["enabled"] = bool(texture)
    if cfg["gas"] not in cfg["gas_tissues"]:
        raise ValueError(f"gas must be one of {sorted(cfg['gas_tissues'])}")
    cfg["itis"] = load_itis(path.parent / cfg["itis_table"])
    needed = {*cfg["labels"].values(), *cfg["classes"].values(), *cfg["bone"]["tissues"],
              *cfg["gas_tissues"].values()}
    missing = sorted(t for t in needed if t not in cfg["itis"])
    if missing:
        raise KeyError(f"tissues not in the IT'IS table: {missing}")
    lo = [int(a) for a, _ in cfg["bone"]["label_ranges"]]
    hi = [int(b) for _, b in cfg["bone"]["label_ranges"]]
    cfg["bone_ids"] = sorted({i for a, b in zip(lo, hi) for i in range(a, b + 1)} | set(cfg["bone"]["label_ids"]))
    return cfg


def region_smooth(values, regions, sigma_px):
    """Gaussian smoothing within each region only (normalized convolution)."""
    out = np.array(values, dtype=float, copy=True)
    if np.all(np.asarray(sigma_px) <= 0):
        return out
    for r in np.unique(regions):
        m = regions == r
        num = ndimage.gaussian_filter(np.where(m, values, 0.0), sigma_px)
        den = ndimage.gaussian_filter(m.astype(float), sigma_px)
        out[m] = num[m] / np.maximum(den[m], 1e-6)
    return out


def assign_tissues(hu, label, spacing_mm, cfg):
    """Tissue name per pixel, following the priority order in tissue_map.yaml.

    Returns (names array of dtype object, body mask). Classification uses HU
    smoothed at the denoising scale so CT noise does not speckle class edges.
    """
    sigma_px = cfg["denoise_sigma_mm"] / np.asarray(spacing_mm, dtype=float)
    body = ct_to_speed.body_mask_from_hu(hu.astype(float))  # same body as the advisor's maps
    # Smooth inside and outside the body separately: otherwise the -1000 HU air
    # outside bleeds into the skin layer and edge pixels are classed as gas.
    hu_s = region_smooth(hu.astype(float), body.astype(np.int8), sigma_px)
    classes, _, _ = ct_to_speed.segment_tissues(hu_s, spacing_mm=tuple(spacing_mm))

    names = np.full(hu.shape, WATER, dtype=object)
    for cls, tissue in cfg["classes"].items():  # 6. class defaults
        names[body & (classes == CLASS_CODES[cls])] = tissue
    for lid, tissue in cfg["labels"].items():  # 5. organ labels
        names[body & (label == int(lid))] = tissue
    fat = np.isin(classes, [ct_to_speed.SAT, ct_to_speed.VAT])
    names[body & fat] = np.where(classes == ct_to_speed.SAT, cfg["classes"]["SAT"], cfg["classes"]["VAT"])[body & fat]
    names[body & (classes == ct_to_speed.AIR)] = GAS  # 3. gas
    names[body & np.isin(label, cfg["bone_ids"])] = BONE  # 2. bone labels
    return names, body


def build_maps(hu, label, spacing_mm, cfg) -> dict:
    """Speed, density and tissue maps for one slice (radiological orientation)."""
    hu = np.asarray(hu, dtype=float)
    names, body = assign_tissues(hu, label, spacing_mm, cfg)
    vocab = sorted(set(names.ravel()))
    index = np.zeros(hu.shape, dtype=np.int16)
    for i, v in enumerate(vocab):
        index[names == v] = i
    sigma_px = cfg["denoise_sigma_mm"] / np.asarray(spacing_mm, dtype=float)
    hu_dn = region_smooth(hu, index, sigma_px)

    itis = cfg["itis"]
    water_c, water_rho = cfg["coupling_water_m_s"], cfg["coupling_water_density_kg_m3"]
    gas_tissue = cfg["gas_tissues"][cfg["gas"]]
    speed = np.full(hu.shape, water_c)
    density = np.full(hu.shape, water_rho)
    tex = cfg["texture"]

    for name in vocab:
        m = names == name
        if name == WATER:
            continue
        if name == BONE:
            anchors = np.asarray(cfg["bone"]["hu_anchors"], dtype=float)
            speed[m] = np.interp(hu_dn[m], anchors, [itis[t]["speed"] for t in cfg["bone"]["tissues"]])
            density[m] = np.interp(hu_dn[m], anchors, [itis[t]["density"] for t in cfg["bone"]["tissues"]])
            continue
        tissue = gas_tissue if name == GAS else name
        speed[m], density[m] = itis[tissue]["speed"], itis[tissue]["density"]
        if tex["enabled"] and name != GAS:
            sd = itis[tissue]["speed_sd"] if itis[tissue]["speed_sd"] > 0 else tex["default_sd_m_s"]
            lim = tex["clip_sd"] * sd
            speed[m] += np.clip(tex["slope_m_s_per_hu"] * (hu_dn[m] - np.median(hu_dn[m])), -lim, lim)

    display = {WATER: "Coupling water", GAS: f"Gas as {gas_tissue}", BONE: "Bone (graded)"}
    return {
        "speed_m_s": speed, "density_kg_m3": density, "body_mask": body,
        "tissue_index": index, "tissue_names": np.array([display.get(v, v) for v in vocab]),
        "hu_denoised": hu_dn.astype(np.float32),
    }


def save_npz(path, maps, spacing_mm, cfg):
    settings = {k: cfg[k] for k in ("gas", "denoise_sigma_mm", "texture", "coupling_water_m_s")}
    np.savez_compressed(
        path, speed_m_s=maps["speed_m_s"], density_kg_m3=maps["density_kg_m3"],
        body_mask=maps["body_mask"], spacing_mm=np.asarray(spacing_mm, dtype=float),
        tissue_index=maps["tissue_index"], tissue_names=maps["tissue_names"],
        hu_denoised=maps["hu_denoised"], mapping=MAPPING_NAME, settings=json.dumps(settings),
    )


def output_name(cfg) -> str:
    parts = ["acoustic_itis"]
    if cfg["gas"] != "water":
        parts.append(f"gas-{cfg['gas']}")
    if not cfg["texture"]["enabled"]:
        parts.append("notexture")
    return "_".join(parts) + ".npz"


def process_sample(sample: Path, cfg) -> Path:
    meta = json.loads((sample / "meta.json").read_text())
    s = meta["slices"][0]
    hu = np.load(sample / s["hu_file"])
    label = np.load(sample / s["label_file"])
    maps = build_maps(hu, label, s["spacing_mm"], cfg)
    out = sample / output_name(cfg)
    save_npz(out, maps, s["spacing_mm"], cfg)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="generate_slices.py run directory (all samples)")
    target.add_argument("--sample", type=Path, help="one sample directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gas", choices=("water", "lumen", "air"))
    parser.add_argument("--no-texture", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, gas=args.gas, texture=False if args.no_texture else None)
    samples = [args.sample] if args.sample else sorted(p for p in args.run.glob("sample_*") if (p / "meta.json").exists())
    for s in samples:
        print(f"wrote {process_sample(s, cfg)}")


if __name__ == "__main__":
    main()
