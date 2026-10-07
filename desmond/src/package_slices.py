"""Package generated L3 slices into a small, git-tracked dataset for sharing.

Copies only what a user of the slices needs (HU slice, MAISI labels, both
sound-speed maps, metadata, preview) and writes index.csv. The 3D volumes and
simulation traces stay in the gitignored run directory; they can be
regenerated from the seeds in meta.json.

    python desmond/src/package_slices.py --run desmond/data/generated/pathB_retrieved_n10 \\
        --out desmond/data/ct_slices_l3_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

FILES = ("L3_mid_hu.npy", "L3_mid_label.npy", "acoustic.npz", "acoustic_itis.npz", "meta.json", "preview.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in sorted(args.run.glob("sample_*")):
        if not (s / "meta.json").exists():
            continue
        dst = args.out / s.name
        dst.mkdir(exist_ok=True)
        for name in FILES:
            if (s / name).exists():
                shutil.copy2(s / name, dst / name)
        meta = json.loads((s / "meta.json").read_text())
        sl = meta["slices"][0]
        z = np.load(s / "acoustic_itis.npz")
        names = [str(n) for n in z["tissue_names"]]
        body = z["body_mask"]
        share = lambda t: float(np.mean(z["tissue_index"][body] == names.index(t))) if t in names else 0.0  # noqa: E731
        mask = meta.get("retrieved_mask", {})
        rows.append({
            "sample": s.name, "seed": meta["seed"], "mask_file": mask.get("file", ""),
            "slice_index": sl["slice_index"], "pixel_spacing_mm": sl["spacing_mm"][0],
            "shape": "x".join(map(str, sl["shape"])),
            "body_area_cm2": round(float(body.sum()) * sl["spacing_mm"][0] * sl["spacing_mm"][1] / 100, 1),
            "fat_share": round(share("Fat"), 3), "bone_share": round(share("Bone (graded)"), 3),
            "mean_body_speed_itis_m_s": round(float(z["speed_m_s"][body].mean()), 1),
        })
    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"packaged {len(rows)} samples into {args.out}")


if __name__ == "__main__":
    main()
