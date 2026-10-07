"""Compare landmark-slice runs from src/generate_slices.py (e.g. Path A vs Path B).

For every sample it measures the slice the simulators will see:

* body size: cross-sectional area, AP and lateral diameters;
* tissue composition from ct_to_speed.segment_tissues (SAT, VAT, muscle,
  visceral organs, bone, internal gas), as fractions of the body area;
* median HU of key MAISI-labelled structures (liver, spleen, kidneys, L3,
  paraspinal/psoas muscle) and of SAT;
* bright non-bone pixels (> 200 HU outside bone labels), split into bowel
  (oral contrast), vessels and kidneys (IV contrast); ct_to_speed calls them bone;
* hybrid sound-speed statistics (ct_to_speed.map_hu_to_speed, air as water);
* generation cost: attempts and rejection reasons.

Outputs (in --out): per_sample.csv, summary.csv, summary.md, montage.png,
metrics.png.

    python desmond/src/compare_runs.py desmond/data/generated/pathA_generated_n10 \\
        desmond/data/generated/pathB_retrieved_n10 --names "Path A" "Path B" \\
        --out desmond/data/generated/compare_A_vs_B
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # advisor's code: ct_to_speed.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ct_to_speed  # noqa: E402  (repo-root module)

# MAISI label ids (configs/label_dict.json in the model repo).
BONE_IDS = set(range(33, 57)) | set(range(63, 98)) | {120, 122, 127}
BOWEL_IDS = {12, 13, 19, 62}  # stomach, duodenum, small bowel, colon
VESSEL_IDS = {6, 7, 17, 58, 59, 60, 61, 125}  # aorta, IVC, portal/splenic, iliac vessels, SVC
STRUCTURES = {
    "liver": [1],
    "spleen": [3],
    "kidney": [5, 14],
    "L3": [35],
    "muscle": [104, 105, 106, 107],  # autochthon + iliopsoas
}
BRIGHT_HU = 200.0
REJECTION_KINDS = {
    "landmark_truncated": ("touches the volume boundary",),
    "body_cut_in_plane": ("in-plane field of view",),
    "landmark_missing": ("not present", "does not contain required class labels"),
    "qc_failed": ("quality check",),
    "tumor_in_slice": ("tumor labels",),
}

# Reference categorical palette, slots 1-2 (dataviz skill references/palette.md).
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SERIES_MARKERS = ["o", "s"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e57"
GRID = "#e4e3dc"


def slice_metrics(hu: np.ndarray, label: np.ndarray, spacing_mm) -> dict:
    """Body, tissue, HU and sound-speed metrics for one radiological HU slice."""
    dy, dx = spacing_mm
    pixel_cm2 = dy * dx / 100.0
    tissue, body, _ = ct_to_speed.segment_tissues(hu.astype(float), spacing_mm=(dy, dx))
    rows, cols = np.nonzero(body)
    n_body = body.sum()

    out = {
        "body_area_cm2": n_body * pixel_cm2,
        "ap_diameter_mm": (rows.max() - rows.min() + 1) * dy,
        "lateral_diameter_mm": (cols.max() - cols.min() + 1) * dx,
    }
    for name, code in (
        ("sat", ct_to_speed.SAT),
        ("vat", ct_to_speed.VAT),
        ("muscle", ct_to_speed.MUSCLE),
        ("visceral", ct_to_speed.VISCERAL_ORGANS),
        ("bone", ct_to_speed.BONE),
        ("gas", ct_to_speed.AIR),
    ):
        out[f"frac_{name}"] = np.count_nonzero((tissue == code) & body) / n_body

    for name, ids in STRUCTURES.items():
        mask = np.isin(label, ids)
        out[f"hu_{name}"] = float(np.median(hu[mask])) if mask.any() else np.nan
    sat = (tissue == ct_to_speed.SAT) & body
    out["hu_sat"] = float(np.median(hu[sat])) if sat.any() else np.nan

    bright = (hu > BRIGHT_HU) & body & ~np.isin(label, list(BONE_IDS))
    out["bright_nonbone_frac"] = bright.sum() / n_body
    out["bright_bowel_cm2"] = np.count_nonzero(bright & np.isin(label, list(BOWEL_IDS))) * pixel_cm2
    out["bright_vessel_cm2"] = np.count_nonzero(bright & np.isin(label, list(VESSEL_IDS))) * pixel_cm2
    out["bright_kidney_cm2"] = np.count_nonzero(bright & np.isin(label, STRUCTURES["kidney"])) * pixel_cm2

    speed, _, _ = ct_to_speed.map_hu_to_speed(hu.astype(float), tissue, mapping="hybrid", air_as_water=True)
    out["speed_mean_body"] = float(speed[body].mean())
    out["speed_soft_mean"] = float(speed[body & (speed < 2000)].mean())
    out["frac_speed_gt_2000"] = np.count_nonzero(speed[body] > 2000) / n_body
    out["n_structures"] = len(set(np.unique(label[body]).tolist()) - {0, 200})
    return out


LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \w+ generate_slices: (.*)$")
ATTEMPT_START = re.compile(r"^sample (\d+) attempt (\d+): seed")
ATTEMPT_REJECTED = re.compile(r"^sample (\d+) attempt (\d+) rejected: (.*)$")
SAMPLE_SAVED = re.compile(r"^sample (\d+): saved")


def parse_log(run_dir: Path) -> dict:
    """Per-attempt outcomes and sampling time from <run>/run.log, if present.

    Attempts are keyed by (sample, attempt) and the latest session wins, so a
    resumed run (which replays deterministic attempts, or regenerates deleted
    samples) is counted once. Time is the summed duration of those attempts,
    i.e. GPU time spent sampling, excluding model loading.
    """
    log = run_dir / "run.log"
    if not log.exists():
        return {}
    attempts, open_attempt, pool_rejected = {}, {}, set()
    for line in log.read_text().splitlines():
        m = LOG_LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        msg = m.group(2)
        if (a := ATTEMPT_START.match(msg)):
            key = (int(a.group(1)), int(a.group(2)))
            open_attempt[key[0]] = (key, t)
        elif (r := ATTEMPT_REJECTED.match(msg)):
            key, t0 = open_attempt.pop(int(r.group(1)))
            attempts[key] = ("rejected", r.group(3), (t - t0).total_seconds())
        elif (sv := SAMPLE_SAVED.match(msg)):
            key, t0 = open_attempt.pop(int(sv.group(1)))
            attempts[key] = ("saved", "", (t - t0).total_seconds())
        elif msg.startswith("mask ") and " rejected:" in msg:
            pool_rejected.add(msg.split()[1])
    rejected = [reason for outcome, reason, _ in attempts.values() if outcome == "rejected"]
    counts = {kind: sum(any(k in reason for k in keys) for reason in rejected) for kind, keys in REJECTION_KINDS.items()}
    return {
        "rejections": counts,
        "n_attempts": len(attempts),
        "n_rejected": len(rejected),
        "n_pool_rejected": len(pool_rejected),
        "sampling_seconds": sum(sec for _, _, sec in attempts.values()),
    }


def load_run(run_dir: Path, name: str) -> list[dict]:
    rows = []
    for meta_path in sorted(run_dir.glob("sample_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        (slice_meta,) = meta["slices"][:1]
        hu = np.load(meta_path.parent / slice_meta["hu_file"])
        label = np.load(meta_path.parent / slice_meta["label_file"])
        row = {
            "run": name,
            "sample": meta["sample_index"],
            "seed": meta["seed"],
            "attempts": meta["attempt"] + 1,
            "mask_source": meta.get("mask_source", "generated"),
            "mask_id": (
                meta["retrieved_mask"]["file"]
                if "retrieved_mask" in meta
                else f"anatomy_db_{meta.get('anatomy_size_db_index')}"
            ),
            "slice_index": slice_meta["slice_index"],
            "l3_extent_mm": slice_meta["label_z_extent_mm"],
            "generation_seconds": meta["generation_seconds"],
        }
        row.update(slice_metrics(hu, label, slice_meta["spacing_mm"]))
        row["_hu"] = hu
        rows.append(row)
    return rows


SUMMARY_METRICS = [
    ("attempts", "Attempts per accepted sample", "{:.1f}"),
    ("slice_index", "L3 slice index (of 128)", "{:.0f}"),
    ("l3_extent_mm", "L3 z-extent (mm)", "{:.0f}"),
    ("body_area_cm2", "Body area (cm²)", "{:.0f}"),
    ("ap_diameter_mm", "AP diameter (mm)", "{:.0f}"),
    ("lateral_diameter_mm", "Lateral diameter (mm)", "{:.0f}"),
    ("frac_sat", "SAT fraction", "{:.2f}"),
    ("frac_vat", "VAT fraction", "{:.2f}"),
    ("frac_muscle", "Muscle fraction", "{:.2f}"),
    ("frac_visceral", "Visceral-organ fraction", "{:.2f}"),
    ("frac_bone", "Bone fraction", "{:.3f}"),
    ("frac_gas", "Internal gas fraction", "{:.3f}"),
    ("hu_liver", "Liver HU (median)", "{:.0f}"),
    ("hu_spleen", "Spleen HU (median)", "{:.0f}"),
    ("hu_kidney", "Kidney HU (median)", "{:.0f}"),
    ("hu_muscle", "Psoas/paraspinal HU (median)", "{:.0f}"),
    ("hu_sat", "SAT HU (median)", "{:.0f}"),
    ("hu_L3", "L3 vertebra HU (median)", "{:.0f}"),
    ("bright_nonbone_frac", "Bright non-bone fraction (>200 HU)", "{:.4f}"),
    ("bright_bowel_cm2", "Bright bowel area (cm², oral contrast)", "{:.1f}"),
    ("bright_vessel_cm2", "Bright vessel area (cm², IV contrast)", "{:.1f}"),
    ("bright_kidney_cm2", "Bright kidney area (cm², IV contrast)", "{:.1f}"),
    ("speed_soft_mean", "Mean soft-tissue speed (m/s)", "{:.0f}"),
    ("frac_speed_gt_2000", "Fraction of body > 2000 m/s", "{:.3f}"),
    ("n_structures", "Labelled structures in slice", "{:.0f}"),
]


def summarize(rows, names):
    """Median [min-max] per metric per run, plus SD for spread."""
    table = []
    for key, title, fmt in SUMMARY_METRICS:
        entry = {"metric": key, "title": title}
        for name in names:
            vals = np.array([r[key] for r in rows if r["run"] == name], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                entry[name] = "n/a"
                continue
            entry[name] = (
                f"{fmt.format(np.median(vals))} [{fmt.format(vals.min())}–{fmt.format(vals.max())}]"
            )
            entry[f"{name} sd"] = fmt.format(vals.std(ddof=1)) if vals.size > 1 else "n/a"
        table.append(entry)
    return table


def plot_montage(rows, names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = max(sum(r["run"] == n for r in rows) for n in names)
    fig, axes = plt.subplots(len(names), n_cols, figsize=(1.9 * n_cols, 2.2 * len(names)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r_i, name in enumerate(names):
        run_rows = [r for r in rows if r["run"] == name]
        for c_i, ax in enumerate(axes[r_i]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c_i >= len(run_rows):
                continue
            r = run_rows[c_i]
            hu = r["_hu"]
            body = ct_to_speed.body_mask_from_hu(hu.astype(float))
            ys, xs = np.nonzero(body)
            half = int(max(np.ptp(ys), np.ptp(xs)) / 2) + 12
            cy, cx = int(ys.mean()), int(xs.mean())
            crop = hu[max(cy - half, 0) : cy + half, max(cx - half, 0) : cx + half]
            ax.imshow(crop, cmap="gray", vmin=-200, vmax=300)
            ax.set_title(f"#{r['sample']}  seed {r['seed']}", fontsize=7, color=TEXT_SECONDARY)
        axes[r_i, 0].set_ylabel(name, fontsize=10, color=TEXT_PRIMARY)
    fig.suptitle("Mid-L3 slices (W500/L50, cropped to body)", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


PLOT_METRICS = [
    ("body_area_cm2", "Body area (cm²)"),
    ("lateral_diameter_mm", "Lateral diameter (mm)"),
    ("frac_sat", "SAT fraction"),
    ("frac_vat", "VAT fraction"),
    ("frac_muscle", "Muscle fraction"),
    ("hu_liver", "Liver HU"),
    ("hu_muscle", "Psoas/paraspinal HU"),
    ("bright_nonbone_frac", "Bright non-bone fraction"),
    ("speed_soft_mean", "Soft-tissue speed (m/s)"),
    ("attempts", "Attempts per sample"),
]


def plot_metrics(rows, names, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(PLOT_METRICS)
    n_cols = 5
    fig, axes = plt.subplots(int(np.ceil(n / n_cols)), n_cols, figsize=(3.0 * n_cols, 3.0 * np.ceil(n / n_cols)))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes.ravel(), PLOT_METRICS):
        ax.set_facecolor(SURFACE)
        for i, name in enumerate(names):
            vals = np.array([r[key] for r in rows if r["run"] == name], dtype=float)
            jitter = rng.uniform(-0.12, 0.12, vals.size)
            ax.scatter(i + jitter, vals, s=36, color=SERIES_COLORS[i], marker=SERIES_MARKERS[i],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
            if np.isfinite(vals).any():
                ax.hlines(np.nanmedian(vals), i - 0.28, i + 0.28, color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax.set_title(title, fontsize=9, color=TEXT_PRIMARY)
        ax.set_xticks(range(len(names)), names, fontsize=8, color=TEXT_SECONDARY)
        ax.tick_params(axis="y", labelsize=7, colors=TEXT_SECONDARY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Per-sample metrics (dots) and medians (bars)", fontsize=11, color=TEXT_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories from generate_slices.py")
    parser.add_argument("--names", nargs="+", help="Display names (default: run directory names)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args(argv)

    names = args.names or [r.name for r in args.runs]
    if len(names) != len(args.runs):
        parser.error("--names must match the number of runs")
    if len(names) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} runs are supported")
    args.out.mkdir(parents=True, exist_ok=True)

    rows, logs = [], {}
    for run_dir, name in zip(args.runs, names):
        rows += load_run(run_dir, name)
        logs[name] = parse_log(run_dir)

    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(args.out / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    table = summarize(rows, names)
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    lines = ["| Metric | " + " | ".join(f"{n}: median [min–max]" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for entry in table:
        lines.append(f"| {entry['title']} | " + " | ".join(entry[n] for n in names) + " |")
    lines.append("")
    for name in names:
        run_rows = [r for r in rows if r["run"] == name]
        log = logs[name]
        unique = len({r["mask_id"] for r in run_rows})
        lines.append(f"- **{name}**: {len(run_rows)} samples, {unique} distinct masks/conditions"
                     + (f", {log['n_attempts']} generation attempts ({log['n_rejected']} rejected: "
                        f"{log['rejections']}), {log['n_pool_rejected']} masks dropped from the retrieval pool, "
                        f"GPU sampling time {log['sampling_seconds'] / 60:.1f} min "
                        f"({log['sampling_seconds'] / max(len(run_rows), 1) / 60:.1f} min per accepted sample)"
                        if log else ""))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")

    plot_montage(rows, names, args.out / "montage.png")
    plot_metrics(rows, names, args.out / "metrics.png")
    print("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
