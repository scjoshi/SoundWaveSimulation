"""Validate Phase 1 sound-speed maps against the IT'IS tissue database.

For every tissue in every sample, compares the advisor's ``ct_to_speed.py``
hybrid map (``acoustic.npz``) and the IT'IS-anchored map
(``acoustic_itis.npz``, from speed_map.py) on the same pixels. The pixels'
tissue comes from the IT'IS map's tissue assignment.

Pass rule (stated in desmond/docs/phase1_speed_maps.md): a tissue passes when
the median speed over all its pixels, pooled across samples, lies within the
IT'IS mean +- 1 SD. A single-study tissue uses +-15 m/s.

Outputs (default desmond/docs/figures/phase1/): tissue_speed_table.csv,
tissue_speed_validation.png, speed_map_comparison.png.

    python desmond/src/validate_speed_maps.py --run desmond/data/generated/pathB_retrieved_n10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_simulation_figures import (  # noqa: E402
    DIVERGING, GRID, INK, QUIET, SERIES, SURFACE, WATER, forward_analysis,
)
from speed_map import load_config  # noqa: E402

DEFAULT_SD = 15.0
SKIP = {"Coupling water"}
BONE_REF = "Bone (Cancellous)"  # the L3 vertebral body is mostly cancellous bone


def pooled(run: Path, itis_name: str):
    """Per-tissue pixel speeds pooled over samples, for both mappings."""
    old, new, share = {}, {}, {}
    n_body = 0
    for s in sorted(run.glob("sample_*")):
        if not (s / itis_name).exists():
            continue
        z, a = np.load(s / itis_name), np.load(s / "acoustic.npz")
        names, idx, body = z["tissue_names"], z["tissue_index"], z["body_mask"]
        n_body += body.sum()
        for i, name in enumerate(names):
            name = str(name)
            m = (idx == i) & body
            if name in SKIP or not m.any():
                continue
            old.setdefault(name, []).append(a["speed_m_s"][m])
            new.setdefault(name, []).append(z["speed_m_s"][m])
            share[name] = share.get(name, 0) + m.sum()
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}  # noqa: E731
    return cat(old), cat(new), {k: v / n_body for k, v in share.items()}


def reference(name, itis):
    key = BONE_REF if name == "Bone (graded)" else name.replace("Gas as ", "")
    row = itis[key]
    sd = row["speed_sd"] if row["speed_sd"] > 0 else DEFAULT_SD
    return key, row["speed"], sd, row["speed_n"]


def table(old, new, share, itis):
    rows = []
    for name in sorted(new, key=lambda k: -share[k]):
        key, mean, sd, n = reference(name, itis)
        q_old, q_new = np.percentile(old[name], [25, 50, 75]), np.percentile(new[name], [25, 50, 75])
        rows.append({
            "tissue": name, "itis_reference": key, "itis_mean": mean, "itis_sd": sd, "itis_n": n,
            "body_share_pct": 100 * share[name],
            "advisor_median": q_old[1], "advisor_q25": q_old[0], "advisor_q75": q_old[2],
            "new_median": q_new[1], "new_q25": q_new[0], "new_q75": q_new[2],
            "advisor_pass": abs(q_old[1] - mean) <= sd, "new_pass": abs(q_new[1] - mean) <= sd,
        })
    return rows


def fig_validation(rows, out):
    soft = [r for r in rows if r["tissue"] != "Bone (graded)"]
    bone = [r for r in rows if r["tissue"] == "Bone (graded)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows) + 2.2),
                             gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.06})
    for ax, group in zip(axes, (soft, bone)):
        for y, r in enumerate(group):
            ax.add_patch(plt.Rectangle((r["itis_mean"] - r["itis_sd"], y - 0.32), 2 * r["itis_sd"], 0.64,
                                       color=GRID, zorder=1))
            ax.plot([r["itis_mean"]] * 2, [y - 0.32, y + 0.32], color=QUIET, lw=1.5, zorder=2)
            for dy, pre, color, marker in ((0.12, "advisor", SERIES[1], "s"), (-0.12, "new", SERIES[0], "o")):
                ax.plot([r[f"{pre}_q25"], r[f"{pre}_q75"]], [y + dy] * 2, color=color, lw=2, zorder=3)
                ax.plot(r[f"{pre}_median"], y + dy, marker, color=color, ms=7, mec=SURFACE, mew=1.2, zorder=4)
        ax.set_yticks(range(len(group)), [f"{r['tissue']} ({r['body_share_pct']:.1f} %)" for r in group])
        ax.set_ylim(-0.7, max(len(group), 1) - 0.3)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("sound speed (m/s)")
    axes[1].set_yticks([])
    axes[0].set_title("Soft tissues (share of body area)")
    axes[1].set_title(f"Bone, graded ({bone[0]['body_share_pct']:.1f} %)\nvs IT'IS cancellous" if bone else "Bone")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRID),
               plt.Line2D([], [], color=SERIES[1], marker="s", lw=2, mec=SURFACE),
               plt.Line2D([], [], color=SERIES[0], marker="o", lw=2, mec=SURFACE)]
    fig.legend(handles, ["IT'IS mean ± 1 SD", "advisor hybrid map: median, IQR", "IT'IS-anchored map: median, IQR"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Every tissue now sits in its IT'IS range; the advisor map misses the contrast-enhanced organs and bone",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.12)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_maps(run: Path, itis_name: str, picks, out):
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)), gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    norm = TwoSlopeNorm(vmin=1380, vcenter=WATER, vmax=1620)
    dnorm = TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)
    for row, k in zip(np.atleast_2d(axes), picks):
        s = run / f"sample_{k:04d}"
        hu = np.load(s / "L3_mid_hu.npy")
        a, z = np.load(s / "acoustic.npz")["speed_m_s"], np.load(s / itis_name)["speed_m_s"]
        rows_, cols_ = np.nonzero(np.load(s / itis_name)["body_mask"])
        crop = (slice(rows_.min() - 8, rows_.max() + 8), slice(cols_.min() - 8, cols_.max() + 8))
        row[0].imshow(hu[crop], cmap="gray", vmin=-200, vmax=300)
        row[1].imshow(a[crop], cmap=DIVERGING, norm=norm)
        im = row[2].imshow(z[crop], cmap=DIVERGING, norm=norm)
        imd = row[3].imshow((z - a)[crop], cmap=DIVERGING, norm=dnorm)
        row[0].set_ylabel(f"sample {k}", fontsize=10)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    titles = ["CT (HU)", "Advisor hybrid map", "IT'IS-anchored map", "Change (new − advisor)"]
    for ax, t in zip(np.atleast_2d(axes)[0], titles):
        ax.set_title(t, fontsize=10)
    c1 = fig.colorbar(im, ax=np.atleast_2d(axes)[:, 1:3], orientation="horizontal", fraction=0.04, pad=0.03)
    c1.set_label("sound speed (m/s); bone saturates")
    c2 = fig.colorbar(imd, ax=np.atleast_2d(axes)[:, 3], orientation="horizontal", fraction=0.04, pad=0.03)
    c2.set_label("change (m/s)")
    for c in (c1, c2):
        c.outline.set_visible(False)
    fig.suptitle("Contrast-enhanced kidneys, vessels and bowel slow by 50-90 m/s (blue), the spine is graded, CT noise is smoothed",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


VARIANTS = (  # (label, traces file, acoustic file, colour, marker)
    ("advisor hybrid map", "ring_traces.npz", "acoustic.npz", SERIES[1], "s"),
    ("IT'IS map, gas as water", "ring_traces_itis.npz", "acoustic_itis.npz", SERIES[0], "o"),
    ("IT'IS map, gas as bowel contents", "ring_traces_itis_gas-lumen.npz", "acoustic_itis_gas-lumen.npz", SERIES[2], "D"),
)


def travel_time_speed(a):
    """Straight-ray average body speed from one shot's first-arrival delays (as in make_simulation_figures)."""
    sel = a["valid"] & (a["path_mm"] > 20) & np.isfinite(a["delay_us"])
    L = a["path_mm"][sel].sum() * 1e-3
    return L / (L / WATER + np.nansum(a["delay_us"][sel]) * 1e-6)


def forward_effect(run: Path, out_dir: Path, profile_sample: int):
    """How much do the Phase 1 corrections and the gas choice change what the ring records?"""
    rows, profiles = [], {}
    for s in sorted(run.glob("sample_*")):
        if not all((s / t).exists() for _, t, *_ in VARIANTS) or not (s / "ring_traces_water.npz").exists():
            continue
        k = int(s.name[-4:])
        row = {"sample": k}
        for label, traces, acoustic, *_ in VARIANTS:
            a = forward_analysis(s, traces=traces, acoustic=acoustic)
            z = np.load(s / acoustic)
            n = len(a["xy"])
            row[f"{label}|travel_speed"] = travel_time_speed(a)
            row[f"{label}|true_mean"] = float(z["speed_m_s"][z["body_mask"]].mean())
            row[f"{label}|opposite_delay_us"] = float(a["delay_us"][(a["tx"] + n // 2) % n])
            if k == profile_sample:
                profiles[label] = a["delay_us"]
        rows.append(row)
    if not rows:
        return []
    with open(out_dir / "forward_effect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.25})
    x = np.arange(len(rows))
    for j, (label, _, _, color, marker) in enumerate(VARIANTS):
        y = [r[f"{label}|travel_speed"] for r in rows]
        a1.plot(x + (j - 1) * 0.18, y, marker, color=color, ms=7, mec=SURFACE, mew=1.2, ls="none", label=label)
    a1.set_xticks(x, [f"#{r['sample']}" for r in rows])
    a1.set_xlabel("sample")
    a1.set_ylabel("average body speed from travel times (m/s)")
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    a1.set_title("Travel-time body speed, per sample")
    n = len(next(iter(profiles.values()))) if profiles else 0
    for label, _, _, color, _ in VARIANTS:
        if label in profiles:
            a2.plot(np.arange(n), profiles[label], color=color, lw=1.5, label=label)
    a2.axhline(0, color=QUIET, lw=1)
    a2.set_xlabel("receiver index (transmitter = 0)")
    a2.set_ylabel("first-arrival delay vs water (µs)")
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_title(f"Delay profile, sample {profile_sample}")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Phase 1 corrections lower travel-time body speed by a median 26 m/s; the gas choice changes it by only 1-8 m/s",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.savefig(out_dir / "forward_effect.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--itis-name", default="acoustic_itis.npz")
    parser.add_argument("--picks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "phase1")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    itis = load_config()["itis"]
    old, new, share = pooled(args.run, args.itis_name)
    rows = table(old, new, share, itis)
    with open(args.out / "tissue_speed_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_validation(rows, args.out / "tissue_speed_validation.png")
    fig_maps(args.run, args.itis_name, args.picks, args.out / "speed_map_comparison.png")
    print("| Tissue | Body share | IT'IS mean ± SD (n) | Advisor median [IQR] | New median [IQR] | Advisor pass | New pass |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tissue']} | {r['body_share_pct']:.1f} % | {r['itis_mean']:.0f} ± {r['itis_sd']:.0f} ({r['itis_n']}) "
              f"| {r['advisor_median']:.0f} [{r['advisor_q25']:.0f}–{r['advisor_q75']:.0f}] "
              f"| {r['new_median']:.0f} [{r['new_q25']:.0f}–{r['new_q75']:.0f}] | {'yes' if r['advisor_pass'] else 'no'} | {'yes' if r['new_pass'] else 'no'} |")
    rows_fwd = forward_effect(args.run, args.out, args.picks[0])
    if rows_fwd:
        labels = [v[0] for v in VARIANTS]
        print("\nsample | " + " | ".join(f"{l}: travel speed (true mean), opposite delay" for l in labels))
        for r in rows_fwd:
            print(r["sample"], " | ".join(f"{r[l + '|travel_speed']:.0f} ({r[l + '|true_mean']:.0f}), {r[l + '|opposite_delay_us']:.1f} us" for l in labels))
        for l in labels[1:]:
            d = np.array([r[l + "|travel_speed"] - r[labels[0] + "|travel_speed"] for r in rows_fwd])
            print(f"{l} minus advisor: travel speed change median {np.median(d):.1f}, range {d.min():.1f} to {d.max():.1f} m/s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
