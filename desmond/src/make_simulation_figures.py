"""Make the interpretation figures for docs/simulation_pipeline.md.

Inputs (from a generate_slices.py run, after ct_to_speed.py and the
simulations described in the doc):

* sample_XXXX/L3_mid_hu.npy, acoustic.npz           every sample
* sample_XXXX/ring_traces.npz, ring_traces_water.npz   forward ring FDTD on the
  CT map and on the same geometry filled with water (2DRingFDTD --save-traces)
* <inversion sample>/inversions/{ring,arc,fdfd}_maps.npz  from capture_inversion.py

Outputs (docs/figures/): sample_gallery.png, forward_explained.png,
delay_profiles.png, travel_time_vs_fat.png, inversion_maps.png,
inversion_misfit.png, plus simulation_metrics.csv next to them.

    python desmond/src/make_simulation_figures.py \\
        --run desmond/data/generated/pathB_retrieved_n10 --inversion-sample 1
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # desmond/
REPO_ROOT = ROOT.parent  # advisor's code: ct_to_speed.py, fdtd2d/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ct_to_speed  # noqa: E402
from fdtd2d import load_ct_ring_grid  # noqa: E402

WATER = 1480.0
BONE_SPEED = 2000.0  # anything faster is the bone class (3476 m/s)
EXCLUDE_NEAR_TX = 8  # receivers this close to the transmitter see water only
PICK_FRACTION = 0.02  # first arrival: |p| above 2 % of the water trace's peak

# Reference palette (dataviz skill references/palette.md).
SURFACE, INK, QUIET, GRID = "#fcfcfb", "#1a1a19", "#5f5e57", "#e4e3dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Diverging blue <-> red around a neutral gray: slower than water is blue,
# faster is red. The red arm mirrors the blue ramp's steps.
DIVERGING = LinearSegmentedColormap.from_list(
    "speed", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a9a8", "#e34948", "#8f2525"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": QUIET, "ytick.color": QUIET, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


def samples(run: Path):
    return sorted(p for p in run.glob("sample_*") if (p / "acoustic.npz").exists())


def first_arrivals(traces, dt, reference, dt_ref):
    """First time each receiver's |p| exceeds 2 % of its water-only peak.

    The two runs have their own time steps: the solver sizes dt and the record
    length from the fastest and slowest speeds in each map.
    """
    thresholds = PICK_FRACTION * np.abs(reference).max(axis=1, keepdims=True)
    picks = []
    for trace, step in ((traces, dt), (reference, dt_ref)):
        above = np.abs(trace) > thresholds
        idx = np.where(above.any(axis=1), above.argmax(axis=1), -1)
        picks.append(np.where(idx >= 0, idx * step, np.nan))
    return picks  # body, water


def body_path_mm(grid, tx_xy, rx_xy):
    """Straight-ray path length (mm) inside the body from tx to each receiver."""
    x, y = grid.x_m, grid.y_m
    dx = x[1] - x[0]
    lengths = []
    for rx in rx_xy:
        n = int(np.hypot(*(rx - tx_xy)) / (dx / 2)) + 1
        px = np.linspace(tx_xy[0], rx[0], n)
        py = np.linspace(tx_xy[1], rx[1], n)
        ix = np.clip(np.rint((px - x[0]) / dx).astype(int), 0, x.size - 1)
        iy = np.clip(np.rint((py - y[0]) / dx).astype(int), 0, y.size - 1)
        inside = grid.body_mask[iy, ix]  # fdtd2d arrays are [row = y, col = x]
        lengths.append(inside.mean() * np.hypot(*(rx - tx_xy)) * 1e3)
    return np.array(lengths)


def forward_analysis(sample: Path, traces="ring_traces.npz", water_traces="ring_traces_water.npz",
                     acoustic="acoustic.npz"):
    body = np.load(sample / traces)
    water = np.load(sample / water_traces)
    grid = load_ct_ring_grid(sample / acoustic, grid_shape=body["sound_speed_m_s"].shape)
    tx = int(body["tx"])
    xy = np.stack([body["x"], body["y"]], axis=1)
    t_body, t_water = first_arrivals(body["traces"], float(body["dt"]), water["traces"], float(water["dt"]))
    n = len(xy)
    offset = np.minimum(np.abs(np.arange(n) - tx), n - np.abs(np.arange(n) - tx))
    valid = offset > EXCLUDE_NEAR_TX
    path = body_path_mm(grid, xy[tx], xy)
    return {
        "body": body, "water": water, "grid": grid, "tx": tx, "xy": xy,
        "delay_us": np.where(valid, (t_body - t_water) * 1e6, np.nan),
        "t_body": t_body, "t_water": t_water, "path_mm": path, "valid": valid,
    }


def tissue_fractions(sample: Path):
    z = np.load(sample / "acoustic.npz")
    labels, body = z["labels"], z["body_mask"]
    frac = lambda code: np.count_nonzero((labels == code) & body) / body.sum()  # noqa: E731
    return {
        "fat": frac(ct_to_speed.SAT) + frac(ct_to_speed.VAT),
        "bone": frac(ct_to_speed.BONE),
        "mean_speed": float(z["speed_m_s"][body].mean()),
    }


def fig_gallery(paths, out):
    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(1.75 * n, 4.6), gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.24)
    norm = TwoSlopeNorm(vmin=1380, vcenter=WATER, vmax=1620)
    for i, p in enumerate(paths):
        hu = np.load(p / "L3_mid_hu.npy").astype(float)
        speed = np.load(p / "acoustic.npz")["speed_m_s"]
        rows, cols = np.nonzero(ct_to_speed.body_mask_from_hu(hu))
        half = max(np.ptp(rows), np.ptp(cols)) // 2 + 10
        cy, cx = int(rows.mean()), int(cols.mean())
        crop = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
        axes[0, i].imshow(hu[crop], cmap="gray", vmin=-200, vmax=300)
        im = axes[1, i].imshow(speed[crop], cmap=DIVERGING, norm=norm)
        axes[0, i].set_title(f"#{int(p.name[-4:])}", fontsize=9, color=QUIET, loc="center", fontweight="normal")
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("CT (HU)", fontsize=10)
    axes[1, 0].set_ylabel("Sound speed", fontsize=10)
    cbar = fig.colorbar(im, cax=fig.add_axes([0.3, 0.13, 0.4, 0.035]), orientation="horizontal")
    cbar.set_label("m/s   (blue: slower than water, mostly fat · red: faster, lean tissue · dark red: bone 3476)")
    cbar.outline.set_visible(False)
    fig.suptitle("The 10 synthetic mid-L3 slices and the sound-speed maps the simulators see",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_forward(a, sample_name, out):
    body, water, grid, tx = a["body"], a["water"], a["grid"], a["tx"]
    t_us = body["time_s"] * 1e6
    n = len(a["xy"])
    opp = (tx + n // 2) % n
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.25], wspace=0.5)

    ax = fig.add_subplot(gs[0])
    ext = [grid.x_m[0] * 1e3, grid.x_m[-1] * 1e3, grid.y_m[0] * 1e3, grid.y_m[-1] * 1e3]
    im = ax.imshow(body["sound_speed_m_s"], origin="lower", extent=ext, cmap=DIVERGING,
                   norm=TwoSlopeNorm(vmin=1380, vcenter=WATER, vmax=1620))
    ax.plot(a["xy"][:, 0] * 1e3, a["xy"][:, 1] * 1e3, ".", ms=1.5, color=QUIET)
    ax.plot(*(a["xy"][tx] * 1e3), "o", ms=8, color=SERIES[1], mec=SURFACE, mew=1.5, ls="none", label="transmitter (0)")
    ax.plot(*(a["xy"][opp] * 1e3), "s", ms=7, color=SERIES[0], mec=SURFACE, mew=1.5, ls="none", label=f"opposite receiver ({opp})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1, frameon=False, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    ax.set_title("A. Sound-speed map in the ring\n(simulator view: anterior at bottom)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = fig.add_subplot(gs[1])
    gather = body["traces"]
    lim = np.percentile(np.abs(gather), 99.5)
    ax.imshow(gather, aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim,
              extent=[t_us[0], t_us[-1], n - 0.5, -0.5])
    ax.plot(a["t_water"] * 1e6, np.arange(n), "--", color=INK, lw=1, label="water-only first arrival")
    ax.set_xlim(0, t_us[-1])
    ax.set_title("B. Recorded pressure at all 256 receivers")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("receiver index (transmitter = 0)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2])
    ax.plot(water["time_s"] * 1e6, water["traces"][opp], color=QUIET, lw=1.2, label="water only")
    ax.plot(t_us, body["traces"][opp], color=SERIES[0], lw=1.4, label="through the abdomen")
    d = a["delay_us"][opp]
    ax.axvline(a["t_water"][opp] * 1e6, color=QUIET, lw=0.8, ls=":")
    ax.axvline(a["t_body"][opp] * 1e6, color=SERIES[0], lw=0.8, ls=":")
    ax.set_xlim(a["t_water"][opp] * 1e6 - 40, a["t_water"][opp] * 1e6 + 120)
    ax.set_title(f"C. Opposite receiver: arrives {abs(d):.1f} µs {'earlier' if d < 0 else 'later'} than water")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("pressure")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"Forward ring simulation, sample {sample_name}: the body speeds up the first arrival and scatters the rest into a long coda",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_delay_profiles(analyses, names, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    n = len(analyses[0]["xy"])
    for i, (a, name) in enumerate(zip(analyses, names)):
        ax.plot(np.arange(n), a["delay_us"], color=SERIES[i], lw=1.6, label=name)
    ax.axhline(0, color=QUIET, lw=1)
    lo = np.nanmin([np.nanmin(a["delay_us"]) for a in analyses])
    ax.text(n - 2, 1.2, "↑ later than water: path mostly through fat", ha="right", va="bottom", fontsize=9, color=QUIET)
    ax.text(n - 2, lo, "↓ earlier than water: path through lean tissue and bone", ha="right", va="bottom", fontsize=9, color=QUIET)
    ax.set_ylim(lo - 1.5, 4)
    ax.set_xlabel("receiver index (transmitter = 0; 128 = straight across)")
    ax.set_ylabel("first-arrival delay vs water (µs)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Straight-across paths arrive early through lean tissue; fattier bodies shrink the lead")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_travel_time_vs_fat(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.array([r["fat_fraction"] for r in rows]) * 100
    y = np.array([r["travel_time_speed"] for r in rows])
    truth = np.array([r["true_mean_speed"] for r in rows])
    ax.scatter(x, truth, s=50, facecolor="none", edgecolor=QUIET, lw=1.2, label="true mean body speed (map)", zorder=2)
    ax.scatter(x, y, s=55, color=SERIES[0], edgecolor=SURFACE, lw=1.5, label="estimated from first arrivals", zorder=3)
    for xi, yi, r in zip(x, y, rows):
        ax.annotate(f"#{r['sample']}", (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8, color=QUIET)
    ax.axhline(WATER, color=QUIET, lw=0.8, ls=":")
    ax.text(x.min(), WATER + 2, "water 1480", fontsize=8, color=QUIET, va="bottom")
    ax.set_xlabel("fat share of the body cross-section (%)")
    ax.set_ylabel("average speed along straight paths (m/s)")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title("Fatter abdomens are slower: travel time encodes body composition")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def load_maps(sample: Path):
    maps = {}
    for key, label in (("ring", "Ring FWI (256 el.)"), ("arc", "Arc FWI (2 × 64 el.)"), ("fdfd", "Frequency-domain FWI")):
        f = sample / "inversions" / f"{key}_maps.npz"
        if f.exists():
            z = np.load(f)
            maps[key] = {"label": label, **{k: z[k] for k in ("c_true", "c_est", "mask", "misfit")}}
    return maps


def rms(est, true, sel):
    return float(np.sqrt(np.mean((est[sel] - true[sel]) ** 2))) if sel.any() else float("nan")


def inversion_metrics(maps):
    rows = []
    first = next(iter(maps.values()))
    for key, m in [("start", {**first, "label": "Uniform 1480 m/s start", "c_est": np.full_like(first["c_true"], WATER)}),
                   *maps.items()]:
        mask, ct = m["mask"], m["c_true"]
        soft = mask & (ct < BONE_SPEED)
        rows.append({
            "method": m["label"], "rms_interior": rms(m["c_est"], ct, mask),
            "rms_soft_tissue": rms(m["c_est"], ct, soft),
            # each method runs on its own grid, so its starting error is measured on that grid
            "rms_soft_tissue_start": rms(np.full_like(ct, WATER), ct, soft),
            "rms_bone": rms(m["c_est"], ct, mask & ~soft),
            "est_min": float(m["c_est"][mask].min()), "est_max": float(m["c_est"][mask].max()),
        })
    return rows


def fig_inversion_maps(maps, sample: Path, out):
    keys = list(maps)
    fig, axes = plt.subplots(2, len(keys) + 1, figsize=(3.3 * (len(keys) + 1), 6.6),
                             gridspec_kw={"hspace": 0.22, "wspace": 0.06})
    speed_norm = TwoSlopeNorm(vmin=1380, vcenter=WATER, vmax=1620)
    err_norm = TwoSlopeNorm(vmin=-150, vcenter=0, vmax=150)

    def show(ax, field, mask, norm, title):
        grid = load_ct_ring_grid(sample / "acoustic.npz", grid_shape=field.shape)
        ext = [grid.x_m[0] * 1e3, grid.x_m[-1] * 1e3, grid.y_m[0] * 1e3, grid.y_m[-1] * 1e3]
        shown = np.where(mask, field, np.nan)
        im = ax.imshow(shown, origin="lower", extent=ext, cmap=DIVERGING, norm=norm)
        r = grid.ring_radius_m * 1e3
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10)
        return im

    ref = maps[keys[0]]
    im_speed = show(axes[0, 0], ref["c_true"], ref["mask"], speed_norm, "True map")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.55, "Bottom row: error\n(reconstruction − truth).\nThe spine saturates\ndark blue: it is 3476 m/s\nbut capped at 2000.",
                    transform=axes[1, 0].transAxes, fontsize=9, color=QUIET, va="center")
    for j, key in enumerate(keys, start=1):
        m = maps[key]
        show(axes[0, j], m["c_est"], m["mask"], speed_norm, m["label"])
        im_err = show(axes[1, j], m["c_est"] - m["c_true"], m["mask"], err_norm, "error")
    c1 = fig.colorbar(im_speed, ax=axes[0, :], fraction=0.025, pad=0.01)
    c1.set_label("sound speed (m/s)")
    c2 = fig.colorbar(im_err, ax=axes[1, :], fraction=0.025, pad=0.01)
    c2.set_label("error (m/s)")
    for c in (c1, c2):
        c.outline.set_visible(False)
    fig.suptitle(f"Default inversions, sample {int(sample.name[-4:])}: the ring starts to recover soft tissue; the arc and frequency-domain runs barely move",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_misfit(maps, out):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (key, m) in enumerate(maps.items()):
        j = np.asarray(m["misfit"], dtype=float)
        if j.size:
            ax.plot(np.arange(j.size), j / j[0], marker="o", ms=3.5, lw=1.5, color=SERIES[i], label=m["label"])
    ax.axhline(1, color=QUIET, lw=0.8, ls=":")
    ax.set_xlabel("iteration (final stage)")
    ax.set_ylabel("misfit / starting misfit")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("A falling misfit is not a better map: the arc fits its data best\nyet its soft-tissue error rises")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="generate_slices.py run directory")
    parser.add_argument("--inversion-sample", type=int, default=1)
    parser.add_argument("--profile-samples", type=int, nargs="+", help="Samples for delay_profiles.png")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = samples(args.run)
    fig_gallery(paths, args.out / "sample_gallery.png")

    rows, analyses = [], {}
    for p in paths:
        if not (p / "ring_traces_water.npz").exists():
            continue
        a = forward_analysis(p)
        analyses[int(p.name[-4:])] = a
        through = a["valid"] & (a["path_mm"] > 20) & np.isfinite(a["delay_us"])
        # Straight-ray estimate: delay = L_body (1/c - 1/c_water)  =>  c = L / (L / c_w + delay)
        L = a["path_mm"][through].sum() * 1e-3
        dly = np.nansum(a["delay_us"][through]) * 1e-6
        tf = tissue_fractions(p)
        n = len(a["xy"])
        rows.append({
            "sample": int(p.name[-4:]), "fat_fraction": tf["fat"], "bone_fraction": tf["bone"],
            "true_mean_speed": tf["mean_speed"], "travel_time_speed": L / (L / WATER + dly),
            "opposite_delay_us": float(a["delay_us"][(a["tx"] + n // 2) % n]),
        })

    if analyses:
        typical = args.inversion_sample if args.inversion_sample in analyses else next(iter(analyses))
        fig_forward(analyses[typical], f"{typical}", args.out / "forward_explained.png")
        by_fat = sorted(rows, key=lambda r: r["fat_fraction"])
        picks = args.profile_samples or [by_fat[0]["sample"], by_fat[len(by_fat) // 2]["sample"], by_fat[-1]["sample"]]
        names = [f"#{s} (fat {next(r['fat_fraction'] for r in rows if r['sample'] == s) * 100:.0f} %)" for s in picks]
        fig_delay_profiles([analyses[s] for s in picks], names, args.out / "delay_profiles.png")
        fig_travel_time_vs_fat(rows, args.out / "travel_time_vs_fat.png")

    inv_sample = args.run / f"sample_{args.inversion_sample:04d}"
    maps = load_maps(inv_sample)
    metrics = []
    if maps:
        fig_inversion_maps(maps, inv_sample, args.out / "inversion_maps.png")
        fig_misfit(maps, args.out / "inversion_misfit.png")
        metrics = inversion_metrics(maps)

    with open(args.out / "simulation_metrics.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if metrics:
            f.write("\n")
            w = csv.DictWriter(f, fieldnames=list(metrics[0]))
            w.writeheader()
            w.writerows(metrics)
    for r in rows:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    for m in metrics:
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in m.items()})
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
