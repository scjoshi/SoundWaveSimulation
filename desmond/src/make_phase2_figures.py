"""Figures for desmond/docs/phase2_numerics.md from phase2_numerics.py results.

Re-runs a few cheap water/reciprocity shots for the waveform panels; the rest
comes from desmond/data/generated/phase2/results.json and aliasing_*.npz.

    python desmond/src/make_phase2_figures.py --sample desmond/data/generated/pathB_retrieved_n10/sample_0001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_simulation_figures import GRID, INK, QUIET, SERIES, SURFACE  # noqa: E402,F401
from phase2_numerics import F1, WATER, Medium, analytic_trace, run_shot  # noqa: E402

from fdtd2d import load_ct_ring_grid  # noqa: E402

LEVEL_COLORS = {"1x": SERIES[1], "2x": SERIES[3], "4x": SERIES[0]}


def fig_water(base: Medium, results, out):
    w = results["water"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.22})
    j = 128  # straight across the ring
    for f, key in ((1, "1x_bilinear"), (2, "2x_bilinear"), (4, "4x_bilinear")):
        m = base.water().refined(f)
        t, tr, src, ring, dt = run_shot(m, 0, source="bilinear")
        r = math.hypot(ring.x[j] - src[0], ring.y[j] - src[1])
        if f == 4:
            ref = analytic_trace(r, t, m.h, dt)
            a1.plot(t * 1e6, ref / np.abs(ref).max(), color=INK, lw=2.4, alpha=0.35, label="exact solution")
        y = tr[j] / np.abs(tr[j]).max()
        a1.plot(t * 1e6, y, color=LEVEL_COLORS[f"{f}x"], lw=1.2,
                label=f"{f}x grid: h = {m.h*1e3:.2f} mm, {WATER/F1/m.h:.0f} pts/λ at 250 kHz")
    T = r / WATER * 1e6
    a1.set_xlim(T - 3, T + 30)
    a1.set_xlabel("time (µs)")
    a1.set_ylabel("pressure (normalized)")
    a1.set_title(f"Receiver straight across, {r*1e3:.0f} mm of water")
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    a1.grid(axis="y", color=GRID, lw=0.8)
    for key, color, label in (("1x_bilinear", LEVEL_COLORS["1x"], "1x (default)"), ("2x_bilinear", LEVEL_COLORS["2x"], "2x"),
                              ("4x_bilinear", LEVEL_COLORS["4x"], "4x")):
        d = np.array(w[key]["distance_mm"])
        s = np.abs(np.array(w[key]["pick_err_ns"])) / 1e3
        a2.plot(d, s, "o-", color=color, ms=4, lw=1.4, label=label)
    a2.set_yscale("log")
    a2.axhspan(1e-3, 0.15, color=GRID, zorder=0)
    a2.text(5, 0.11, "budget: < 1 % of the 9–22 µs body delays", fontsize=8, color=QUIET, va="top")
    a2.set_xlabel("source–receiver distance (mm)")
    a2.set_ylabel("|first-arrival error| vs exact (µs)")
    a2.set_title("First-arrival error vs distance: 0.85 → 0.52 → 0.19 µs at 300 mm")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    a2.grid(color=GRID, lw=0.8)
    fig.suptitle("Water test: on the default grid the pulse arrives late and smeared; 4x refinement matches the exact solution",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.savefig(out / "water_dispersion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_converge(results, out):
    cv, tt = results["converge"], results.get("traveltime", {})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"wspace": 0.22})
    valid = np.array(cv["reference"]["valid"])
    n = valid.size
    for key, color, label in (("1x_nearest", SERIES[1], "1x, nearest-node source (as in 2DRingFDTD.py)"),
                              ("1x_bilinear", SERIES[2], "1x, bilinear source"),
                              ("2x_bilinear", SERIES[3], "2x, bilinear source")):
        e = np.array(cv[key]["delay_err_ns"], dtype=float) / 1e3
        e[~valid] = np.nan
        a1.plot(np.arange(n), e, color=color, lw=1.3, label=label)
    a1.axhspan(-0.15, 0.15, color=GRID, zorder=0)
    a1.axhline(0, color=QUIET, lw=0.8)
    a1.set_xlabel("receiver index (transmitter = 0)")
    a1.set_ylabel("error in body − water delay (µs)")
    a1.set_title("Sample 1: delay error vs 4x reference (gray = budget)")
    a1.legend(frameon=False, fontsize=8, loc="lower center")
    a1.grid(axis="y", color=GRID, lw=0.8)
    if tt:
        names = sorted(k for k in tt if k.startswith("sample_"))
        x = np.arange(len(names))
        for k, (key, color, marker, label) in enumerate((("1x", SERIES[1], "s", "1x grid (default)"),
                                                         ("4x", SERIES[0], "o", "4x grid"))):
            a2.plot(x + (k - 0.5) * 0.2, [tt[s][key]["travel_speed"] for s in names], marker, color=color, ms=7,
                    mec=SURFACE, mew=1.2, ls="none", label=label)
        a2.plot(x, [tt[s]["true_mean"] for s in names], "_", color=INK, ms=16, mew=2, ls="none", label="true mean body speed")
        a2.set_xticks(x, [f"#{int(s[-4:])}" for s in names])
        a2.set_ylabel("travel-time body speed (m/s)")
        a2.set_title("All 10 slices: default grid adds +5 to +8 m/s")
        a2.legend(frameon=False, fontsize=8, loc="upper right")
        a2.grid(axis="y", color=GRID, lw=0.8)
    fig.suptitle("CT medium: grid dispersion puts up to 1.3 µs (5–11 %) of error into the delays we analyze",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.savefig(out / "ct_convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_aliasing(data_dir, out):
    z = np.load(data_dir / "aliasing_body.npz")
    f, m, p = z["freqs"], z["m"], z["power"]
    f = f[f <= 400e3]  # power was saved up to 400 kHz
    order = np.argsort(m)
    m, p = m[order], p[order]
    keep = f <= 350e3
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    db = 10 * np.log10(p[:, keep] / p[:, keep].max() + 1e-12)
    im = ax.pcolormesh(f[keep] / 1e3, m, db, vmin=-50, vmax=0, cmap="Greys", shading="auto")
    R = float(z["radius_m"])
    ax.plot(f[keep] / 1e3, 2 * np.pi * f[keep] * R / WATER, color=SERIES[0], lw=1.2, ls="--", label="kR (wavefield's angular bandwidth)")
    ax.plot(f[keep] / 1e3, -2 * np.pi * f[keep] * R / WATER, color=SERIES[0], lw=1.2, ls="--")
    for n_el, color in ((256, SERIES[1]), (512, SERIES[2])):
        ax.axhline(n_el / 2, color=color, lw=1.4, label=f"Nyquist limit, {n_el} elements")
        ax.axhline(-n_el / 2, color=color, lw=1.4)
    ax.axvspan(100, 250, color=SERIES[3], alpha=0.08, label="chirp band 100–250 kHz")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("angular harmonic m (cycles around the ring)")
    ax.set_ylim(-300, 300)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("power (dB)")
    cb.outline.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("256 elements capture the wavefield only below ~190 kHz (R = 150 mm); 512 cover the whole band",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "ring_aliasing.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_reciprocity(base: Medium, out, i=0, j=96):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, source in zip(axes, ("nearest", "bilinear")):
        ti, tri, *_ = run_shot(base, i, source=source)
        tj, trj, *_ = run_shot(base, j, source=source)
        a, b = tri[j], np.interp(ti, tj, trj[i])
        k = np.abs(b).argmax()
        sl = slice(max(k - 500, 0), min(k + 900, ti.size))
        sc = np.abs(b).max()
        ax.plot(ti[sl] * 1e6, b[sl] / sc, color=QUIET, lw=2.2, alpha=0.5, label=f"receiver {i}, transmitter {j}")
        ax.plot(ti[sl] * 1e6, a[sl] / sc, color=SERIES[0] if source == "bilinear" else SERIES[1], lw=1.1,
                label=f"receiver {j}, transmitter {i}")
        err = np.linalg.norm(a - b) / np.linalg.norm(b)
        ax.set_title(f"{source} source: mismatch {err*100:.1f} %", fontsize=10)
        ax.set_xlabel("time (µs)")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("pressure (normalized)")
    fig.suptitle("Reciprocity (swap transmitter and receiver): exact with a bilinear source, broken by nearest-node snapping",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.savefig(out / "reciprocity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--map", default="acoustic_itis.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "generated" / "phase2")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "phase2")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads((args.data / "results.json").read_text())
    g = load_ct_ring_grid(args.sample / args.map, grid_shape=(512, 512))
    base = Medium(g.c, g.spacing_m, g.ring_radius_m)
    fig_water(base, results, args.out)
    fig_converge(results, args.out)
    fig_aliasing(args.data, args.out)
    fig_reciprocity(base, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
