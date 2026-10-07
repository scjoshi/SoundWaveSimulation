"""Phase 2: verify the forward simulator's numerical accuracy.

Uses the advisor's GPU solver (fdtd2d.torch_backend: TorchScalarWave2D,
TorchFDTD2D, TorchRingSampler) unchanged; only the driver loop is new, so it
can (a) run the *same* medium at finer grids, (b) inject the source either at
the nearest node (as 2DRingFDTD.py does) or bilinearly, and (c) use a fixed
continuous source pulse on every grid.

Experiments (geometry of one sample, its Phase 1 map):

  water      homogeneous water vs the exact 2D Green's function: absolute
             timing error (numerical dispersion) at 1x, 2x, 4x refinement
  converge   the CT map at 1x, 2x, 4x refinement (nearest-neighbour
             upsampling, so the medium is identical): first-arrival and
             body-minus-water delay errors against the 4x reference
  boundary   default domain vs one padded by 128 cells per side (Mur
             absorbing-boundary reflections)
  reciprocity d_ij vs d_ji for several element pairs, nearest vs bilinear source
  aliasing   1024 receivers on the ring; energy beyond the angular Nyquist
             limit of 256 and 512 elements

    python desmond/src/phase2_numerics.py --sample desmond/data/generated/pathB_retrieved_n10/sample_0001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fdtd2d import RingArray, axis_centers, load_ct_ring_grid, n_steps_for_crossing, stable_dt  # noqa: E402
from fdtd2d.torch_backend import TorchFDTD2D, TorchRingSampler, TorchScalarWave2D  # noqa: E402

# Source: the same Hann-windowed linear chirp as 2DRingFDTD.py, but as a
# continuous function of time so every grid sees an identical pulse.
F0, F1, T_PULSE = 100e3, 250e3, 20e-6
WATER = 1480.0
CFL = 0.45  # 2DRingFDTD.py / invert_ring.py value


def chirp(t):
    t = np.asarray(t, dtype=float)
    inside = (t >= 0) & (t <= T_PULSE)
    k = (F1 - F0) / T_PULSE
    s = 0.5 - 0.5 * np.cos(2 * np.pi * t / T_PULSE)
    out = s * np.sin(2 * np.pi * (F0 * t + 0.5 * k * t**2))
    return np.where(inside, out, 0.0)


class Medium:
    """A sound-speed grid with its axes and ring geometry (fdtd2d conventions: [row=y, col=x])."""

    def __init__(self, c, spacing_m, ring_radius_m):
        self.c = np.asarray(c, dtype=float)
        self.h = float(spacing_m)
        self.ring_radius_m = float(ring_radius_m)
        ny, nx = self.c.shape
        self.x = axis_centers(nx, self.h)
        self.y = axis_centers(ny, self.h)

    def refined(self, factor):
        """Same piecewise-constant medium on a grid `factor` times finer."""
        return Medium(np.kron(self.c, np.ones((factor, factor))), self.h / factor, self.ring_radius_m)

    def padded(self, cells, value=WATER):
        return Medium(np.pad(self.c, cells, constant_values=value), self.h, self.ring_radius_m)

    def water(self):
        return Medium(np.full_like(self.c, WATER), self.h, self.ring_radius_m)


def run_shot(medium: Medium, tx: int, n_elements=256, source="nearest", t_end=None, device="cuda:0",
             dtype=torch.float32, c_for_dt=None):
    """Simulate one shot; returns (time_s, traces[n_elements, n_t], source_xy, ring)."""
    dev = torch.device(device)
    c_max = float(medium.c.max()) if c_for_dt is None else c_for_dt
    dt = stable_dt(c_max, medium.h, medium.h, cfl=CFL)
    if t_end is None:
        t_end = n_steps_for_crossing(medium.ring_radius_m, float(medium.c.min()), dt) * dt
    n_steps = int(math.ceil(t_end / dt))
    model = TorchScalarWave2D(medium.c, medium.h, medium.h, device=dev, dtype=dtype)
    solver = TorchFDTD2D(model, dt, compile_step=False)
    ring = RingArray(n_elements, medium.ring_radius_m, medium.x, medium.y)
    sampler = TorchRingSampler(ring, device=dev, dtype=dtype)
    # forcing f(t_n) = pulse_n / dt^2 enters u_{n+1}; trace column n holds u at (n+1) dt
    pulse = torch.as_tensor(chirp(np.arange(n_steps) * dt), device=dev, dtype=dtype)
    traces = torch.empty((n_elements, n_steps), device=dev, dtype=dtype)
    solver.reset()
    if source == "nearest":
        r, c = ring.inject_rows_cols(tx)
        solver.set_source(r, c)
        src_xy = (medium.x[c], medium.y[r])
        field = None
    else:  # bilinear: spread the point source over the 4 surrounding nodes
        col = (ring.x[tx] - medium.x[0]) / medium.h
        row = (ring.y[tx] - medium.y[0]) / medium.h
        c0, r0 = int(np.floor(col)), int(np.floor(row))
        wx, wy = col - c0, row - r0
        field = torch.zeros(medium.c.shape, device=dev, dtype=dtype)
        for dr, dc, w in ((0, 0, (1 - wx) * (1 - wy)), (0, 1, wx * (1 - wy)), (1, 0, (1 - wx) * wy), (1, 1, wx * wy)):
            field[r0 + dr, c0 + dc] = w
        src_xy = (ring.x[tx], ring.y[tx])
    zero = torch.zeros((), device=dev, dtype=dtype)
    with torch.inference_mode():
        for n in range(n_steps):
            if field is None:
                u = solver.inject_and_step(pulse[n])
            else:
                u = solver._step(zero, field * pulse[n], 2)  # source_mode 2 adds a field to u_next
            traces[:, n] = sampler.record(u)
    time_s = (np.arange(n_steps) + 1) * dt
    return time_s, traces.cpu().numpy().astype(np.float64), src_xy, ring, dt


def analytic_trace(r, t, h, dt_sim, c=WATER, dt_a=2e-9):
    """Exact 2D response at distance r to the solver's point source (u += pulse_n per step).

    The discrete source is the forcing f = s(t) h^2 / dt^2 at a point, and the
    2D Green's function of u_tt - c^2 lap u = delta(x) delta(t) is
    H(t - r/c) / (2 pi c^2 sqrt(t^2 - r^2/c^2)). The singular kernel is
    integrated exactly over each fine time step (arccosh) and convolved with s.
    """
    T = r / c
    ta = np.arange(0, t[-1] + dt_a, dt_a)
    lo = np.maximum(ta, T)
    hi = np.maximum(ta + dt_a, T)
    G = np.arccosh(hi / T) - np.arccosh(lo / T)  # integral of 1/sqrt(t^2-T^2) over each step
    s = chirp(ta + 0.5 * dt_a)
    n = ta.size
    m = 1 << int(np.ceil(np.log2(2 * n)))
    u = np.fft.irfft(np.fft.rfft(s, m) * np.fft.rfft(G, m), m)[:n] / (2 * np.pi * c**2)
    u *= h**2 / dt_sim**2
    return np.interp(t, ta + dt_a, u)


def xcorr_shift(a, b, t, window, upsample_dt=1e-9):
    """Time shift of trace a relative to b (positive = a later), within window (t0, t1)."""
    tt = np.arange(window[0], window[1], upsample_dt)
    A, B = np.interp(tt, t, a), np.interp(tt, t, b)
    A, B = A - A.mean(), B - B.mean()
    cc = np.fft.irfft(np.fft.rfft(A, 2 * tt.size) * np.conj(np.fft.rfft(B, 2 * tt.size)))
    k = int(np.argmax(cc))
    if 0 < k < cc.size - 1:  # parabolic refinement
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    if k > tt.size:
        k -= 2 * tt.size
    return k * upsample_dt


def first_arrival(trace, t, threshold):
    """Sub-sample time where |trace| first exceeds threshold (linear interpolation)."""
    above = np.flatnonzero(np.abs(trace) > threshold)
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return t[0]
    a0, a1 = abs(trace[i - 1]), abs(trace[i])
    return t[i - 1] + (threshold - a0) / (a1 - a0) * (t[i] - t[i - 1])


def on_common_time(t_src, traces, t_ref):
    return np.stack([np.interp(t_ref, t_src, tr, left=0.0, right=0.0) for tr in traces])


# ---------------------------------------------------------------- experiments

def exp_water(base: Medium, out: Path, levels=(1, 2, 4), receivers=range(8, 256, 8)):
    res = {}
    receivers = list(receivers)
    for f in levels:
        for source in (("nearest", "bilinear") if f == 1 else ("bilinear",)):
            m = base.water().refined(f)
            t0 = time.time()
            t, tr, src, ring, dt = run_shot(m, 0, source=source)
            sec = time.time() - t0
            shifts, rel, amp, picks = [], [], [], []
            for j in receivers:
                r = math.hypot(ring.x[j] - src[0], ring.y[j] - src[1])
                ref = analytic_trace(r, t, m.h, dt)
                T = r / WATER
                win = (T - 2e-6, T + T_PULSE + 8e-6)
                shifts.append(xcorr_shift(tr[j], ref, t, win))
                thr = 0.02 * np.abs(ref).max()  # first-arrival pick: cannot skip a cycle
                picks.append(first_arrival(tr[j], t, thr) - first_arrival(ref, t, thr))
                sel = (t > win[0]) & (t < win[1])
                rel.append(np.linalg.norm(tr[j][sel] - ref[sel]) / np.linalg.norm(ref[sel]))
                amp.append(np.abs(tr[j][sel]).max() / np.abs(ref[sel]).max())
            # also: shift relative to the *nominal* element position (what a user assumes)
            nominal = [xcorr_shift(tr[j], analytic_trace(math.hypot(ring.x[j] - ring.x[0], ring.y[j] - ring.y[0]), t, m.h, dt),
                                   t, (math.hypot(ring.x[j] - ring.x[0], ring.y[j] - ring.y[0]) / WATER - 2e-6,
                                       math.hypot(ring.x[j] - ring.x[0], ring.y[j] - ring.y[0]) / WATER + T_PULSE + 8e-6))
                       for j in receivers]
            dist = [math.hypot(ring.x[j] - ring.x[0], ring.y[j] - ring.y[0]) for j in receivers]
            key = f"{f}x_{source}"
            res[key] = {"h_mm": m.h * 1e3, "ppw_250k": WATER / F1 / m.h, "dt_ns": dt * 1e9, "seconds": sec,
                        "distance_mm": [d * 1e3 for d in dist], "shift_ns": [s * 1e9 for s in shifts],
                        "shift_vs_nominal_ns": [s * 1e9 for s in nominal], "rel_l2": rel, "amp_ratio": amp,
                        "pick_err_ns": [x * 1e9 for x in picks],
                        "source_offset_mm": 1e3 * math.hypot(src[0] - ring.x[0], src[1] - ring.y[0])}
            print(f"water {key}: h={m.h*1e3:.3f} mm, PPW@250k={WATER/F1/m.h:.1f}, max |pick err| {np.max(np.abs(picks))*1e9:.0f} ns, "
                  f"max |xcorr shift| {np.max(np.abs(shifts))*1e9:.0f} ns, "
                  f"vs nominal {np.max(np.abs(nominal))*1e9:.0f} ns, median rel L2 {np.median(rel):.3f}, {sec:.0f} s")
    return res


def exp_converge(base: Medium, out: Path, levels=(1, 2, 4)):
    """CT medium at several refinements; errors against the finest level, for tx 0."""
    runs = {}
    t_end = None
    for f in sorted(levels, reverse=True):  # finest first, to fix a common time window
        for source in (("nearest", "bilinear") if f == 1 else ("bilinear",)):
            for kind in ("body", "water"):
                m = (base if kind == "body" else base.water()).refined(f)
                t0 = time.time()
                t, tr, *_ = run_shot(m, 0, source=source, t_end=t_end)
                if t_end is None:
                    t_end = t[-1]
                runs[(f, source, kind)] = (t, tr)
                print(f"converge {f}x {source} {kind}: {time.time() - t0:.0f} s, {tr.shape[1]} steps")
    fine = max(levels)
    t_ref = runs[(fine, "bilinear", "body")][0]
    ref_b = runs[(fine, "bilinear", "body")][1]
    ref_w = on_common_time(*runs[(fine, "bilinear", "water")], t_ref)
    thr = 0.02 * np.abs(ref_w).max(axis=1)
    pick = lambda tr: np.array([first_arrival(x, t_ref, th) for x, th in zip(tr, thr)])  # noqa: E731
    ref_arr_b, ref_arr_w = pick(ref_b), pick(ref_w)
    ref_delay = ref_arr_b - ref_arr_w
    n = ref_b.shape[0]
    valid = np.minimum(np.arange(n), n - np.arange(n)) > 8
    res = {}
    for (f, source, kind) in runs:
        if kind != "body" or (f == fine and source == "bilinear"):
            continue
        b = on_common_time(*runs[(f, source, "body")], t_ref)
        w = on_common_time(*runs[(f, source, "water")], t_ref)
        ab, aw = pick(b), pick(w)
        rel = np.linalg.norm(b - ref_b, axis=1) / np.linalg.norm(ref_b, axis=1)
        key = f"{f}x_{source}"
        res[key] = {
            "arrival_err_ns": ((ab - ref_arr_b) * 1e9).tolist(),
            "delay_err_ns": ((ab - aw - ref_delay) * 1e9).tolist(),
            "trace_rel_l2": rel.tolist(),
            "max_abs_arrival_err_ns": float(np.nanmax(np.abs(ab - ref_arr_b)[valid]) * 1e9),
            "max_abs_delay_err_ns": float(np.nanmax(np.abs(ab - aw - ref_delay)[valid]) * 1e9),
            "median_trace_rel_l2": float(np.median(rel[valid])),
        }
        print(f"converge {key}: max |arrival err| {res[key]['max_abs_arrival_err_ns']:.0f} ns, "
              f"max |delay err| {res[key]['max_abs_delay_err_ns']:.0f} ns, median trace rel L2 {res[key]['median_trace_rel_l2']:.3f}")
    res["reference"] = {"delay_us": (ref_delay * 1e6).tolist(), "valid": valid.tolist(), "level": fine}
    return res


def exp_boundary(base: Medium, out: Path, pad=128):
    t, tr, *_ = run_shot(base, 0, source="bilinear")
    tp, trp, *_ = run_shot(base.padded(pad), 0, source="bilinear", t_end=t[-1])
    trp = on_common_time(tp, trp, t)
    diff = np.linalg.norm(tr - trp, axis=1) / np.linalg.norm(trp, axis=1)
    # where the difference starts: earliest time the difference exceeds 1 % of the trace peak
    onset = []
    for a, b in zip(tr, trp):
        d = np.abs(a - b) > 0.01 * np.abs(b).max()
        onset.append(t[np.argmax(d)] * 1e6 if d.any() else np.nan)
    res = {"rel_l2_median": float(np.median(diff)), "rel_l2_max": float(diff.max()),
           "onset_us_median": float(np.nanmedian(onset)), "margin_cells": 8, "pad_cells": pad,
           "rel_l2": diff.tolist()}
    print(f"boundary: trace difference vs padded domain, median {res['rel_l2_median']:.4f}, max {res['rel_l2_max']:.4f}, "
          f"median onset {res['onset_us_median']:.0f} us")
    return res


def exp_reciprocity(base: Medium, out: Path, txs=(0, 40, 96, 128, 200)):
    res = {}
    for source in ("nearest", "bilinear"):
        shots = {tx: run_shot(base, tx, source=source) for tx in txs}
        t = shots[txs[0]][0]
        pairs, errs, shifts = [], [], []
        for i in txs:
            for j in txs:
                if i < j:
                    a = shots[i][1][j]
                    b = np.interp(t, shots[j][0], shots[j][1][i])
                    errs.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
                    T = np.abs(b).argmax()
                    shifts.append(float(xcorr_shift(a, b, t, (t[max(T - 400, 0)], t[min(T + 400, t.size - 1)])) * 1e9))
                    pairs.append([i, j])
        res[source] = {"pairs": pairs, "rel_l2": errs, "shift_ns": shifts}
        print(f"reciprocity {source}: rel L2 median {np.median(errs):.4f} max {max(errs):.4f}; "
              f"max |shift| {np.max(np.abs(shifts)):.1f} ns")
    return res


def exp_aliasing(base: Medium, out: Path, n_rx=1024):
    res = {}
    for kind, m in (("water", base.water()), ("body", base)):
        t, tr, *_ = run_shot(m, 0, n_elements=n_rx, source="bilinear")
        dt = t[1] - t[0]
        # Mute channels near the transmitter (its near field is a spike in angle that
        # looks like broadband "aliasing"); same exclusion as the delay analysis
        # (8 of 256 elements), with a cosine ramp to avoid leakage.
        off = np.minimum(np.arange(n_rx), n_rx - np.arange(n_rx)) * 256 / n_rx
        taper = np.clip((off - 8) / 8, 0, 1)
        taper = 0.5 - 0.5 * np.cos(np.pi * taper)
        tr = tr * taper[:, None]
        spec = np.fft.rfft(tr, axis=1)  # time -> frequency
        freqs = np.fft.rfftfreq(tr.shape[1], dt)
        ang = np.fft.fft(spec, axis=0)  # element index -> angular harmonic m
        mags = np.abs(ang) ** 2
        m_idx = np.abs(np.fft.fftfreq(n_rx, 1.0 / n_rx))
        band = (freqs >= F0) & (freqs <= F1)
        out_band = {}
        for n_el in (256, 512):
            beyond = m_idx[:, None] > n_el / 2
            frac = mags[:, band][beyond[:, 0]].sum() / mags[:, band].sum()
            per_f = (mags[beyond[:, 0]].sum(axis=0) / np.maximum(mags.sum(axis=0), 1e-300))
            ok = freqs[(per_f < 0.01) & (freqs > 20e3)]
            # highest frequency below which every band frequency keeps <1 % beyond-Nyquist energy
            bad = freqs[(per_f >= 0.01) & (freqs > 20e3) & (freqs <= 400e3)]
            f_clean = float(bad.min()) if bad.size else float("inf")
            out_band[n_el] = {"energy_beyond_nyquist_100_250k": float(frac), "alias_free_up_to_hz": f_clean}
        res[kind] = out_band
        print(f"aliasing {kind}: " + "; ".join(f"{k} el: {v['energy_beyond_nyquist_100_250k']*100:.1f} % beyond Nyquist in band, "
                                               f"clean below {v['alias_free_up_to_hz']/1e3:.0f} kHz" for k, v in out_band.items()))
        np.savez_compressed(out / f"aliasing_{kind}.npz", freqs=freqs, m=np.fft.fftfreq(n_rx, 1.0 / n_rx),
                            power=mags[:, freqs <= 400e3].astype(np.float32), radius_m=m.ring_radius_m)
    return res


def exp_traveltime(base: Medium, out: Path, run: Path = None, map_name="acoustic_itis.npz", fine=4):
    """Travel-time body speed (as in Phase 1) at the default grid vs a `fine`-times finer grid."""
    sys.path.insert(0, str(ROOT / "src"))
    from make_simulation_figures import body_path_mm

    res = {}
    for s in sorted(run.glob("sample_*")):
        if not (s / map_name).exists():
            continue
        g = load_ct_ring_grid(s / map_name, grid_shape=(512, 512))
        b0 = Medium(g.c, g.spacing_m, g.ring_radius_m)
        row = {}
        for f in (1, fine):
            mb, mw = b0.refined(f), b0.water().refined(f)
            t, tb, src, ring, _ = run_shot(mb, 0, source="bilinear")
            tw, tww, *_ = run_shot(mw, 0, source="bilinear", t_end=t[-1])
            tww = on_common_time(tw, tww, t)
            thr = 0.02 * np.abs(tww).max(axis=1)
            ab = np.array([first_arrival(x, t, th) for x, th in zip(tb, thr)])
            aw = np.array([first_arrival(x, t, th) for x, th in zip(tww, thr)])
            n = len(ring.x)
            valid = np.minimum(np.arange(n), n - np.arange(n)) > 8
            xy = np.stack([ring.x, ring.y], axis=1)
            path = body_path_mm(g, xy[0], xy)
            sel = valid & (path > 20) & np.isfinite(ab - aw)
            L = path[sel].sum() * 1e-3
            row[f"{f}x"] = {"travel_speed": float(L / (L / WATER + np.sum((ab - aw)[sel]))),
                            "opposite_delay_us": float((ab - aw)[n // 2] * 1e6),
                            "delay_us": ((ab - aw) * 1e6).tolist()}
        row["true_mean"] = float(np.load(s / map_name)["speed_m_s"][np.load(s / map_name)["body_mask"]].mean())
        d = np.array(row["1x"]["delay_us"]) - np.array(row[f"{fine}x"]["delay_us"])
        row["max_abs_delay_err_us"] = float(np.nanmax(np.abs(d[valid])))
        res[s.name] = row
        print(f"traveltime {s.name}: speed 1x {row['1x']['travel_speed']:.1f} vs {fine}x {row[f'{fine}x']['travel_speed']:.1f} "
              f"(true mean {row['true_mean']:.1f}); max |delay err| {row['max_abs_delay_err_us']:.2f} us")
    return res


EXPERIMENTS = {"water": exp_water, "converge": exp_converge, "boundary": exp_boundary,
               "reciprocity": exp_reciprocity, "aliasing": exp_aliasing, "traveltime": exp_traveltime}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--map", default="acoustic_itis.npz")
    parser.add_argument("--grid", type=int, default=512, help="base grid (2DRingFDTD default)")
    parser.add_argument("--experiments", nargs="+", default=[e for e in EXPERIMENTS if e != "traveltime"],
                        choices=list(EXPERIMENTS))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "generated" / "phase2")
    parser.add_argument("--run", type=Path, help="run directory for the traveltime experiment (all samples)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4],
                        help="refinement factors for the converge experiment (finest = reference)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    g = load_ct_ring_grid(args.sample / args.map, grid_shape=(args.grid, args.grid))
    base = Medium(g.c, g.spacing_m, g.ring_radius_m)
    print(f"base: {args.grid}^2, h = {g.spacing_m*1e3:.3f} mm, ring R = {g.ring_radius_m*1e3:.1f} mm, "
          f"c {g.c.min():.0f}-{g.c.max():.0f} m/s")
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["geometry"] = {"sample": str(args.sample), "map": args.map, "grid": args.grid,
                           "h_mm": g.spacing_m * 1e3, "ring_radius_mm": g.ring_radius_m * 1e3}
    for name in args.experiments:
        t0 = time.time()
        kwargs = {"run": args.run or args.sample.parent} if name == "traveltime" else {}
        if name == "converge":
            kwargs = {"levels": tuple(args.levels)}
        results[name] = EXPERIMENTS[name](base, args.out, **kwargs)
        results[name]["wall_seconds"] = time.time() - t0
        results_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
