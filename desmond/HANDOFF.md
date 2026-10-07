# Handoff: where this work stands

*Last updated 2026-10-07. Read this first when resuming. Update it at the end of every working session: state, next steps, open questions.*

## Goal

A **single-slice abdominal CT study** (the advisor's direction). Synthetic mid-L3 CT slices with known sound speed are simulated with the shared 2D ring/arc ultrasound code. We measure how well travel times and full-waveform inversion (FWI) recover the sound-speed map, especially fat versus lean tissue.

**Out of scope for now:** the TFNO dataset (`TFNO_Full_Ring_Plan.tex`), multi-slice data, 3D.

## Status

| Phase | State | Where |
| --- | --- | --- |
| Synthetic CT (MAISI `rflow-ct`, retrieved masks) | Done; 10 slices shared in `data/ct_slices_l3_v1/` | `docs/simulation_pipeline.md`, `docs/model_notes.md` |
| 0: goal and baseline | Done; the advisor's answers are pending (below) | `docs/decisions.md` |
| 1: realistic sound-speed maps | Done | `docs/phase1_speed_maps.md` |
| 2: simulator accuracy | Done; configuration proposed | `docs/phase2_numerics.md` |
| 3: one inversion that works | **Next** | `docs/critical_review.md` §7 |
| 4–5: scale up, studies | Not started | `docs/critical_review.md` §7 |

## Key results so far

- **Sound-speed maps.** The shared `ct_to_speed.py` map is fine for fat, muscle and liver, but off for the contrast-enhanced kidney (+72 m/s), blood (+27), bowel contents (+44 to +64) and the spine (3476 vs about 2118 m/s). The corrected maps (`acoustic_itis.npz`) put every tissue within IT'IS mean ± 1 SD.
- **Travel times track fat.** Ring travel-time body speed correlates with fat share at r = −0.97 (10 slices). It runs about +7 m/s fast from fastest-path physics, plus +5 to +8 m/s on the default grid.
- **The default grid is too coarse.** At about 7 points per wavelength, body delays are off by a median 3 % (worst 16 %) and waveforms by about 110 %. A 4× grid gives 0.7 % (worst 2.4 %).
- **Inversions with default settings barely move.** The Adam step caps each pixel at about 1.6 m/s per iteration. The water start is cycle-skipped (up to 22 µs vs a 20 µs half period). Bone is capped at 2000 m/s.
  - Ring FWI improves soft tissue by 21 % (24 % with 10× step); arc FWI gets worse.

## Open questions for the advisor (blocking Phase 3 choices)

1. A fixed ring radius for all patients, or body-adaptive (current)?
2. Band and elements: 512 elements, or cap the band near 150–190 kHz? (256 elements alias above that.)
3. Grids: 4× for simulated data, 2× for inversion. Acceptable cost (about 11 s per shot at 4×)?
4. Will he take the upstream fixes in `docs/upstream_suggestions.md` (device bug, default PNG paths, flaky test seeding, bilinear source, boundary)?

## Next steps (Phase 3, single slice: sample 1)

1. Full 256-transmitter data on the 4× grid with a bilinear source; invert on 2× (avoids the inverse crime).
2. Travel-time tomography starting model from first arrivals (straight-ray first).
3. FWI from that start: sane step size or L-BFGS, more shots per gradient, bounds that contain bone (or bone fixed), light regularization.
4. Report error by tissue. Target: soft-tissue RMS under 25 m/s (start: 71 m/s).

## Where things are

| What | Path |
| --- | --- |
| Shareable dataset (10 slices, tracked) | `data/ct_slices_l3_v1/` |
| Full runs, volumes, traces, inversion outputs (gitignored) | `data/generated/pathB_retrieved_n10/`, `data/generated/phase2/` |
| MAISI mask cache, 14 GB (gitignored) | `data/maisi_cache/` |
| MAISI model repo (read-only) | `/sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR` (commit da438fe) |
| Code | `src/` (see `README.md` for each script) |
| Tests | `tests/`: `python -m pytest desmond/tests -m "not slow"` (28 pass) |

## Gotchas

- **Run from the repo root** in `conda activate soundwave`. Never edit the advisor's files.
- **Advisor scripts:** pass `--device cuda:0`, and always `--save-figure <path under desmond/>`; otherwise they overwrite committed PNGs in the root.
- **`conda run` buffers output** until exit. Use `conda run --no-capture-output` to stream logs.
- **Peak time ≠ travel time.** Pick first arrivals.
- **Interior RMS hides soft tissue** (bone dominates it). Report soft-tissue or per-tissue error.
- **Water references** need an identical geometry. Fill the same `.npz` with 1480 m/s (see `docs/simulation_pipeline.md` §3).
- **The advisor's `tests/test_neural.py` has a flaky test** (unseeded hidden layers), not caused by our changes.
- **Commit messages:** keep them short (subject plus a few lines).
