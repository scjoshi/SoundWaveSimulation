# Decision log

*Newest first. Each entry: what was decided, why, and what would reopen it.*

## 2026-10-07: Phase 1 sound-speed modelling choices

| Decision | Why | Reopen if |
| --- | --- | --- |
| Tissue speeds and densities come from the **IT'IS Tissue Properties Database** (`desmond/configs/itis_tissue_properties.csv`, file dated 2024-06-04), not Culjat 2010 | Largest curated source, with mean, SD and number of studies per tissue | The advisor prefers another source |
| Each tissue is **anchored to its IT'IS mean**, with texture = 1.0 m/s per HU around the tissue's median HU, clipped to ±1 SD | Synthetic CT is always contrast-enhanced; anchoring at the median removes the contrast offset but keeps within-organ variation | Real non-contrast CT becomes available (then HU → speed regression is defensible again) |
| Fat = IT'IS "Fat" 1440 ± 22 m/s (16 studies) for both SAT and VAT | IT'IS also lists "SAT" at 1477 from only 2 studies; Culjat (used by `ct_to_speed.py`) gives 1478 | A dedicated SAT source with more studies |
| Bowel (stomach, duodenum, small bowel, colon) = IT'IS "lumen" 1535 m/s | IT'IS has **no measured** wall speed for these organs (placeholder 1500), and the label mostly covers contents | Better bowel data; contents vary with meal and contrast |
| Bone graded by HU: ≤100 HU red marrow 1450 → 200 HU cancellous 2118 → 1000 HU cortical 3515 (speed and density interpolated linearly) | The L3 body is mostly cancellous (median 181–242 HU); one cortical class put 3476 m/s on 40–66 % of it | Bone-specific measurements; note MAISI clips HU at 1000 |
| Pixels ≥ 200 HU **outside** bone labels = soft tissue (contrast or calcification), not bone | They are oral or IV contrast in these images | Non-contrast CT |
| Gas default = water (1482 m/s); options `lumen` and `air` | The constant-density solver cannot represent gas reflections, and 343 m/s gives about 1 grid point per wavelength | A variable-density solver (Phase 2+) |
| HU denoised with a 1 mm Gaussian **within each tissue only**; classification smoothing stays inside the body | Removes 8–27 HU synthetic noise without blurring boundaries. Smoothing across the skin made a 1-pixel layer look like gas (bug found by a unit test) | — |
| Density map kept in every output | Needed for a future variable-density solver | — |

## 2026-10-07: Phase 0, goal and baseline geometry

**Goal.** Per the advisor, the near-term study is **a single slice of abdominal CT**. For each synthetic patient (MAISI `rflow-ct` with retrieved real masks), we take the mid-L3 slice and build a physically defensible sound-speed map. We then simulate 2D ultrasound transmission through it with a ring (and arcs), and measure how well travel-time analysis and full-waveform inversion recover that map, especially fat versus lean tissue.

**Out of scope for now:** the TFNO training dataset in `TFNO_Full_Ring_Plan.tex`, multi-slice data, and 3D.

**Baseline geometry (frozen as the reference; Phase 2 may revise the band or grid):**

| Item | Value | Source |
| --- | --- | --- |
| Dimension | 2D, constant-density scalar wave | `fdtd2d/wave.py` |
| Forward grid | 512 × 512; spacing = max(CT pixel 0.781 mm, body fit), which is 0.781 mm for all 10 slices | `fdtd2d/ct_medium.py` |
| Ring | 256 elements; radius = body radius + 10 mm clearance (150–190 mm here) | advisor default (`--ring-clearance-mm 10`) |
| Source | Hann-windowed linear chirp 100 → 250 kHz, 20 µs | `2DRingFDTD.py` constants |
| Coupling medium | Water 1480 m/s | `--ct-padding-speed` default |
| Sound-speed map | Phase 1 IT'IS-anchored map, gas as water (`acoustic_itis.npz`) | `desmond/src/speed_map.py` |

**Open questions for the advisor:**

1. Body-adaptive ring radius (current) or one fixed radius for every patient, like a real scanner? Fixed makes patients directly comparable.
2. Is 100–250 kHz with 256 elements intended? It gives 6.7–7.3 grid points per wavelength and a pitch of 1.25–1.57× half a wavelength at 250 kHz (Phase 2 will quantify the error).
3. Are the three upstream fixes in `upstream_suggestions.md` welcome?
