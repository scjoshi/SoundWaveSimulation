# Critical review: what went wrong, why, and how to move forward

*As of 2026-10-06. An independent assessment of the work so far: my own pipeline, the advisor's simulation code, and the synthetic CT. No one's choices are assumed correct. Every claim below cites a number measured in this repo or a line in its code.*

## Summary

**Nothing so far is broken beyond repair, but nothing is ready to answer a research question.** The four layers each have problems, and their severities differ:

| Layer | Verdict | Most serious issue |
| --- | --- | --- |
| Inversion settings (advisor's FWI) | **Main cause of the poor reconstructions** | The step size caps each pixel's change at about 1.6 m/s per iteration; beyond that, the water starting model is cycle-skipped and the bounds exclude bone |
| Forward simulator (advisor's `fdtd2d`) | Usable, with numerical limits to verify | 4–7 grid points per wavelength at the top of the band; element spacing exceeds half a wavelength at 250 kHz |
| CT → sound-speed mapping (advisor's `ct_to_speed.py`) | Biased in predictable ways | Contrast-enhanced organs too fast, spongy vertebral bone gets dense-bone speed, noisy fat tail too slow, gas becomes water (fixed in Phase 1) |
| Synthetic CT (MAISI) | Fit for purpose, with known biases | Always contrast-enhanced, with 8–27 HU pixel noise; anatomy is limited to real training masks |
| My pipeline and process | Mostly fixed; one strategic misalignment | Built around an L3-only study, while the codebase's real goal (a learned full-ring reconstructor) needs fixed-geometry, full-acquisition, diverse data |

The single most useful next step is to **settle the goal with the advisor** (section 1). The second is to **make one inversion actually work** on one slice (Phase 3 below), because every downstream comparison depends on it.

## 1. The strategic issue: what is this codebase for?

The advisor's `TFNO_Full_Ring_Plan.tex` states the target. The goal is to train a **Tensorized Fourier Neural Operator** that maps full-ring ultrasound data (256 transmitters × 256 receivers × time) to a 512×512 sound-speed image. It builds on Dai, Penwarden, Kirby & Joshi (MIDL 2023), and adjoint FWI is the baseline to beat. The plan calls for:

- **One fixed geometry** for every example: a 512×512 grid at about 1 mm, a ring radius of about 230 mm, and 256 elements.
- **Full acquisitions:** all 256 transmitters per example.
- **At least 1000 diverse examples,** with train/test splits by patient.

Our work so far diverges from that in three ways:

1. **L3 only.** I built an L3-landmark pipeline because that was your request. For training data, L3 is one slice level among many. Diversity across slice levels (e.g., L1–S1 and other abdominal levels) matters more, while L3 stays useful as a body-composition evaluation set.
2. **Body-adaptive geometry.** The existing CT loader (`fdtd2d/ct_medium.py`) rescales each body into its own ring: radius 150–190 mm and grid spacing 0.78–0.99 mm in our runs. The plan explicitly forbids this, because the network must learn a single forward operator.
3. **One transmitter per slice** in the forward runs, versus 256 in the plan.

**Decision needed from the advisor:** is the immediate goal (a) the TFNO dataset, (b) an L3 body-composition study, or (c) both? The plan below covers both, but the order changes.

## 2. What went wrong in my own work

I'm listing these first because they are the easiest to verify.

| Problem | Effect | Status |
| --- | --- | --- |
| L3 truncation check used the array edge, not the labelled field of view | 5 of 10 first-round Path B slices had a partial L3 (6–12 mm instead of about 50 mm) | Fixed; regenerated |
| No in-plane truncation check | 3 of 10 Path A bodies were cut flat at the 384 mm mask edge | Fixed; regenerated |
| Reported *peak* time as travel time | Claimed waves arrive 0–8 % late; first arrivals actually come 9–22 µs *early* | Corrected in docs |
| Reported interior RMS including bone | Concluded "no inversion helps"; on soft tissue the ring actually improved 21 % | Corrected; soft-tissue RMS added |
| Ran inversion scripts without `--save-figure` | Overwrote the advisor's committed `invert_ring_dashboard.png` | Restored from git; commands fixed |
| One transmitter per slice; one slice inverted; 10 samples | Findings are illustrative, not statistics | Open (Phases 3–4) |

The pattern is that my checks tested what I expected rather than what could go wrong. Each one was found by looking at the data: previews, extents, montages. Lesson for this project: **always look at the picture before trusting a number.**

## 3. The synthetic CT (MAISI `rflow-ct`, Path B)

The CT generator is broadly fit for purpose. Its anatomy comes from real segmentation masks, the CT texture passed visual review, and every slice carries per-pixel ground truth. Its limitations are inherent to the model and its training data:

1. **Always contrast-enhanced.** MAISI's training data is dominated by contrast CT, and `rflow-ct` accepts only modality 1 ("CT") with no non-contrast option. Our slices show oral contrast in bowel and enhancing kidneys (median kidney HU 96–152 across samples). The HU → speed mapping assumes non-contrast tissue, so this propagates directly into speed errors (section 4).
2. **Pixel noise of 8–27 HU** (high-pass standard deviation inside liver and fat). At 1.0–1.15 m/s per HU, this becomes 8–31 m/s of random speed texture per pixel. It is not anatomy, but the simulator treats it as real scatterers.
3. **HU clipped to −1000…1000.** Harmless here because bone speed is set by class, but HU above 1000 is unrecoverable.
4. **Limited, non-independent anatomy.** Each sample is one of about 1865 eligible real masks, from the same public datasets MAISI was trained on. A model trained on these may learn MAISI's anatomy distribution, not the population's. Train/test splits must go by source mask; `meta.json` records it.
5. **Selection bias by body size.** Masks whose original field of view cuts the body are rejected, which removes the largest patients.
6. **No quantitative HU validation.** MAISI's quality check only bounds organ *median* HU, so texture statistics are unvalidated.

**Verdict:** acceptable for method development and TFNO pilots, provided contrast and noise are handled in the speed mapping. For clinical claims, validate against real non-contrast CT.

## 4. The CT → sound-speed mapping (`ct_to_speed.py`, `hybrid`)

This is the advisor's tool, and it is transparent about being "a simulation-preprocessing tool, not a clinical segmentation system". Measured on our slices, its biases are:

| Tissue | Mapping gives | Typically quoted | Why |
| --- | --- | --- | --- |
| Fat at −100 HU | 1418 m/s; 16–68 % of fat pixels below 1430 m/s; 5th percentile 1362–1407. **Correction (Phase 1):** the pooled fat median is 1436 m/s, inside IT'IS 1440 ± 22. The bias is in the noisy tail, not the typical fat pixel | about 1430–1480 m/s (IT'IS 1440 ± 22) | Noise widens the regression's low tail down to 1315 m/s |
| Water-equivalent tissue at 0 HU | 1524 m/s | 1480 (20 °C bath) to about 1524 (37 °C) | Regression calibrated near body temperature, while the coupling bath is 1480 m/s; fine if deliberate |
| Contrast-enhanced kidney (96–152 HU) | about 1620–1680 m/s | Non-enhanced kidney is near soft-tissue values | Contrast raises HU, and the mapping reads that as density |
| Spongy (trabecular) vertebral body | 40–66 % of L3 pixels are ≥ 200 HU and become **3476 m/s** | Cancellous bone is far slower than cortical | One bone class at cortical speed; median L3 HU is only 181–242 |
| Bowel gas | 1480 m/s (water), via `--air-as-water` | Gas reflects almost all ultrasound | Keeps the simulation stable, but hides the main real-world obstacle |
| Density | Computed, then discarded | — | The solver is constant-density (section 5) |

**Verdict:** the mapping is reasonable as a first pass, but two biases are large enough to matter, plus one tail: enhanced organs too fast (kidney +72 m/s, blood +27, bowel contents +44 to +64 against IT'IS), spine too fast and too large, and a noisy low tail in fat (the fat median itself is fine; see Phase 1). All three are correctable in a wrapper, without touching the advisor's file (Phase 1).

## 5. The forward simulator (`fdtd2d`, `2DRingFDTD.py`)

The physics model is deliberately simple: the 2D constant-density scalar wave equation u_tt = c²∇²u, leapfrog time stepping, and a first-order Mur absorbing boundary. The code is clean and has gradient checks. Its numerical limits need verifying before the data are trusted:

1. **Points per wavelength (PPW).** At each band's top frequency, using fat speed:

    | Grid | Top frequency | PPW |
    | --- | --- | --- |
    | 101 | 75 kHz | 3.9–4.2 |
    | 201 | 125 kHz | 5.1–5.5 |
    | 401 | 250 kHz | 5.3–5.8 |
    | 512 (forward) | 250 kHz | 6.7–7.3 |

    A second-order scheme usually needs 10 or more PPW to keep phase errors small over paths of about 60 wavelengths. The inverse crime (data and inversion on the same grid) hides this error completely. Under the advisor's planned fixed geometry (about 1 mm spacing, 250 kHz), PPW would be about 5.7.
2. **Element spacing versus wavelength.** 256 elements on a 150–190 mm ring gives a 3.7–4.7 mm pitch. That is 1.25–1.58× half a wavelength at 250 kHz, so the upper band is spatially aliased at the receivers. The planned 230 mm ring raises this to about 1.9×.
3. **Physics omitted:**
    - density, so reflections at bone and gas interfaces are wrong;
    - attenuation, so amplitudes are qualitative;
    - shear waves in bone;
    - the third dimension: 2D line sources spread cylindrically.

    These are acceptable for a methods study but limit realism.
4. **Body-adaptive geometry** (section 1) conflicts with the TFNO plan.
5. **Three code hazards:**
    - `resolve_device("auto")` returns `torch.device("cuda")` without an index, which `torch.cuda.set_device` rejects under torch 2.14. A one-line fix: use `torch.device("cuda", torch.cuda.current_device())`.
    - The scripts write `*_dashboard.png` into the repo root when `--save-figure` is omitted, overwriting committed results.
    - `tests/test_neural.py::test_speed_is_bounded_and_background_outside_mask` is flaky: it failed in 2 of 5 identical runs, with every pixel saturating at 1400 m/s. The model is built with `seed=0`, but `SoundSpeedNet`'s hidden `nn.Linear` layers keep PyTorch's default initialization from the global, unseeded RNG (only the Fourier features and output layer are seeded). Fix in `upstream_suggestions.md`.

**Verdict:** fine for qualitative work today. Before it generates training data, it needs a grid-convergence test, a reciprocity test (already an acceptance criterion in the plan), and a choice of band and element count that avoids aliasing.

## 6. The inversions (`invert_ring.py`, `invert_arc.py`, FDFD)

This is where most of the poor results come from, and the causes are **settings, not fundamental limits**:

1. **The step size caps how far a pixel can move.** Adam's update is lr × m̂ / √v̂, so each pixel changes by at most about `lr` per iteration (`invert_ring.py`, line 654). With the default `--adam-lr 1e-9` in slowness² units, that is about **1.6 m/s per iteration** near 1480 m/s.
    - **Arithmetic:** dm/dc = −2/c³ = 6.2e-10 s²/m² per (m/s).
    - **Cap:** three stages × 15 iterations allow at most about ±70 m/s.
    - **Observed:** the reconstructions span 1413–1557 m/s, about ±70 m/s around water. The cap, not the physics, set that range.
    - **Needed:** organs are 100–150 m/s faster than water and bone about 2000 m/s faster.
2. **Cycle skipping from the water start.** First arrivals through the body lead water by up to 22 µs (`delay_profiles.png`). Half a period is 20 µs at 25 kHz, the lowest frequency of the first multiscale stage, and 5 µs at 100 kHz. A homogeneous starting model is therefore off by more than half a cycle on the longest paths even in the first stage. FWI then converges to the wrong cycle. This is the textbook FWI failure; the standard remedy is a travel-time tomography starting model.
3. **Bounds exclude the truth.** `--c-min 1400 --c-max 2000` cuts the fat tail (down to 1315 m/s) and all bone (3476 m/s). Bone alone puts a 211 m/s floor under interior RMS.
4. **Noisy, short optimization.** 4 random transmitters per gradient, 15 iterations per stage, no preconditioning, and `--reg 0`. The ring misfit oscillates (`inversion_misfit.png`).
5. **The arc is ill-posed.** Two 30° arcs leave most directions unmeasured (a "missing wedge"). Our arc run fit its data best (misfit −57 %) while its soft-tissue error *rose* 9 %, and it produced vertical streaks. Limited-angle reconstruction needs priors or regularization; without them a lower misfit is not a better image.
6. **The inverse crime flatters every method.** The same grid makes and inverts the data, with no noise.
7. **The reported metric misleads.** The scripts print interior RMS, which bone dominates. The plan already asks for tissue-specific errors.

**Step-size test.** A ring inversion with `--adam-lr 1e-8` (10× the default, everything else unchanged) confirms item 1 and shows what limits the result next:

| Ring FWI, sample 1, `--multiscale` | lr 1e-9 (default) | lr 1e-8 |
| --- | --- | --- |
| Reconstructed range | 1417–1557 m/s | 1400–2000 m/s (fills the bounds) |
| Interior RMS (incl. bone) | 293 m/s | **245 m/s** (−16 %) |
| Soft-tissue RMS (start 71.3 m/s) | 56.5 m/s (−21 %) | 54.0 m/s (−24 %) |
| Final misfit / start | 0.83 | 0.77 |

With the larger step, the vertebra and some organ boundaries appear (`desmond/data/generated/pathB_retrieved_n10/sample_0001/inversions/ring_lr1e-8.png`). Soft tissue barely improves, though, and an over-shoot halo forms around the spine. So the step size explains why the image could not move, but not why soft tissue stays inaccurate. The next suspects, in order:

1. the cycle-skipped water start (item 2);
2. bone clipped at 2000 m/s, whose unexplained data leak into nearby soft tissue (item 3);
3. noisy 4-transmitter gradients (item 4).

Phase 3 tests them one at a time.

## 7. Plan to move forward

The phases are ordered so that each one de-risks the next. Each lists its exit criterion.

### Phase 0: Align and organize (this week)

- [x] **Goal decided (2026-10-07): a single-slice abdominal CT study** (`decisions.md`). Still to confirm with the advisor: fixed vs body-adaptive ring, and the band and element count, given the PPW and aliasing numbers above.
- [ ] **Push `desmond/` on a branch.** It's already separated from the advisor's code.
- [x] **Upstream fixes written up** in `upstream_suggestions.md` (CUDA device index, safe default output paths, the flaky test's seeding, plus the PPW/aliasing discussion). None applied; leave that to the advisor.
- **Exit:** a written one-paragraph goal and a frozen geometry. **Done 2026-10-07:** goal and baseline geometry in `decisions.md`; pushing the branch is up to Desmond.

### Phase 1: Make the sound-speed maps physically defensible (`desmond/src/`, wrapping `ct_to_speed.py`)

- [x] **Denoise HU** before mapping, e.g. an edge-preserving filter at about 1 mm, to remove the 8–27 HU synthetic noise.
- [x] **Correct contrast with labels.** Map enhanced organs (kidney, liver, spleen, vessels, bowel contents) to non-contrast HU or literature speeds.
- [x] **Split bone.** Cortical bone stays near 3500 m/s; trabecular bone gets a lower, HU-graded speed.
- [x] **Floor or recalibrate fat** to literature values (about 1430–1480 m/s), and document the source.
- [x] **Gas options:** water (current), or a strongly reflecting or absorbing treatment. Run both to bound the effect.
- [x] **Keep the density map** for a future variable-density solver.
- **Exit:** a per-tissue speed table that matches the literature within stated tolerances, with a figure. **Done 2026-10-07:** see `phase1_speed_maps.md`.

### Phase 2: Verify the forward simulator

- [ ] **Grid convergence.** Run the same medium at h, h/2 and h/4. Measure first-arrival time and phase error against PPW, then choose the grid and band.
- [ ] **Reciprocity.** Check d_ij(t) ≈ d_ji(t), an acceptance criterion in the advisor's plan.
- [ ] **Aliasing.** Compare 256 vs 512 elements at the chosen band.
- [ ] **Boundary reflections.** Measure them at the chosen margin.
- **Exit:** a numerical error budget, e.g. arrival-time error below 1 % of the 9–22 µs body signal.

### Phase 3: One inversion that works (single slice, sample 1)

- [ ] **Full data:** 256 transmitters (about 41 s at 401², from the inversion precompute timings).
- [ ] **Avoid the inverse crime:** make the data on a finer grid than the inversion, and add noise.
- [ ] **Travel-time tomography starting model** from first arrivals (we already showed they are accurate), straight-ray first, bent-ray next.
- [ ] **FWI from that start:**
    - a sane step size or L-BFGS with line search (`invert_ring_fdfd.py` already has projected L-BFGS);
    - more transmitters per gradient;
    - bounds that contain the truth, or bone fixed as known;
    - light regularization;
    - frequency continuation from the lowest band.
- [ ] **Report per-tissue error** (fat, muscle, organs, bone), as the advisor's plan specifies.
- **Exit:** soft-tissue RMS clearly below the 71 m/s starting error (target under 25 m/s) on the non-inverse-crime data.

### Phase 4: Data at scale

- [ ] If the goal is TFNO: follow the advisor's dataset contract (fixed geometry, full 256 × 256 × T traces, chunked storage, patient-level splits by mask file). Sizes: 16 → 250 → 1000 examples. At about 1 min per full acquisition, 1000 examples take roughly 17 GPU-hours and about 67 GB of float16 traces (per the plan).
- [ ] Extend slice selection beyond L3 (several abdominal levels per volume) for diversity. Keep L3 as a tagged evaluation subset.

### Phase 5: Studies

- [ ] **Ring vs arc** with the working FWI from Phase 3, and arc with regularization or priors.
- [ ] **Travel time vs body composition** at N ≥ 50: does ring travel time predict fat fraction? Here r = −0.80 at N = 10.
- [ ] **TFNO vs FWI vs a homogeneous-water baseline**, as in the advisor's plan.

## 8. What I'd tell the advisor in one minute

1. The FWI default step size caps each pixel at about 1.6 m/s per iteration, which explains the flat reconstructions. 10× the step brings out the spine (interior RMS −16 %), but soft tissue stays at about 54 m/s RMS. The water start (cycle-skipped by up to 22 µs) and the 2000 m/s bone cap are the next limits. All are fixable.
2. The CT → speed mapping biases fat low, contrast organs high and the spine high. That's fixable in preprocessing.
3. At 250 kHz the current grids give 4–7 points per wavelength and the ring is spatially aliased. Please confirm the band, grid and element count before generating the TFNO dataset.
4. MAISI with real-mask conditioning is a good source of diverse, labelled synthetic anatomy. It's always contrast-enhanced, which we now correct for.
