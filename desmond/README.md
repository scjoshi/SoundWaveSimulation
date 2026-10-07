# desmond/: student workspace

Desmond's code, configs, data and notes for this repository. Everything here builds on the advisor's code in the repository root **without modifying it**: the root scripts are called as they are, or imported read-only.

## Start here

| Document | What it is for |
| --- | --- |
| [docs/learning_guide.md](docs/learning_guide.md) | The topic from the ground up: ultrasound tomography, wave simulation, FWI, the TFNO plan, a map of the codebase, hands-on exercises |
| [docs/critical_review.md](docs/critical_review.md) | What has gone wrong so far (CT, mapping, simulator, inversion, my own pipeline) and the phased plan forward |
| [docs/simulation_pipeline.md](docs/simulation_pipeline.md) | How the synthetic CT samples were made and simulated, with interpretation figures |
| [docs/model_notes.md](docs/model_notes.md) | Reference for the MAISI (NV-Generate-CTMR) CT generation model |
| [docs/phase1_speed_maps.md](docs/phase1_speed_maps.md) | Phase 1: IT'IS-anchored sound-speed maps, validation, and their effect on simulations |
| [docs/decisions.md](docs/decisions.md) | Decision log: goal, baseline geometry, modelling choices, and why |
| [docs/upstream_suggestions.md](docs/upstream_suggestions.md) | Small fixes proposed for the advisor's shared code (not applied) |

## Layout

```text
desmond/
├── README.md
├── environment.yml            conda env "soundwave" (also runs the advisor's scripts)
├── configs/
│   ├── slices.yaml            synthetic-CT generation settings
│   ├── tissue_map.yaml        Phase 1 anatomy -> tissue rules (editable)
│   └── itis_tissue_properties.csv   IT'IS tissue speeds/densities (with provenance)
├── src/
│   ├── generate_slices.py     MAISI rflow-ct -> 3D CT -> mid-L3 slice (+ labels, metadata)
│   ├── compare_runs.py        compare generation runs (e.g. generated vs retrieved masks)
│   ├── capture_inversion.py   run the advisor's inversion scripts unchanged and save their maps
│   ├── make_simulation_figures.py   all figures/metrics in docs/simulation_pipeline.md
│   ├── speed_map.py           Phase 1: CT slice + labels -> IT'IS-anchored speed/density maps
│   └── validate_speed_maps.py Phase 1: per-tissue validation + forward-effect figures
├── tests/                     unit tests + GPU smoke tests for src/
├── docs/                      the documents above; figures/ holds their images
└── data/                      gitignored: generated samples, simulation outputs, MAISI mask cache (14 GB)
```

## Conventions

- **Run from the repository root:** `conda activate soundwave`, then `python desmond/src/...` or the advisor's scripts (`python 2DRingFDTD.py ...`).
- **Outputs go under `desmond/data/`.** When running the advisor's scripts, always pass `--save-figure desmond/data/...`; without it they overwrite the committed PNGs in the root.
- **GPU device:** pass `--device cuda:0` to the advisor's scripts; the default `auto` fails under torch 2.14.
- **The MAISI model repo** (`/sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR`) is read-only.
- **Tests:**
  - `python -m pytest desmond/tests -m "not slow"` (CPU, seconds)
  - `python -m pytest desmond/tests -m slow` (GPU, minutes)
  - `python -m pytest tests` (the advisor's tests)

## Reproducing the current results

```bash
conda env create -f desmond/environment.yml && conda activate soundwave
python desmond/src/generate_slices.py --run-name pathB_retrieved_n10      # 10 synthetic L3 slices (~4 min)
# then follow docs/simulation_pipeline.md sections 2-3 (ct_to_speed, forward runs, inversions)
python desmond/src/make_simulation_figures.py --run desmond/data/generated/pathB_retrieved_n10 --inversion-sample 1
```
