# MAISI (NV-Generate-CTMR) model notes

Notes on generating synthetic abdominal CT with the MAISI rectified-flow CT model
(`rflow-ct`) to feed this repo's ultrasound simulations. The model repo is
read-only: never edit files in it or write outputs into it.

| | |
|---|---|
| Model repo | `/sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR` |
| Commit read | `da438fec6484cdb6f421f8c7051d954ebefff730` ("Update inference.md") |
| Upstream docs | `README.md`, `docs/inference.md`, `docs/performance.md`, `skills/*.md` in the model repo |
| Python env | conda `soundwave` from `desmond/environment.yml` (Python 3.12, torch 2.14.1+cu130, monai 1.6.1). Run `conda activate soundwave` before every command below |
| GPU | NVIDIA RTX A6000, 48 GB |

## TL;DR

- **The model is 3D only. It has no 2D mode.** The smallest valid CT volume is
  256×256×128. To get a landmark slice, generate a volume and extract the slice.
- **Use the paired mask + image pipeline for abdomen** via `desmond/src/generate_slices.py` (section 7). Retrieved masks (Path B) are about 15× faster per accepted L3 slice than generated masks (Path A).
  The image-only pipeline (`scripts.diff_model_infer`) has no anatomy or
  body-region control. A test run at the recommended chest+abdomen FOV came out
  mostly thorax (heart and lungs). The paired pipeline also writes a 132-class
  label map that includes individual vertebrae (`vertebrae L3` = label 35), so a
  landmark slice can be found from the labels instead of guessed.
- Output is a NIfTI in **RAS** orientation, **int16 HU clipped to [-1000, 1000]**,
  with spacing on the affine diagonal.
- About **20 GB of GPU memory** for a 512×512×128 volume. Takes about 45 s per
  volume on the A6000, including weight loading.

## 1. Checkpoints

The `rflow-ct` weights are already downloaded to `<model repo>/models/` (untracked
in that repo). Source: HuggingFace `nvidia/NV-Generate-CT`, NVIDIA Open Model
License.

| File | Size | Role | Used by |
|---|---:|---|---|
| `autoencoder_v1.pt` | 84 MB | Image VAE (encode/decode latent ↔ HU) | all CT pipelines |
| `diff_unet_3d_rflow-ct.pt` | 2.2 GB | Image latent diffusion UNet (rectified flow, 30 steps) | image-only, paired |
| `controlnet_3d_rflow-ct.pt` | 288 MB | ControlNet: conditions image on a label mask | paired, image-from-mask |
| `mask_generation_autoencoder.pt` | 21 MB | Mask VAE | paired (Path A) |
| `mask_generation_diffusion_unet.pt` | 789 MB | Mask diffusion UNet (DDPM, 1000 steps) | paired (Path A) |
| `autoencoder_v2.pt`, `diff_unet_3d_rflow-mr.pt` | — | MR model (`rflow-mr`), non-commercial license | not used here |

**Not downloaded yet:** the `datasets/` auxiliary data that the paired pipeline
needs:

- `datasets/all_anatomy_size_conditions.json`: needed for Path A (masks generated
  by diffusion).
- `datasets/all_masks_flexible_size_and_spacing_4000.zip` and
  `candidate_masks_flexible_size_and_spacing_4000.json`: needed for Path B
  (masks retrieved from the training set). This is several GB.

Other variants: `ddpm-ct` (MAISI-v1; 1000 steps, about 33× slower; takes explicit
body-region one-hot inputs), `rflow-mr`, `rflow-mr-brain`. None of them is needed
for abdominal CT.

Download command (run in the model repo; it skips files that already exist):

```bash
cd /sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR
python -m scripts.download_model_data --version rflow-ct --root_dir "./" --model_only   # weights only
python -m scripts.download_model_data --version rflow-ct --root_dir "./"                # + datasets/ (writes into the model repo!)
```

> The second command writes `datasets/` into the model repo. To keep the model
> repo clean, point `--root_dir` at a cache directory outside it. Paths
> containing `datasets/` resolve against `root_dir`.

## 2. Pipelines and how to run them

All entry points are modules (`python -m scripts.X`), so the working directory
must be the model repo root. Two things in a run would write into the model repo,
and both need to be redirected:

1. Shipped environment configs use relative output paths (`./output`, `output`).
   Copy the config elsewhere and set an absolute `output_dir`.
2. Importing `scripts.*` writes `scripts/__pycache__`. Set
   `PYTHONDONTWRITEBYTECODE=1`.

### 2a. Image only: `scripts.diff_model_infer` (verified)

Noise → image latent diffusion (30 RFlow steps) → VAE decode with a fixed
80×80×80 sliding window. No mask and no anatomical control.

This exact sequence was run on 2026-10-05. It finished in 47 s, exited 0, and
left the model repo unchanged:

```bash
M=/sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR
OUT=/path/outside/model/repo           # e.g. this repo's data/ directory
mkdir -p "$OUT"

# 1) Copy configs out of the model repo, with absolute paths and the abdomen FOV
python - <<EOF
import json
M, OUT = "$M", "$OUT"
env = json.load(open(f"{M}/configs/environment_maisi_diff_model_rflow-ct.json"))
env.update(model_dir=f"{M}/models", output_dir=f"{OUT}/maisi_raw",
           trained_autoencoder_path=f"{M}/models/autoencoder_v1.pt",
           existing_ckpt_filepath=f"{M}/models/diff_unet_3d_rflow-ct.pt",
           modality_mapping_path=f"{M}/configs/modality_mapping.json")
json.dump(env, open(f"{OUT}/env_diff_rflow-ct.json", "w"), indent=2)
cfg = json.load(open(f"{M}/configs/config_maisi_diff_model_rflow-ct.json"))
cfg["diffusion_unet_inference"].update(dim=[512, 512, 128],
                                       spacing=[0.781, 0.781, 2.981],
                                       random_seed=0)
json.dump(cfg, open(f"{OUT}/cfg_diff_rflow-ct.json", "w"), indent=2)
EOF

# 2) Run
cd "$M" && PYTHONDONTWRITEBYTECODE=1 python -m scripts.diff_model_infer \
    -t ./configs/config_network_rflow.json \
    -e "$OUT/env_diff_rflow-ct.json" \
    -c "$OUT/cfg_diff_rflow-ct.json"
```

Upstream form (README §2.4). It writes to `<model repo>/output/` and changes
settings by editing configs in place, so **don't use it as-is**:

```bash
python -m scripts.diff_model_infer -t ./configs/config_network_rflow.json \
    -e ./configs/environment_maisi_diff_model_rflow-ct.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json
```

Seeding: `diffusion_unet_inference.random_seed` (+ GPU rank) is passed to
`monai.utils.set_determinism`. There is no CLI seed flag, so the config is the
only place to set it.

### 2b. Paired mask + image: `scripts.inference` (recommended for abdomen)

> In this repo, use `desmond/src/generate_slices.py` (section 7) instead of this CLI. It calls the same
> `LDMSampler` methods in-process, but keeps the vertebra labels that the CLI filters out.

1. Mask stage. It takes one of two paths:
   - **Path A** (`controllable_anatomy_size` non-empty): the mask diffusion model
     generates a new 132-class mask. It generates at a fixed 256³ × 1.5 mm and
     then resamples to `output_size`/`spacing`.
   - **Path B** (`controllable_anatomy_size == []`): a real training mask matching
     `body_region` + `anatomy_list` is retrieved and lightly augmented.
2. Image stage. ControlNet + image diffusion (30 steps) conditioned on the mask,
   then VAE decode with a configurable sliding window and tensor-parallel splits.
3. Quality check. The median HU of each major organ is checked against
   `configs/image_median_statistics_ct.json`. A failing pair is regenerated up to
   2 times.

```bash
M=/sci-it/projects/sarang-lab/desmond/nv-maisi/NV-Generate-CTMR
export MONAI_DATA_DIRECTORY=/path/outside/model/repo/maisi_cache   # where datasets/ lives
cd "$M" && PYTHONDONTWRITEBYTECODE=1 python -m scripts.inference \
    -t ./configs/config_network_rflow.json \
    -i /path/to/copied/config_infer.json \
    -e /path/to/copied/environment_rflow-ct.json \
    --random-seed 0 --version rflow-ct
```

Before running, set absolute paths in the copied `environment_rflow-ct.json`:
`output_dir`, every `trained_*_path`, `label_dict_json`, and
`label_dict_remap_json`. `scripts.inference` calls `download_model_data()` first.
That call fetches anything missing into `MONAI_DATA_DIRECTORY` (`datasets/`) and
into `models/` relative to the working directory, and skips files that exist.

### 2c. Image from your own mask: `scripts.infer_image_from_mask`

This runs ControlNet only, on a mask you supply. The mask must use the MAISI
132-class vocabulary plus body envelope label 200. That's useful if we later want
to edit masks, for example to change organ sizes in a controlled way.

```bash
python -m scripts.infer_image_from_mask -t ./configs/config_network_rflow.json \
    -i ./configs/config_infer.json -e ./configs/environment_rflow-ct.json \
    --mask /path/to/mask.nii.gz
```

## 3. Inputs and conditioning

| Input | Image-only (`config_maisi_diff_model_rflow-ct.json` → `diffusion_unet_inference`) | Paired (`config_infer.json`) |
|---|---|---|
| Volume size | `dim` | `output_size` |
| Voxel spacing (mm) | `spacing`, which is also fed to the UNet as a conditioning tensor | `spacing` |
| Modality | `modality` = 1 (CT) | `modality` = 1 |
| Body region | Ignored by `rflow-ct` (`top/bottom_region_index` only matter for `ddpm-ct`) | `body_region`, used for the Path B mask lookup |
| Anatomy | none | `anatomy_list`: organs that must be present (Path B) and the organs kept in the saved label map |
| Organ/tumor size | none | `controllable_anatomy_size`: up to 10 `[name, size]` pairs; triggers Path A |
| Steps | `num_inference_steps` = 30 | `num_inference_steps` = 30; `mask_generation_num_inference_steps` = 1000 |
| CFG | `cfg_guidance_scale` = 0 for CT (modality CFG) | `cfg_guidance_scale`: tumor CFG, 0 = off |
| Seed | `random_seed` in config | `--random-seed` CLI flag |
| Memory knobs | none (fixed 80³ window, overlap 0.4) | `autoencoder_sliding_window_infer_size`, `_overlap`, `autoencoder_tp_num_splits` |

Controllable anatomy-size keys (Path A): `gallbladder, liver, stomach, pancreas,
colon, lung tumor, pancreatic tumor, hepatic tumor, colon cancer primaries,
bone lesion`.

**Hard constraints** (`check_input_ct`):

- `dim[0] == dim[1] ∈ {256, 384, 512}`
- `dim[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1] ∈ [0.5, 3.0]` mm
- `spacing[2] ∈ [0.5, 5.0]` mm

**FOV (= dim × spacing) is the main quality knob.** Stay near the training
medians (`docs/inference.md`). The rows relevant here:

| Training body region | Share of training data | `output_size` | `spacing` (mm) | FOV (mm) |
|---|---:|---|---|---|
| chest + abdomen | 58.6 % | 512×512×128 | 0.781, 0.781, 2.981 | 400 × 400 × 382 |
| abdomen + lower | 0.4 % | 512×512×384 | 0.808, 0.808, 0.729 | 414 × 414 × 280 |
| abdomen only | 0.1 % | 512×512×128 | 0.723, 0.723, 1.182 | 370 × 370 × 151 |
| (generic) | — | 256×256×256 | 1.5, 1.5, 1.5 | 384³ |

Out-of-distribution FOVs pass validation but produce unusable images. An example
is a 128 mm cube.

## 4. Output format

Verified on the image-only run above (seed 0, 512×512×128 at 0.781/0.781/2.981 mm):

| Property | Value |
|---|---|
| File | NIfTI `.nii.gz`. Image-only: `unet_3d_seed<seed>_size<X>x<Y>x<Z>_spacing<sx>x<sy>x<sz>_<timestamp>_rank<r>_modality1.nii.gz`. Paired: `sample_<timestamp>_image.nii.gz` + `sample_<timestamp>_label.nii.gz` |
| Shape | Exactly `dim`/`output_size`, as (X, Y, Z) = (512, 512, 128) |
| dtype | `int16` |
| Spacing | Affine diagonal = `spacing`. The image-only affine is `diag(sx, sy, sz, 1)` with zero origin and no direction cosines |
| Orientation | `RAS`: axis 0 = patient Right (+x), axis 1 = Anterior (+y), axis 2 = Superior (+z). `vol[:, :, k]` is an axial slice. Display it with `imshow(vol[:, :, k].T, origin="lower")` for anterior-up. The paired pipeline also forces RAS (`Orientationd(axcodes="RAS")` on the mask) |
| HU range | `[-1000, 1000]`, hard-clipped. The decoder output in [0, 1] is mapped linearly to [-1000, 1000]. In the test volume, 22 % of voxels were at -1000 (air or background) and 0.01 % at +1000. **Cortical bone and metal saturate at 1000 HU** |
| Labels (paired only) | `int` label map in the MAISI 132-class vocabulary (`configs/label_dict.json`), filtered to `anatomy_list`. Useful IDs: liver 1, spleen 3, pancreas 4, aorta 6, `vertebrae L5…L1` = 33…37, `T12` = 38, `S1` = 127, body envelope 200 |

The test volume also showed a CT table (couch) and a mostly thoracic FOV. Expect
generated scans to include a couch, so don't treat every non-air pixel as body.
`ct_to_speed.py` already keeps the largest connected component.

## 5. GPU memory and time

The peak happens during the VAE decode. Upstream figures for the paired pipeline
on an A100-80G (`docs/performance.md`), with `rflow` timing:

| `output_size` | Peak memory | VAE + DM time | Preset config |
|---|---:|---:|---|
| 256×256×128 | 15.0 GB | 3 s | `config_infer_16g_256x256x128.json` |
| 256×256×256 | 15.4 GB | 8 s | `config_infer_16g_256x256x256.json` |
| 512×512×128 | 15.7 GB | 13 s | `config_infer_16g_512x512x128.json` |
| 512×512×128 | 21.0 GB | 11 s | `config_infer_24g_512x512x128.json` |
| 512×512×512 | 22.8 GB | 48 s | `config_infer_24g_512x512x512.json` |
| 512×512×512 | 45.3 GB | 51 s | `config_infer_80g_512x512x512.json` |
| 512×512×768 | 49.7 GB | 87 s | `config_infer_80g_512x512x768.json` |

These figures leave out model loading and, for Path A, the 1000-step mask
diffusion.

**Measured here** (A6000, image-only `diff_model_infer`, 512×512×128): **about
19.3 GB** peak GPU memory, from `nvidia-smi` sampled at 2 Hz, minus 0.6 GB
baseline. Wall time was 47 s, including loading the 2.2 GB UNet and the AE. Host
RSS was 2.0 GB. The 48 GB A6000 can run any of the presets above, including
512×512×512.

Out-of-memory knobs (paired pipeline only): raise `autoencoder_tp_num_splits`
(∈ {1, 2, 4, 8, 16}; small effect on quality) or shrink
`autoencoder_sliding_window_infer_size` (must be divisible by 16; can cause seams).

## 6. Hand-off to this repo's simulators

`ct_to_speed.py` takes a 2D HU slice (`.npy` + `--pixel-spacing-mm`). It produces
the `.npz` (`speed_m_s`, `body_mask`, `spacing_mm`) that `--ct-speed` consumes in
`2DRingFDTD.py`, `invert_ring.py`, `invert_arc.py`, and `fdfd_ring/*`. An extracted
MAISI slice `vol[:, :, k]` has in-plane spacing `(sx, sy)` and goes in as
`--pixel-spacing-mm sx` (in-plane spacing is always isotropic for CT). The 1000 HU
clip only affects the bone/metal tail, and `ct_to_speed`'s bone class starts at
200 HU, so the clipping does not change tissue classification.

## 7. Generating landmark slices in this repo: `desmond/src/generate_slices.py`

`desmond/src/generate_slices.py` drives the paired pipeline in-process and saves mid-L3
slices. Settings live in `desmond/configs/slices.yaml`. `generation.mask_source` (or
`--mask-source`) picks where the anatomy mask comes from:

| | Path A: `generated` | Path B: `retrieved` |
|---|---|---|
| Mask | Mask diffusion model (DDPM, 1000 steps) conditioned on a 10-d anatomy-size vector from `all_anatomy_size_conditions.json`, with tumor slots set to −1 | Real training mask from `candidate_masks_flexible_size_and_spacing_4000.json` (MAISI `find_masks` with `body_region: [abdomen]`, L3 present, tumor-free), resampled, plus MAISI's body augmentation (±1 % zoom only) |
| Upstream equivalent | `controllable_anatomy_size` non-empty | `controllable_anatomy_size: []` |
| Extra data | 0.3 MB JSON | 14 GB unpacked under `desmond/data/maisi_cache/datasets/` (the 11.6 GB zip was deleted after unpacking) |
| Per-sample variety | New mask each seed | One mask per sample, from a pool of the closest-FOV masks shuffled with `base_seed`. A retry keeps the mask and redraws the zoom and image noise |

```bash
conda activate soundwave
python desmond/src/generate_slices.py --config configs/slices.yaml                     # Path B (default), 10 volumes
python desmond/src/generate_slices.py --mask-source generated --run-name l3_A_n10      # Path A
python desmond/src/generate_slices.py --num-volumes 50 --run-name l3_n50 --no-volumes --max-attempts 20
python desmond/src/compare_runs.py desmond/data/generated/<runA> desmond/data/generated/<runB> --names "Path A" "Path B" \
    --out desmond/data/generated/compare_A_vs_B
python -m pytest tests/test_generate_slices.py -m "not slow"                   # CPU unit tests
python -m pytest tests/test_generate_slices.py -m slow                         # 1-volume GPU smoke test per path
```

Each run writes `desmond/data/generated/<run_name>/` (gitignored). It contains
`config.yaml`, `manifest.csv`, and one `sample_XXXX/` per volume:

- `L3_mid_hu.npy`: int16, 512×512, radiological orientation
- `L3_mid_label.npy`: MAISI labels for the same slice
- `meta.json`: seed, model commit, slice index, spacing, mask source/condition,
  rejected attempts, and so on
- `volume_*.nii.gz`: full volumes, unless `--no-volumes`
- `preview.png`

Re-running the same command skips finished samples.

**Rejection rules.** A sample is redrawn (seed `base_seed + i + 100000·attempt`)
in any of these cases:

- **L3 touches the first or last slice of the labelled field of view.** The
  check uses the labelled FOV, not the array. Resampling pads empty slices,
  which otherwise hide a truncated vertebra.
- **The body in the L3 slice touches the in-plane edge of the source mask's
  FOV.** That FOV is 384 mm for the mask model, or `dim × spacing` for a
  retrieved mask. Large patients get cut with straight flanks.
- **A tumor label appears in the slice.**
- **MAISI's organ-HU quality check fails.**

The first three depend only on the mask, so they are checked before the image
stage, and Path B's pool is pre-filtered with them.

### Path A vs Path B (10 samples each, 2026-10-06, A6000)

**Decision: Path B (retrieved masks) is the default** (`mask_source: retrieved`). The Path A samples were deleted after the comparison; `desmond/data/generated/compare_A_vs_B/` keeps the record.

Full table: `desmond/data/generated/compare_A_vs_B/summary.md`. Figures: `montage.png`
and `metrics.png`.

| | Path A (generated) | Path B (retrieved) |
|---|---|---|
| Generation attempts for 10 samples | 32 (22 rejected: 12 L3 truncated in z, 6 body cut in-plane, 1 L3 missing, 3 QC) | 11 (1 QC); 17 of the closest-FOV masks dropped while building the pool |
| GPU sampling time per accepted sample | **≈ 4.7 min** (about 80 s mask diffusion + 15 s CT per attempt) | **≈ 0.3 min** |
| L3 position in the volume (z of 128) | 13–87, spread across the FOV | 72–91; the retrieved scans are abdomen-centred |
| Body area (cm²), median [range] | 498 [384–852] | 542 [397–645] |
| SAT / VAT / muscle fraction (median) | 0.21 / 0.13 / 0.14 | 0.25 / 0.17 / 0.09 |
| Kidney HU (median of samples) | 152; enhancing cortex > 200 HU in most | 96; mostly non-enhanced or venous |
| Bright non-bone pixels > 200 HU (become 3476 m/s) | 1.7 % of body | 1.1 % of body |
| Mean soft-tissue speed (hybrid) | 1523 m/s | 1511 m/s |

Takeaways:

- **Path B is about 15× cheaper per accepted slice.** Its anatomy comes from
  real abdominal scans (AMOS, KiTS-like cases, and others) with complete L3 and
  natural body outlines. The trade-off is that each mask is a real patient's
  segmentation (only ±1 % zoom), so slice diversity is bounded by the
  closest-FOV pool. The pool held 22 masks here, and N ≫ 22 needs a larger
  pool. Only the CT texture is synthesized.
- **Path A gives fully synthetic anatomy and a new mask per seed,** but the
  mask model's fixed, chest-centred 384 mm cube cuts L3 or the flanks about
  2 times in 3. That adds selection bias against large bodies; the largest
  accepted body was 852 cm², versus 1110 cm² before the in-plane check. Path A
  also tends to produce contrast-enhanced kidneys.
- **Both paths produce contrast-enhanced CT.** Oral contrast in bowel and
  enhancing kidneys above 200 HU are mapped to bone speed by `ct_to_speed.py`.
  Consider remapping them using `L3_mid_label.npy`.

Other observations:

- **Reproducible.** The same seed gives bit-identical slices across runs
  (`set_determinism` per seed).
- **Peak GPU memory about 18 GB** for 512×512×128.
- **Tumor-free anatomy conditions (Path A).** Only 65 of the 1,429 anatomy-size
  entries are tumor-free. Each sample therefore draws one of the 1,207 entries
  with all five organs present and sets the tumor slots to −1. In
  `sample_mask.py`, −1 in a tumor slot means no target tumor.
- **`2DRingFDTD.py --device`.** In the `soundwave` env (torch 2.14) the default
  device fails with "Expected a torch.device with a specified index". Pass
  `--device cuda:0`. This is an issue in the existing script, not in the slices.

End-to-end check that was run on a generated slice:

```bash
D=desmond/data/generated/<run>/sample_0000
python ct_to_speed.py $D/L3_mid_hu.npy --pixel-spacing-mm 0.781 --mapping hybrid \
    --air-as-water --output $D/acoustic.npz --figure $D/acoustic.png
python 2DRingFDTD.py --ct-speed $D/acoustic.npz --device cuda:0 --no-show --save-figure $D/ring_fdtd.png
```

The FDTD run used a 512×512 grid and 5315 steps and took 8 s.
