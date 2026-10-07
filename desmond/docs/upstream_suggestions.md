# Suggested fixes for the shared code (for the advisor)

*Proposed 2026-10-07 by Desmond. **None of these are applied**: the files in the repository root are unchanged. Each fix is small and independent, so you can take any subset.*

## 1. `--device auto` / `--device cuda` fails under torch 2.14

**Symptom.** Every GPU script stops with `Could not configure PyTorch: Expected a torch.device with a specified index or an integer, but got:cuda`. Passing `--device cuda:0` works around it.

**Cause.** `resolve_device("auto")` returns `torch.device("cuda")` with no index. `configure_runtime()` then calls `torch.cuda.set_device(device)`, which newer torch rejects for an index-less device.

**Fix**, in `fdtd2d/torch_backend.py`, `resolve_device`:

```diff
     if requested == "auto":
         if torch.cuda.is_available():
-            return torch.device("cuda")
+            return torch.device("cuda", torch.cuda.current_device())
```

## 2. Inversion scripts overwrite committed PNGs in the repository root

**Symptom.** Running `invert_ring.py --no-show` (or `invert_arc.py`, `fdfd_ring/invert_ring_fdfd.py`) without `--save-figure` writes `invert_ring_dashboard.png` (or `invert_arc_dashboard.png`, `fdfd_ring_dashboard.png`) into the current directory. That overwrites the committed result images when run from the root.

**Locations:**

- `invert_ring.py`, around line 1093
- `invert_arc.py`, around line 737
- the `--save-figure` default in `fdfd_ring/invert_ring_fdfd.py`, line 124

**Fix (one option):** default to a gitignored `outputs/` directory instead of the working directory, e.g.

```diff
-        figure_path = Path("invert_ring_dashboard.png")
+        figure_path = Path("outputs") / "invert_ring_dashboard.png"
+        figure_path.parent.mkdir(exist_ok=True)
```

and add `outputs/` to `.gitignore`.

## 3. Flaky test: `tests/test_neural.py::test_speed_is_bounded_and_background_outside_mask`

**Symptom.** Fails in about 2 of 5 identical runs. Every pixel saturates at `C_MIN` = 1400 m/s, so `c.max() > 1900` fails.

**Cause.** `SoundSpeedNet.__init__` (`fdtd2d/inversion/neural.py`) seeds the Fourier frequencies and zeroes the output layer, but the hidden `nn.Linear` layers keep PyTorch's default initialization, which draws from the **global, unseeded** RNG. The test's `make_model(seed=0)` therefore gets a different hidden network each run, and sometimes every output has the same sign.

**Fix (either):**

- Initialize the hidden layers from the seeded generator inside `SoundSpeedNet.__init__`. This also makes `--nn-seed` fully reproducible for the inversion:

    ```python
    for layer in self.hidden:
        if isinstance(layer, nn.Linear):
            bound = 1.0 / math.sqrt(layer.in_features)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)
    ```

- Or call `torch.manual_seed(seed)` at the start of `make_model` in the test (fixes only the test).

## 4. Points per wavelength and element spacing (for discussion, not a code bug)

At the 250 kHz top of the chirp, the current grids give 4–7 grid points per wavelength in fat. A second-order FDTD scheme usually wants 10 or more. The 256-element ring's pitch is 1.25–1.6× half a wavelength, i.e. spatially aliased. Details and the planned convergence test: `desmond/docs/critical_review.md`, section 5 and Phase 2.
