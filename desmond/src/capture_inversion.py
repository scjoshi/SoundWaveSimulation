"""Run an existing inversion script unchanged and save its maps to an .npz.

invert_ring.py, invert_arc.py and fdfd_ring/invert_ring_fdfd.py only save a
dashboard image. This wrapper loads the script as a module, intercepts the
function each one already calls with the true and reconstructed maps
(print_summary for the time-domain scripts, make_figure for the
frequency-domain one), saves those arrays, and then lets the original
function run. The scripts themselves are not modified.

    python desmond/src/capture_inversion.py --out result.npz -- invert_ring.py \\
        --ct-speed acoustic.npz --multiscale --device cuda:0 --no-show \\
        --save-figure result.png

Always pass --save-figure: without it the scripts write their default
dashboard PNG into the repository root, overwriting the committed one.

The archive holds c_true, c_est, mask (interior pixels the inversion updates)
and misfit (the objective per iteration).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repository root (advisor's scripts, fdtd2d/)


def load_script(path: Path):
    for directory in (ROOT, path.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f"captured_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it executes
    spec.loader.exec_module(module)
    return module


def _misfit(history):
    if isinstance(history, dict):
        return np.asarray(history.get("misfit", history.get("J", [])), dtype=float)
    return np.asarray([h if np.isscalar(h) else h[0] for h in history], dtype=float)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: capture_inversion.py --out OUT.npz -- SCRIPT.py [script args]")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv[:split])
    script, script_args = Path(argv[split + 1]).resolve(), argv[split + 2 :]

    module = load_script(script)
    captured = {}

    def save(c_true, c_est, mask, history):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            c_true=np.asarray(c_true, dtype=np.float32),
            c_est=np.asarray(c_est, dtype=np.float32),
            mask=np.asarray(mask, dtype=bool),
            misfit=_misfit(history),
            script=str(script.name),
            argv=np.array(script_args),
        )
        print(f"Captured inversion arrays: {args.out}")

    if hasattr(module, "print_summary"):  # invert_ring.py, invert_arc.py
        original = module.print_summary

        def print_summary(c_true, c_est, mask, history, *rest, **kwargs):
            save(c_true, c_est, mask, history)
            return original(c_true, c_est, mask, history, *rest, **kwargs)

        module.print_summary = print_summary
    elif hasattr(module, "make_figure"):  # fdfd_ring/invert_ring_fdfd.py
        original_mask, original_figure = module.interior_mask, module.make_figure

        def interior_mask(*a, **k):
            captured["mask"] = original_mask(*a, **k)
            return captured["mask"]

        def make_figure(c_true, c_est, history, *rest, **kwargs):
            mask = captured["mask"]
            save(c_true, c_est, mask.cpu().numpy() if hasattr(mask, "cpu") else mask, history)
            return original_figure(c_true, c_est, history, *rest, **kwargs)

        module.interior_mask, module.make_figure = interior_mask, make_figure
    else:
        raise SystemExit(f"{script.name}: no print_summary or make_figure to intercept")

    sys.argv = [str(script), *script_args]
    module.main()


if __name__ == "__main__":
    main()
