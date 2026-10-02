#!/usr/bin/env python3
"""Capture Liger RMSNorm forward with single-site source mutations (liger env).

Variants (one change each, same inputs):

* ``original``  : unmodified kernel source
* ``rerun``     : unmodified kernel, captured a second time (negative control)
* ``rsqrt_rn``  : rstd = 1 / sqrt_rn(ms + eps) instead of rsqrt(ms + eps)
* ``div_rn``    : mean square divided by n_cols with correctly rounded division
* ``omit_eps``  : rstd = rsqrt(ms)  (declared semantics changes; needs a specification to detect)
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402

SITES = {
    "rsqrt_rn": ("    rstd = rsqrt(mean_square + eps)\n", "    rstd = 1.0 / tl.sqrt_rn(mean_square + eps)\n"),
    "div_rn": ("    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols\n",
               "    mean_square = tl.div_rn(tl.sum(X_row * X_row, axis=0), n_cols.to(tl.float32))\n"),
    "omit_eps": ("    rstd = rsqrt(mean_square + eps)\n", "    rstd = rsqrt(mean_square)\n"),
}


def patched_module(variant: str, workdir: Path):
    import liger_kernel.ops.rms_norm as rms

    if variant in ("original", "rerun"):
        return rms
    source = inspect.getsource(rms)
    start = source.index("def _rms_norm_forward_kernel(")
    end = source.index("\n@triton.jit", start)
    body = source[start:end]
    old, new = SITES[variant]
    if body.count(old) != 1:
        raise RuntimeError(f"mutation site not found: {old.strip()}")
    patched = source[:start] + body.replace(old, new) + source[end:]
    path = workdir / f"liger_rms_{variant}.py"
    path.write_text(patched)
    spec = importlib.util.spec_from_file_location(f"liger_rms_{variant}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import liger_kernel.ops.rms_norm as rms

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "patched_sources"
    workdir.mkdir(exist_ok=True)
    original_forward = rms._rms_norm_forward_kernel
    manifest = {"variants": {}}
    for variant in ("original", "rerun", "rsqrt_rn", "div_rn", "omit_eps"):
        rms._rms_norm_forward_kernel = patched_module(variant, workdir)._rms_norm_forward_kernel
        torch.manual_seed(20261003)
        x = torch.randn(64, 512, device="cuda") * 2.0
        w = 1.0 + 0.1 * torch.randn(512, device="cuda")
        recorder = TritonLaunchRecorder(select=lambda name, i: name == "_rms_norm_forward_kernel")
        with recorder:
            rms.rms_norm_forward(x, w, 1e-6, 0.0, "llama", True)
            torch.cuda.synchronize()
        save_launch(recorder.launches[-1], args.out / variant / "launch000")
        manifest["variants"][variant] = {"launches": len(recorder.launches)}
        print(variant, len(recorder.launches), flush=True)
    rms._rms_norm_forward_kernel = original_forward
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
