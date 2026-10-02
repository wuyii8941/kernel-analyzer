#!/usr/bin/env python3
"""Capture the Liger cross-entropy kernel with the exp implementation replaced (liger env).

Variants of every ``tl.exp(.)`` inside ``liger_cross_entropy_kernel``:

* ``original``  : tl.exp (control: must reproduce the unmodified kernel bitwise)
* ``exp2_rn``   : tl.exp2(x * RN32(log2 e))
* ``exp2_up``   : tl.exp2(x * nextafter32(RN32(log2 e), +inf))   (constant above log2 e)
* ``libdevice`` : libdevice.exp (compensated expf)
* ``div_rn``    : tl.exp kept, both divisions (by d and by n_non_ignore) correctly rounded
* ``libdevice_div_rn`` : libdevice.exp and correctly rounded divisions

The fused linear cross-entropy workload of capture_liger_kernels.py is run
with the same seed for each variant, so every variant's kernel receives the
same logits; this is checked against the original variant's captured inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402

LOG2E_RN = float(np.float32(1.4426950408889634))
LOG2E_UP = float(np.nextafter(np.float32(LOG2E_RN), np.float32(np.inf)))
HELPERS = {
    "original": "@triton.jit\ndef _EXP(x):\n    return tl.exp(x)\n",
    "exp2_rn": f"@triton.jit\ndef _EXP(x):\n    return tl.exp2(x * {LOG2E_RN!r})\n",
    "exp2_up": f"@triton.jit\ndef _EXP(x):\n    return tl.exp2(x * {LOG2E_UP!r})\n",
    "libdevice": "from triton.language.extra.cuda import libdevice as _libdevice\n"
                 "@triton.jit\ndef _EXP(x):\n    return _libdevice.exp(x)\n",
}
HELPERS["div_rn"] = HELPERS["original"]
HELPERS["libdevice_div_rn"] = HELPERS["libdevice"]
# Variants that also replace the two divisions by correctly rounded division.
DIVISIONS = {
    "                X_block = _EXP(X_block - m) / d\n":
        "                X_block = tl.div_rn(_EXP(X_block - m), d)\n",
    "                    X_block = X_block / n_non_ignore\n":
        "                    X_block = tl.div_rn(X_block, n_non_ignore.to(tl.float32))\n",
}


def patched_kernel(variant: str, workdir: Path):
    import liger_kernel.ops.cross_entropy as ce

    source = inspect.getsource(ce)
    start = source.index("def liger_cross_entropy_kernel(")
    end = source.index("\n@triton.jit", start) if "\n@triton.jit" in source[start:] else len(source)
    body = source[start:end]
    if body.count("tl.exp(") < 3:
        raise RuntimeError("expected exp calls in the kernel were not found")
    body = body.replace("tl.exp(", "_EXP(")
    if variant.endswith("div_rn"):
        for old, new in DIVISIONS.items():
            if body.count(old) != 1:
                raise RuntimeError(f"division site not found: {old.strip()}")
            body = body.replace(old, new)
    patched = source[:start] + body + source[end:]
    marker = "@triton.jit\ndef liger_cross_entropy_kernel("
    patched = patched.replace(marker, HELPERS[variant] + "\n\n" + marker, 1)
    path = workdir / f"liger_ce_{variant}.py"
    path.write_text(patched)
    spec = importlib.util.spec_from_file_location(f"liger_ce_{variant}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.liger_cross_entropy_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "patched_sources"
    workdir.mkdir(exist_ok=True)
    original_kernel = fused.liger_cross_entropy_kernel
    manifest = {"log2e_rn": LOG2E_RN, "log2e_up": LOG2E_UP, "variants": {}}
    reference_inputs = None
    for variant in HELPERS:
        fused.liger_cross_entropy_kernel = patched_kernel(variant, workdir)
        torch.manual_seed(20261002)
        V, H, BT = 32000, 256, 64
        lin = torch.nn.Linear(H, V, bias=False, device="cuda")
        h = torch.randn(BT, H, device="cuda", requires_grad=True)
        target = torch.randint(0, V, (BT,), device="cuda")
        target[5] = -100
        recorder = TritonLaunchRecorder(select=lambda name, i: name == "liger_cross_entropy_kernel")
        with recorder:
            LigerFusedLinearCrossEntropyLoss()(lin.weight, h, target).backward()
            torch.cuda.synchronize()
        inputs = [next(a for a in L.args if a.name == "X_ptr").before for L in recorder.launches]
        if reference_inputs is None:
            reference_inputs = inputs
        same = all(torch.equal(a, b) for a, b in zip(inputs, reference_inputs))
        for j, launch in enumerate(recorder.launches):
            save_launch(launch, args.out / variant / f"launch{j:03d}")
        manifest["variants"][variant] = {"launches": len(recorder.launches), "same_logits_as_first_variant": same}
        print(variant, len(recorder.launches), "same inputs:", same, flush=True)
    fused.liger_cross_entropy_kernel = original_kernel
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
