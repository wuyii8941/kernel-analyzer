#!/usr/bin/env python3
"""Inspect the retained Inductor softmax backward without using eager output as ground truth."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_triton_code


ROOT = Path(__file__).resolve().parents[3]
CALLS = ROOT / "5_specification/corpus/calls"
sys.path.insert(0, str(CALLS))
import inductor_softmax_backward as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    saved_output = candidate.make_saved_output(0, (candidate.ROWS, candidate.COLS), "float32")
    generator = torch.Generator(device="cpu").manual_seed(20261019)
    upstream = torch.randn(candidate.ROWS, candidate.COLS, generator=generator).cuda()
    before = {"saved_output": saved_output.clone(), "upstream": upstream.clone()}
    generated = run_and_get_triton_code(candidate._compiled, upstream, saved_output)
    actual = candidate.call({
        "upstream": upstream, "saved_output": saved_output,
        "dim": candidate.DIM, "input_dtype": candidate.INPUT_DTYPE,
    })["dx"]
    eager = candidate._backward(upstream, saved_output)
    kernel_names = sorted(set(re.findall(r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE)))
    report = {
        "task": "PyTorch saved-output softmax backward",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "dx": {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "finite": bool(torch.isfinite(actual).all()),
            "max_abs_compiled_minus_eager": float((actual - eager).abs().max().cpu()),
        },
        "inputs_unchanged": torch.equal(saved_output, before["saved_output"]) and torch.equal(upstream, before["upstream"]),
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0 and report["inputs_unchanged"] and report["dx"]["finite"]
        and report["dx"]["shape"] == [16, 128]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
