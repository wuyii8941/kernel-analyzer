#!/usr/bin/env python3
"""Inspect compiled standard deviation/VJP without using eager as ground truth."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_triton_code


ROOT = Path(__file__).resolve().parents[3]
CALLS = ROOT / "5_specification/corpus/calls"
sys.path.insert(0, str(CALLS))
import inductor_std as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    generator = torch.Generator(device="cpu").manual_seed(20261019)
    x = (torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2).cuda()
    upstream = torch.randn(candidate.ROWS, generator=generator).cuda()
    before = {"x": x.clone(), "upstream": upstream.clone()}
    generated = run_and_get_triton_code(candidate._compiled, x, upstream)
    actual = candidate.call({
        "x": x,
        "upstream": upstream,
        "dim": candidate.DIM,
        "correction": candidate.CORRECTION,
        "keepdim": candidate.KEEPDIM,
    })
    eager_y, eager_dx = candidate._forward_and_vjp(x, upstream)
    kernel_names = sorted(set(re.findall(
        r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE
    )))
    gradient_scale_literal = "0.015748031496062992"
    gradient_scale_f32 = float(torch.tensor(2 / 127, dtype=torch.float32))
    gradient_scale_error = Fraction.from_float(gradient_scale_f32) - Fraction(2, 127)
    report = {
        "task": "PyTorch corrected standard deviation and input VJP",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "generated_code_has_sqrt_or_rsqrt": (
            "sqrt" in generated.lower() or "rsqrt" in generated.lower()
        ),
        "generated_code_uses_tl_sqrt_rn": "tl.sqrt_rn" in generated,
        "generated_code_has_zero_std_branch": bool(
            re.search(r"==\s*tmp\d+", generated)
            and re.search(r"tl\.where\(", generated)
        ),
        "generated_code_has_gradient_scale_literal": (
            gradient_scale_literal in generated
        ),
        "gradient_scale": {
            "task_factor_before_division_by_two_std": "2/127",
            "generated_decimal_literal": gradient_scale_literal,
            "value_after_float32_conversion": gradient_scale_f32,
            "float32_minus_task": str(gradient_scale_error),
        },
        "outputs": {
            "y": {
                "shape": list(actual["y"].shape),
                "dtype": str(actual["y"].dtype),
                "finite": bool(torch.isfinite(actual["y"]).all()),
                "strictly_positive": bool((actual["y"] > 0).all()),
                "max_abs_compiled_minus_eager": float(
                    (actual["y"] - eager_y).abs().max().cpu()
                ),
            },
            "dx": {
                "shape": list(actual["dx"].shape),
                "dtype": str(actual["dx"].dtype),
                "finite": bool(torch.isfinite(actual["dx"]).all()),
                "max_abs_compiled_minus_eager": float(
                    (actual["dx"] - eager_dx).abs().max().cpu()
                ),
            },
        },
        "inputs_unchanged": (
            torch.equal(x, before["x"])
            and torch.equal(upstream, before["upstream"])
        ),
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["generated_code_has_sqrt_or_rsqrt"]
        and report["generated_code_uses_tl_sqrt_rn"]
        and report["generated_code_has_zero_std_branch"]
        and report["generated_code_has_gradient_scale_literal"]
        and report["inputs_unchanged"]
        and report["outputs"]["y"]["finite"]
        and report["outputs"]["y"]["strictly_positive"]
        and report["outputs"]["dx"]["finite"]
        and report["outputs"]["y"]["shape"] == [candidate.ROWS]
        and report["outputs"]["dx"]["shape"] == [candidate.ROWS, candidate.COLS]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
