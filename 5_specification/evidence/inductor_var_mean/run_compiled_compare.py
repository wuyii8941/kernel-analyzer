#!/usr/bin/env python3
"""Inspect compiled var_mean/combined VJP without treating eager output as ground truth."""
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
import inductor_var_mean as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-code", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if args.generated_code is not None and args.generated_code.exists():
        raise FileExistsError(
            f"refusing to overwrite generated-code evidence: {args.generated_code}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    generator = torch.Generator(device="cpu").manual_seed(20261019)
    x = (torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2).cuda()
    upstream_var = torch.randn(candidate.ROWS, generator=generator).cuda()
    upstream_mean = torch.randn(candidate.ROWS, generator=generator).cuda()
    before = {
        "x": x.clone(),
        "upstream_var": upstream_var.clone(),
        "upstream_mean": upstream_mean.clone(),
    }
    generated = run_and_get_triton_code(
        candidate._compiled, x, upstream_var, upstream_mean
    )
    actual = candidate.call({
        "x": x,
        "upstream_var": upstream_var,
        "upstream_mean": upstream_mean,
        "dim": candidate.DIM,
        "correction": candidate.CORRECTION,
        "keepdim": candidate.KEEPDIM,
    })
    eager_variance, eager_mean, eager_dx = candidate._forward_and_vjp(
        x, upstream_var, upstream_mean
    )
    kernel_names = sorted(set(re.findall(
        r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE
    )))
    variance_gradient_scale = float(torch.tensor(2 / 127, dtype=torch.float32))
    variance_gradient_error = (
        Fraction.from_float(variance_gradient_scale) - Fraction(2, 127)
    )
    mean_gradient_scale = float(torch.tensor(1 / 128, dtype=torch.float32))
    variance_gradient_literal = "0.015748031496062992"
    mean_gradient_literal = "0.0078125"
    report = {
        "task": "PyTorch ordered variance/mean pair and combined input VJP",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "generated_code_uses_welford": "welford" in generated.lower(),
        "generated_code_has_variance_divisor_127": bool(
            re.search(r"= 127\.0(?:\s|$)", generated, re.MULTILINE)
        ),
        "generated_code_has_variance_gradient_scale_literal": (
            variance_gradient_literal in generated
        ),
        "generated_code_has_mean_gradient_scale_literal": (
            mean_gradient_literal in generated
        ),
        "combined_vjp_scales": {
            "variance_task_value": "2/127",
            "variance_generated_decimal_literal": variance_gradient_literal,
            "variance_value_after_float32_conversion": variance_gradient_scale,
            "variance_float32_minus_task": str(variance_gradient_error),
            "mean_task_value": "1/128",
            "mean_generated_decimal_literal": mean_gradient_literal,
            "mean_value_after_float32_conversion": mean_gradient_scale,
        },
        "outputs": {},
        "inputs_unchanged": all(
            torch.equal(tensor, before[name])
            for name, tensor in {
                "x": x,
                "upstream_var": upstream_var,
                "upstream_mean": upstream_mean,
            }.items()
        ),
    }
    eager = {
        "variance": eager_variance,
        "mean": eager_mean,
        "dx": eager_dx,
    }
    expected_shapes = {
        "variance": [candidate.ROWS],
        "mean": [candidate.ROWS],
        "dx": [candidate.ROWS, candidate.COLS],
    }
    for name in ("variance", "mean", "dx"):
        report["outputs"][name] = {
            "shape": list(actual[name].shape),
            "dtype": str(actual[name].dtype),
            "finite": bool(torch.isfinite(actual[name]).all()),
            "max_abs_compiled_minus_eager": float(
                (actual[name] - eager[name]).abs().max().cpu()
            ),
        }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["generated_code_has_variance_divisor_127"]
        and report["generated_code_has_variance_gradient_scale_literal"]
        and report["generated_code_has_mean_gradient_scale_literal"]
        and report["inputs_unchanged"]
        and all(report["outputs"][name]["finite"] for name in expected_shapes)
        and all(
            report["outputs"][name]["shape"] == shape
            for name, shape in expected_shapes.items()
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.generated_code is not None:
        args.generated_code.parent.mkdir(parents=True, exist_ok=True)
        args.generated_code.write_text(generated)
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
