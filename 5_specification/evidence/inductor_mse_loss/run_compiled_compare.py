#!/usr/bin/env python3
"""Inspect the compiled mean-MSE/VJP boundary without treating eager output as ground truth."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import torch
from torch._inductor.utils import run_and_get_triton_code


ROOT = Path(__file__).resolve().parents[3]
CALLS = ROOT / "5_specification/corpus/calls"
sys.path.insert(0, str(CALLS))
import inductor_mse_loss as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-code", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if args.generated_code is not None and args.generated_code.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.generated_code}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    generator = torch.Generator(device="cpu").manual_seed(20261019)
    tensors = {
        "x": (torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2).cuda(),
        "target": (torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2).cuda(),
        "upstream": torch.randn((), generator=generator).cuda(),
    }
    before = {name: value.clone() for name, value in tensors.items()}
    generated = run_and_get_triton_code(
        candidate._compiled, tensors["x"], tensors["target"], tensors["upstream"]
    )
    actual = candidate.call({**tensors, "reduction": "mean"})
    eager = candidate._forward_and_vjp(
        tensors["x"], tensors["target"], tensors["upstream"]
    )
    output_names = ("loss", "dx", "dtarget")
    outputs = {}
    for name, expected in zip(output_names, eager):
        got = actual[name]
        outputs[name] = {
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "finite": bool(torch.isfinite(got).all()),
            "max_abs_compiled_minus_eager": float((got - expected).abs().max().cpu()),
        }
    kernel_names = sorted(set(re.findall(
        r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE
    )))
    report = {
        "task": "PyTorch public unweighted mean-MSE and complete input/target VJP",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "generated_code_has_reduction": "tl.sum" in generated or "reduction" in generated.lower(),
        "outputs": outputs,
        "inputs_unchanged": all(
            torch.equal(tensors[name], value) for name, value in before.items()
        ),
        "upstream": float(tensors["upstream"].cpu()),
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["generated_code_has_reduction"]
        and report["inputs_unchanged"]
        and all(item["finite"] for item in outputs.values())
        and outputs["loss"]["shape"] == []
        and outputs["dx"]["shape"] == [candidate.ROWS, candidate.COLS]
        and outputs["dtarget"]["shape"] == [candidate.ROWS, candidate.COLS]
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
