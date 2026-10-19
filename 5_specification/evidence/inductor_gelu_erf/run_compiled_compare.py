#!/usr/bin/env python3
"""Inspect the retained exact-form Inductor GELU call without using eager output as its specification."""
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
import inductor_gelu_erf as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    generator = torch.Generator(device="cpu").manual_seed(20261019)
    cpu_inputs = {
        "x": torch.randn(candidate.ROWS, candidate.COLS, generator=generator) * 2,
        "upstream": torch.randn(candidate.ROWS, candidate.COLS, generator=generator),
    }
    tensors = {name: value.cuda() for name, value in cpu_inputs.items()}
    before = {name: value.clone() for name, value in tensors.items()}
    generated = run_and_get_triton_code(candidate._compiled, tensors["x"], tensors["upstream"])
    actual = candidate.call(tensors | {"approximate": candidate.APPROXIMATE})
    eager = candidate._forward_and_vjp(tensors["x"], tensors["upstream"])
    rows = []
    for name, expected in zip(("y", "dx"), eager):
        got = actual[name]
        rows.append({
            "name": name,
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "finite": bool(torch.isfinite(got).all()),
            "max_abs_compiled_minus_eager": float((got - expected).abs().max().cpu()),
        })
    kernel_names = sorted(set(re.findall(r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE)))
    precision_pattern = re.compile(
        r"libdevice\.(?:erf|exp)\b|aten\.(?:erf|exp)(?:\.|\b)|0\.7071067811865476|0\.3989422804014327"
    )
    precision_lines = [
        line.strip() for line in generated.splitlines() if precision_pattern.search(line)
    ]
    report = {
        "task": "PyTorch public exact-form GELU forward and complete input VJP",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "generated_precision_lines": precision_lines,
        "rows": rows,
        "inputs_unchanged": all(torch.equal(tensors[name], value) for name, value in before.items()),
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["inputs_unchanged"]
        and all(row["finite"] for row in rows)
        and [row["shape"] for row in rows] == [[32, 128], [32, 128]]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
