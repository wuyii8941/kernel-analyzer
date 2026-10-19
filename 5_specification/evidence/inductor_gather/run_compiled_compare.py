#!/usr/bin/env python3
"""Inspect compiled gather without treating eager output as ground truth."""
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
import inductor_gather as candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    generator = torch.Generator(device="cpu").manual_seed(20261019)
    x = torch.randn(candidate.ROWS, candidate.COLS, generator=generator).cuda()
    index = torch.randint(
        0, candidate.COLS, (candidate.ROWS, candidate.SELECTED),
        generator=generator, dtype=torch.int64,
    ).cuda()
    before = {"x": x.clone(), "index": index.clone()}
    generated = run_and_get_triton_code(candidate._compiled, x, index)
    actual = candidate.call({
        "x": x,
        "index": index,
        "dim": candidate.DIM,
        "sparse_grad": candidate.SPARSE_GRAD,
    })["out"]
    eager = candidate._gather(x, index)
    kernel_names = sorted(set(re.findall(
        r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE
    )))
    report = {
        "task": "PyTorch functional gather",
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "out": {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "finite": bool(torch.isfinite(actual).all()),
            "compiled_equals_eager": bool(torch.equal(actual, eager)),
        },
        "inputs_unchanged": (
            torch.equal(x, before["x"])
            and torch.equal(index, before["index"])
        ),
    }
    report["inspection_passed"] = (
        len(kernel_names) > 0
        and report["inputs_unchanged"]
        and report["out"]["finite"]
        and report["out"]["shape"] == [candidate.ROWS, candidate.SELECTED]
        and report["out"]["compiled_equals_eager"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
