#!/usr/bin/env python3
"""Compare the independent OPT4BIT adapter with installed torchao 0.16.0.

The output is append-only evidence: an existing path is never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from fractions import Fraction
from pathlib import Path

import torch
from torchao.optim.subclass_4bit import OptimState4bit

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC))
from adapters import OPT4BIT  # noqa: E402


def fraction(value):
    return Fraction.from_float(float(value))


def cases():
    generator = torch.Generator(device="cpu").manual_seed(20261019)
    signed_large = torch.randn(4096, generator=generator, dtype=torch.float32)
    unsigned_large = torch.rand(4096, generator=generator, dtype=torch.float32).square()
    return [
        ("signed_fixed", torch.tensor([1, 0.5, -1, 0.125], dtype=torch.float32), True, 2),
        ("unsigned_fixed", torch.tensor([1, 0.5, 0, 0.09375], dtype=torch.float32), False, 2),
        ("signed_zero", torch.zeros(8, dtype=torch.float32), True, 4),
        ("unsigned_zero", torch.zeros(8, dtype=torch.float32), False, 4),
        ("signed_two_scales", torch.tensor([1, 0.8125, -0.5, 0.25, 0.125, -0.0625, 0.03125, 0], dtype=torch.float32), True, 4),
        ("signed_default_block", signed_large, True, 128),
        ("unsigned_default_block", unsigned_large, False, 128),
    ]


def run_case(name, src_cpu, signed, block_size, device):
    src = src_cpu.to(device)
    before = src.clone()
    state = OptimState4bit.zeros(src.shape, signed=signed, block_size=block_size, device=device)
    state.copy_(src)
    inp = {
        "src": src_cpu.tolist(),
        "input_dtype": "f32",
        "signed": signed,
        "block_size": block_size,
        "qmap": list(OPT4BIT.QMAP_SIGNED if signed else OPT4BIT.QMAP_UNSIGNED),
    }
    expected = dict(OPT4BIT.reference(OPT4BIT.PROFILE, inp).fields)
    expected["qmap"] = list(OPT4BIT.QMAP_SIGNED if signed else OPT4BIT.QMAP_UNSIGNED)
    actual = {
        "codes": state.codes.cpu().tolist(),
        "scale": [fraction(x) for x in state.scale.cpu().tolist()],
        "qmap": [fraction(x) for x in state.qmap.cpu().tolist()],
    }
    mismatches = {}
    for field in expected:
        bad = [i for i, (got, want) in enumerate(zip(actual[field], expected[field])) if got != want]
        if len(actual[field]) != len(expected[field]):
            bad.append("length")
        if bad:
            mismatches[field] = bad[:16]
    attributes_ok = (
        state.signed is signed
        and tuple(state.shape) == tuple(src.shape)
        and state.block_size == block_size
        and state.codes.dtype is torch.uint8
        and state.scale.dtype is torch.float32
        and state.qmap.dtype is torch.float32
    )
    source_unchanged = torch.equal(src, before)
    return {
        "case": name,
        "device": device,
        "signed": signed,
        "numel": src.numel(),
        "block_size": block_size,
        "code_coordinates": len(expected["codes"]),
        "scale_coordinates": len(expected["scale"]),
        "qmap_coordinates": len(expected["qmap"]),
        "mismatches": mismatches,
        "attributes_ok": attributes_ok,
        "source_unchanged": source_unchanged,
        "passed": not mismatches and attributes_ok and source_unchanged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = [run_case(name, src, signed, block_size, device)
            for device in devices for name, src, signed, block_size in cases()]
    report = {
        "task": "torchao 0.16.0 OptimState4bit.zeros(...).copy_(float32 source)",
        "torch": torch.__version__,
        "torchao": importlib.metadata.version("torchao"),
        "profile": OPT4BIT.PROFILE,
        "rows": rows,
        "total_rows": len(rows),
        "passed_rows": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("torch", "torchao", "total_rows", "passed_rows", "all_passed")}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
