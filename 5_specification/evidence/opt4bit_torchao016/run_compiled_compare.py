#!/usr/bin/env python3
"""Check the complete physical reference against torch.compile's Triton composition."""
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
from torchao.optim.subclass_4bit import OptimState4bit

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC))
from adapters import OPT4BIT  # noqa: E402


def copy_state(dst, src):
    dst.copy_(src)
    return dst


def fractions(values):
    return [Fraction.from_float(float(value)) for value in values]


def compare(state, src_cpu, signed, block_size):
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
        "scale": fractions(state.scale.cpu().tolist()),
        "qmap": fractions(state.qmap.cpu().tolist()),
    }
    mismatch_counts = {
        field: sum(got != want for got, want in zip(actual[field], expected[field]))
        + abs(len(actual[field]) - len(expected[field]))
        for field in expected
    }
    return expected, mismatch_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    compiled = torch.compile(copy_state, fullgraph=True, dynamic=False)
    generator = torch.Generator(device="cpu").manual_seed(20261019)
    first = torch.randn(4096, generator=generator, dtype=torch.float32)
    first_src = first.cuda()
    first_state = OptimState4bit.zeros(first_src.shape, True, 128, first_src.device)
    generated = run_and_get_triton_code(compiled, first_state, first_src)
    kernel_names = sorted(set(re.findall(r"^def (triton_[A-Za-z0-9_]+)\(", generated, re.MULTILINE)))

    rows = []
    inputs = [("signed_first", first, True)]
    for seed in range(1, 9):
        signed = seed % 2 == 1
        value = torch.randn(4096, generator=generator, dtype=torch.float32)
        if not signed:
            value = value.abs().square()
        inputs.append((f"{'signed' if signed else 'unsigned'}_{seed}", value, signed))
    inputs.extend([
        ("signed_zero", torch.zeros(4096, dtype=torch.float32), True),
        ("unsigned_zero", torch.zeros(4096, dtype=torch.float32), False),
    ])

    for name, src_cpu, signed in inputs:
        src = src_cpu.cuda()
        before = src.clone()
        state = OptimState4bit.zeros(src.shape, signed, 128, src.device)
        compiled(state, src)
        torch.cuda.synchronize()
        expected, mismatch_counts = compare(state, src_cpu, signed, 128)
        row = {
            "case": name,
            "signed": signed,
            "code_coordinates": len(expected["codes"]),
            "scale_coordinates": len(expected["scale"]),
            "qmap_coordinates": len(expected["qmap"]),
            "mismatch_counts": mismatch_counts,
            "source_unchanged": torch.equal(src, before),
        }
        row["passed"] = not any(mismatch_counts.values()) and row["source_unchanged"]
        rows.append(row)

    report = {
        "task": "torch.compile of torchao 0.16.0 OptimState4bit.copy_ on CUDA float32[4096]",
        "torch": torch.__version__,
        "torchao": importlib.metadata.version("torchao"),
        "triton": importlib.metadata.version("triton"),
        "profile": OPT4BIT.PROFILE,
        "generated_triton_kernel_names": kernel_names,
        "generated_triton_kernel_count": len(kernel_names),
        "source_graph_mentions": {
            "abs_amax_clamp_copy": all(name in generated for name in ("aten.abs", "aten.amax", "aten.clamp", "aten.copy_")),
            "divide_and_quantize": all(name in generated for name in ("aten.div", "aten.ge", "aten.index", "aten.sub", "aten.mul")),
            "pack_codes": all(name in generated for name in ("aten.__lshift__", "aten.bitwise_or", "aten.slice")),
        },
        "rows": rows,
        "total_rows": len(rows),
        "passed_rows": sum(row["passed"] for row in rows),
        "all_passed": len(kernel_names) == 3 and all(report_value for report_value in [
            all(name in generated for name in ("aten.abs", "aten.amax", "aten.clamp", "aten.copy_")),
            all(name in generated for name in ("aten.div", "aten.ge", "aten.index", "aten.sub", "aten.mul")),
            all(name in generated for name in ("aten.__lshift__", "aten.bitwise_or", "aten.slice")),
        ]) and all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "torch", "torchao", "triton", "generated_triton_kernel_count", "total_rows", "passed_rows", "all_passed")}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
