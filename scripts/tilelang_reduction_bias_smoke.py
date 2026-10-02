#!/usr/bin/env python3
"""Run a small real TileLang reduction through the shared bias checker.

This example covers a reduction family separately from the GEMM and
elementwise examples.  It is a smoke validation of the adapter contract, not
an assertion that this one reduction has a training-level systematic bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from kernel_analyzer import check_tilelang_bias


@tilelang.jit
def reduce_sum(A, block: int = 128):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    C = T.empty((1,), T.float32)
    with T.Kernel(1, threads=128):
        reducer = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(reducer)
        for i in T.Parallel(N):
            T.reducer_update(reducer[0], A[i])
        out = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(reducer, out)
        T.copy(out, C)
    return C


def make_inputs(index: int):
    generator = torch.Generator(device="cuda").manual_seed(9200 + index)
    return (torch.randn((128,), device="cuda", dtype=torch.float16, generator=generator),)


def reference(a):
    return a.float().sum().reshape(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TileLang reduction smoke run")
    report = check_tilelang_bias(
        candidate=reduce_sum,
        reference=reference,
        make_inputs=make_inputs,
        samples=args.samples,
        calibration_samples=max(1, args.samples // 2),
        check_backward=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": report.get("backend"),
                "status": report.get("status"),
                "measurement_status": report.get("measurement_status"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
