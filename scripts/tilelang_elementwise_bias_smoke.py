#!/usr/bin/env python3
"""Run a small TileLang elementwise kernel through the shared bias checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from kernel_analyzer import check_tilelang_bias


@tilelang.jit
def scale(A, block: int = 128):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(T.ceildiv(N, block), threads=128) as (bx,):
        for i in T.Parallel(block):
            C[bx * block + i] = A[bx * block + i] * T.float16(1.25)
    return C


def make_inputs(index: int):
    generator = torch.Generator(device="cuda").manual_seed(9100 + index)
    return (torch.randn((128,), device="cuda", dtype=torch.float16, generator=generator),)


def reference(a):
    return (a.float() * 1.25).to(torch.float16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TileLang elementwise smoke run")
    report = check_tilelang_bias(
        candidate=scale,
        reference=reference,
        make_inputs=make_inputs,
        samples=args.samples,
        calibration_samples=args.samples // 2,
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
