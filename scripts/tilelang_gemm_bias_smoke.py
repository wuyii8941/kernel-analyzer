#!/usr/bin/env python3
"""Run a small real TileLang GEMM through the shared bias checker.

This is an executable smoke example, not a benchmark or a claim about a
training-level bias.  It is kept separate from the generic runner because the
TileLang DSL needs to see the kernel source from a Python file.
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
def gemm(A, B, block_M: int = 32, block_N: int = 32, block_K: int = 16):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def make_inputs(index: int):
    generator = torch.Generator(device="cuda").manual_seed(9000 + index)
    return (
        torch.randn((64, 64), device="cuda", dtype=torch.float16, generator=generator),
        torch.randn((64, 64), device="cuda", dtype=torch.float16, generator=generator),
    )


def reference(a, b):
    return torch.matmul(a.float(), b.float()).to(torch.float16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TileLang GEMM smoke run")
    report = check_tilelang_bias(
        candidate=gemm,
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
