#!/usr/bin/env python3
"""Run two real TileLang reduction callables through the family diagnostic.

The two kernels have the same mathematical reduction but visit the input in
different orders.  The report intentionally remains ``bias_decision=NOT_ASSESSED``:
it records controlled arithmetic evidence and odd-symmetry checks, not a
reference-free proof that either reduction is biased for an unknown workload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from kernel_analyzer import check_tilelang_reduction_order


@tilelang.jit
def reduce_sum_forward(A, block: int = 128):
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


@tilelang.jit
def reduce_sum_interleaved(A, block: int = 128):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    C = T.empty((1,), T.float32)
    with T.Kernel(1, threads=128):
        reducer = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(reducer)
        for i in T.Parallel(N):
            index = (i % 2) * (N // 2) + i // 2
            T.reducer_update(reducer[0], A[index])
        out = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(reducer, out)
        T.copy(out, C)
    return C


def make_inputs(index: int):
    values = torch.empty((128,), device="cuda", dtype=torch.float16)
    values[:32] = 65504.0
    values[32:64] = -65504.0
    values[64:96] = 0.1
    values[96:] = 0.1 + index * 0.01
    return (values,)


def make_negated_inputs(index: int):
    return (-make_inputs(index)[0],)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and TileLang are required for this smoke run")
    report = check_tilelang_reduction_order(
        candidate=reduce_sum_forward,
        variant=reduce_sum_interleaved,
        make_inputs=make_inputs,
        make_negated_inputs=make_negated_inputs,
        samples=args.samples,
        preserve_rng=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": report.get("backend"),
        "family": report.get("family"),
        "measurement_status": report.get("measurement_status"),
        "diagnostic_status": report.get("diagnostic_status"),
        "bias_decision": report.get("bias_decision"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
