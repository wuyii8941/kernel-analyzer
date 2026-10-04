#!/usr/bin/env python3
"""Probe: how does Triton's default tl.dot on fp32 inputs (input_precision "tf32") convert fp32 to TF32?

Compares the Triton result with exact products of the inputs truncated to TF32 (low 13 mantissa bits dropped)
and rounded to nearest TF32, and with cuBLAS in TF32 mode; reports the aligned relative error
E[(C - C_exact) * sign(C_exact)] / E|C_exact| (negative: magnitudes shrink) and whether the PTX contains a
rounding conversion (cvt.rna.tf32.f32) before the tensor-core instruction.

    python scripts/probe_tf32_dot.py
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def tf32_trunc(x):
    u = x.astype(np.float32).view(np.uint32) & np.uint32(0xFFFFE000)
    return u.view(np.float32).astype(np.float64)


def tf32_rn(x):
    # round to nearest (ties away, like cvt.rna) at 10 explicit mantissa bits
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x1000) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def aligned(c, exact):
    return float(np.mean((c - exact) * np.sign(exact)) / np.mean(np.abs(exact)))


def main():
    M = N = 64
    torch.manual_seed(0)
    rows = []
    for K in (64, 512, 4096):
        a = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")
        c = torch.empty(M, N, device="cuda")
        h = mm[(1,)](a, b, c, M, N, K, 32)
        A, B, C = a.cpu().double().numpy(), b.cpu().double().numpy(), c.cpu().double().numpy()
        exact = A @ B
        trunc = tf32_trunc(A) @ tf32_trunc(B)
        rn = tf32_rn(A) @ tf32_rn(B)
        torch.backends.cuda.matmul.allow_tf32 = True
        cub = (a @ b).cpu().double().numpy()
        torch.backends.cuda.matmul.allow_tf32 = False
        ptx = h.asm["ptx"]
        rows.append((K, aligned(C, exact), aligned(trunc, exact), aligned(rn, exact), aligned(cub, exact),
                     float(np.max(np.abs(C - trunc)) / np.max(np.abs(exact))), float(np.max(np.abs(C - rn)) / np.max(np.abs(exact))),
                     "cvt.rna.tf32" in ptx, ptx.count("mma.sync")))
    print("K | aligned rel error: triton  trunc-model  rn-model  cuBLAS-tf32 | max|C-trunc|  max|C-rn| | cvt.rna in PTX, mma count")
    for r in rows:
        print(f"{r[0]:5d} | {r[1]: .3e} {r[2]: .3e} {r[3]: .3e} {r[4]: .3e} | {r[5]:.2e} {r[6]:.2e} | {r[7]} {r[8]}")


if __name__ == "__main__":
    main()
