#!/usr/bin/env python3
"""Probe: is tensor-core FP32 accumulation of bf16 products biased (e.g. truncation toward zero)?

bf16 x bf16 products are exact in FP32, so any systematic error of C = A @ B (FP32 output) against the exact
float64 product comes from the accumulation.  Reported for Triton tl.dot and cuBLAS, as a function of K: the aligned
relative error E[(C - C_exact) sign(C_exact)] / E|C_exact| (negative: magnitudes shrink) with a t-statistic over
independent repetitions, and the same for an FP32 CUDA-core sum (torch on float32 inputs, IEEE) as a control.

    python scripts/probe_tensor_core_accumulation.py
"""

from __future__ import annotations

import math

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


def aligned(c, exact):
    return float(np.mean((c - exact) * np.sign(exact)) / np.mean(np.abs(exact)))


def main():
    M = N = 64
    reps = 24
    print("K     | Triton bf16 dot        | cuBLAS bf16 (fp32 out)  | ulp(|C|) scale")
    for K in (256, 1024, 4096, 16384):
        tri, cub = [], []
        for r in range(reps):
            g = torch.Generator(device="cuda").manual_seed(1000 * K + r)
            a = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
            b = torch.randn(K, N, device="cuda", generator=g).to(torch.bfloat16)
            c = torch.empty(M, N, device="cuda")
            mm[(1,)](a, b, c, M, N, K, 64)
            exact = a.double().cpu().numpy() @ b.double().cpu().numpy()
            tri.append(aligned(c.double().cpu().numpy(), exact))
            cb = torch.matmul(a, b, out_dtype=torch.float32) if "out_dtype" in torch.matmul.__doc__ else \
                torch.mm(a.float(), b.float())  # fallback: FP32 inputs hold bf16 values exactly
            cub.append(aligned(cb.double().cpu().numpy(), exact))
        tri, cub = np.array(tri), np.array(cub)
        t = lambda x: x.mean() / (x.std(ddof=1) / math.sqrt(x.size))  # noqa: E731
        print(f"{K:5d} | {tri.mean(): .3e} (t={t(tri): 6.1f}) | {cub.mean(): .3e} (t={t(cub): 6.1f}) | "
              f"2^-24 = {2**-24:.1e}")


if __name__ == "__main__":
    main()
