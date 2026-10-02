"""Kernels with declared single-site mutations for the real-kernel mutation experiment.

Each kernel takes ``MUT: tl.constexpr``; 0 is the original.  The mutation
table in run_mutation_experiment.py records, for every MUT value, what
changes and what is known in advance (only local statements; whether the
average effect is zero is recorded as an observation).
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def m_scale(X, Y, alpha, n, BLOCK: tl.constexpr, MUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    if MUT == 1:
        y = libdevice.mul_rd(x, alpha)
    elif MUT == 2:
        y = libdevice.mul_ru(x, alpha)
    elif MUT == 3:
        y = x.to(tl.bfloat16).to(tl.float32) * alpha
    elif MUT == 4:
        y = alpha * x  # bitwise-equivalent rewrite (commutative IEEE multiply)
    else:
        y = x * alpha
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def m_sum4(A, B, C, D, Y, n, BLOCK: tl.constexpr, MUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = tl.load(C + offs, mask=mask)
    d = tl.load(D + offs, mask=mask)
    if MUT == 1:
        s = a + (b + (c + d))  # summation order
    elif MUT == 2:
        s = libdevice.add_rd((a + b) + c, d)  # last addition rounds down
    elif MUT == 3:
        s = ((a + b).to(tl.bfloat16).to(tl.float32) + c) + d  # lower intermediate precision
    elif MUT == 4:
        s = (a + b) + c  # omitted term
    elif MUT == 5:
        s = ((b + a) + c) + d  # bitwise-equivalent rewrite
    else:
        s = ((a + b) + c) + d
    tl.store(Y + offs, s, mask=mask)


@triton.jit
def m_row_sum(X, Y, n_cols, BLOCK: tl.constexpr, MUT: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    if MUT == 3:
        x = tl.load(X + row * n_cols + cols, mask=cols < n_cols - 1, other=0.0)  # omitted term
    else:
        x = tl.load(X + row * n_cols + cols, mask=cols < n_cols, other=0.0)
    if MUT == 1:
        s = tl.sum(tl.flip(x, 0), axis=0)  # different reduction tree
    elif MUT == 2:
        s = tl.sum(x.to(tl.bfloat16).to(tl.float32), axis=0)  # lower input precision
    elif MUT == 4:
        s = tl.sum(x * 1.0, axis=0)  # bitwise-equivalent rewrite
    else:
        s = tl.sum(x, axis=0)
    tl.store(Y + row, s)


@triton.jit
def m_softmax(X, Y, n_cols, stride, BLOCK: tl.constexpr, MUT: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf"))
    if MUT == 3:
        e = tl.exp(x)  # max subtraction removed: same real value, different execution
    else:
        e = tl.exp(x - tl.max(x, axis=0))
    if MUT == 2:
        e = e.to(tl.bfloat16).to(tl.float32)
    s = tl.sum(e, axis=0)
    if MUT == 1:
        y = tl.div_rn(e, tl.broadcast_to(s, e.shape))  # correctly rounded division
    else:
        y = e / s
    tl.store(Y + row * stride + cols, y, mask=mask)


@triton.jit
def m_layernorm(X, W, B, Y, n_cols, eps, BLOCK: tl.constexpr, MUT: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    if MUT == 2:
        var = tl.sum(x * x, axis=0) / n_cols - mean * mean  # algebraic reformulation
    else:
        var = tl.sum(d * d, axis=0) / n_cols
    if MUT == 1:
        r = 1.0 / tl.sqrt_rn(var + eps)
    else:
        r = tl.math.rsqrt(var + eps)
    if MUT == 3:
        r = r.to(tl.bfloat16).to(tl.float32)
    y = d * r * tl.load(W + cols, mask=mask) + tl.load(B + cols, mask=mask)
    tl.store(Y + row * n_cols + cols, y, mask=mask)


@triton.jit
def m_matmul(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MUT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    n_k = tl.cdiv(K, BK)
    if MUT == 4:
        n_k = n_k - 1  # omitted last K block
    for kk in range(0, n_k):
        if MUT == 3:
            k = n_k - 1 - kk  # reversed K order
        else:
            k = kk
        ks = k * BK + rk
        a = tl.load(A + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(B + ks[:, None] * N + rn[None, :], mask=(ks[:, None] < K) & (rn[None, :] < N), other=0.0)
        if MUT == 1:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        if MUT == 2:
            acc = acc.to(tl.bfloat16).to(tl.float32)  # lower accumulator precision
    tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def m_accumulate(ACC, Cc, n, BLOCK: tl.constexpr, MUT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(ACC + offs, mask=mask)
    c = tl.load(Cc + offs, mask=mask)
    if MUT == 1:
        s = libdevice.add_rd(acc, c)
    elif MUT == 2:
        s = libdevice.add_ru(acc, c)
    elif MUT == 3:
        s = c + acc  # bitwise-equivalent rewrite
    else:
        s = acc + c
    tl.store(ACC + offs, s, mask=mask)
