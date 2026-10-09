"""Calls for the end-to-end regressions of the external audit fixes (3_audits/README.md, F04-F07), declared to
``kernel_analyzer.measure`` as ``audit_calls.py:<function>``: a classic Triton kernel, the same kernel written in Gluon
(TTGIR only, no TTIR), a float-atomic row sum, and upstream ATen buffers whose values equal the inputs without being
the inputs.  Helper module, not a test file."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK = 128


@triton.jit
def _affine(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 3.0 + 1.0, mask=m)


@gluon.jit
def _affine_gluon(x_ptr, y_ptr, n, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    m = offs < n
    gl.store(y_ptr + offs, gl.load(x_ptr + offs, mask=m) * 3.0 + 1.0, mask=m)


@triton.jit
def _row_sum_atomic(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + r * cols + offs, mask=offs < cols, other=0.0)
    tl.atomic_add(y_ptr + r, tl.sum(v, axis=0))


def _affine_of(t):
    y = torch.empty_like(t)
    n = t.numel()
    _affine[(triton.cdiv(n, BLOCK),)](t, y, n, BLOCK=BLOCK)
    return y


def classic(inp):
    return {"y": _affine_of(inp["x"])}


def gluon_affine(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    n = x.numel()
    _affine_gluon[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK, num_warps=4)
    return {"y": y}


def atomic_row_sum(inp):
    x = inp["x"]
    rows, cols = x.shape
    y = torch.zeros(rows, device=x.device, dtype=x.dtype)
    _row_sum_atomic[(rows, triton.cdiv(cols, BLOCK))](x, y, cols, BLOCK=BLOCK)
    return {"y": y}


def value_equal_upstream(inp):
    """ATen arithmetic whose rounded result equals an input element bit for bit (x in [1, 2): x + 2^-25 rounds back
    to x), although its real value is not an input value (audit F04 counterexample)."""
    x = inp["x"]
    return {"y": _affine_of((x[:1] + 2.0 ** -25).repeat(x.numel()))}


def zero_upstream(inp):
    """ATen arithmetic that underflows to an all-zero buffer whose real value is not zero."""
    return {"y": _affine_of(inp["x"] * 1e-30 * 1e-30)}


def honest_copy(inp):
    """An ATen layout change: a true copy of the input, but no producer record proves it."""
    x = inp["x"]
    return {"y": _affine_of(x.reshape(32, -1).t().contiguous().reshape(-1))}
