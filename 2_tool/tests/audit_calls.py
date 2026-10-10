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


def slow(inp):
    """Exceeds a one-second case budget (structured 'over budget' result)."""
    import time
    time.sleep(3)
    return {"y": _affine_of(inp["x"])}


def aten_output(inp):
    """An output written by ATen, not by Triton: no reference (structured 'not established' result)."""
    _affine_of(inp["x"])
    return {"y": inp["x"] * 2.0}


_compiled = None


def compiled_after_copy(inp):
    """An ATen copy of the input read by an Inductor-compiled kernel.  Under the trace of the provenance run the
    compiled function runs as eager ATen ops (no Triton launch), so the runs cannot be aligned: no producer record."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(lambda t: t * 3.0 + 1.0)
    x = inp["x"]
    return {"y": _compiled(x.reshape(32, -1).t().contiguous().reshape(-1))}


@triton.jit
def _row_cumsum(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(y_ptr + r * N + offs, tl.cumsum(tl.load(x_ptr + r * N + offs), 0))


def cumsum_rows(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    _row_cumsum[(x.shape[0],)](x, y, N=x.shape[1])
    return {"y": y}


def cancelling_rows(seed, shape, dtype):
    """Rows of +-2^20 pairs plus small values: prefix sums that cancel, where the gamma_n bound is loose."""
    import numpy as np
    rng = np.random.default_rng(seed)
    big = np.where(np.arange(shape[-1]) % 2 == 0, 2.0 ** 20, -2.0 ** 20)
    a = big + rng.uniform(0.5, 1.5, shape)
    return torch.tensor(a, dtype=torch.float32, device="cuda")


# ---------------------------------------------------------------------------------------------------------------------
# Batch 1 (memory-effect rules, storage_effects M1-M5): the three audit classes and the capabilities they must keep.

def cross_input_rounded(inp):
    """Class 1 (cross-input digest): ATen arithmetic writes, through .data (no version increment), a value that rounds
    to the bytes of input w into input x; the kernel then reads x.  Its bytes are an input's bytes, not x's own."""
    x, w = inp["x"], inp["w"]
    x.data.copy_(w * (1 + 2.0 ** -30))
    return {"y": _affine_of(x)}


def inplace_unchanged(inp):
    """Class 2 (in-place arithmetic, bytes unchanged): x + 2^-25 rounds back to x for x in [1, 2)."""
    x = inp["x"]
    x.add_(2.0 ** -25)
    return {"y": _affine_of(x)}


def inplace_unchanged_data(inp):
    """Class 2 through .data: no version increment; the producer trace sees the ATen write."""
    x = inp["x"]
    x.data.add_(2.0 ** -25)
    return {"y": _affine_of(x)}


def inplace_between_launches(inp):
    """Class 2 between recorded launches: t = 3x + 1 in [4, 7); t + 2^-23 (a quarter ulp) leaves every byte."""
    t = _affine_of(inp["x"])
    t.add_(2.0 ** -23)
    return {"y": _affine_of(t)}


def overlap_fill(inp):
    """Class 3 (overlapping view, partial write): a full-size stride-0 view of a computed buffer writes element 0
    only; numel * element_size equals the storage size."""
    b = inp["x"] * 3.0
    b.as_strided((b.numel(),), (0,)).fill_(0.5)
    return {"y": _affine_of(b)}


def partial_init_read(inp):
    """Kept capability: half of an uninitialised buffer is a copy of x, and the kernel reads only that half."""
    x = inp["x"]
    n = x.numel()
    b = torch.empty(2 * n, device=x.device, dtype=x.dtype)
    b[:n].copy_(x)
    return {"y": _affine_of(b[:n])}


def zeros_then_partial_copy(inp):
    """Kept capability: an exact constant buffer, half overwritten by a copy of x, read whole."""
    x = inp["x"]
    n = x.numel()
    b = torch.zeros(2 * n, device=x.device, dtype=x.dtype)
    b[n:].copy_(x)
    return {"y": _affine_of(b)}


def inplace_after_read(inp):
    """Kept capability: an in-place op on x after the kernel read it."""
    x = inp["x"]
    y = _affine_of(x)
    x.add_(1.0)
    return {"y": y}


def input_offset_view(inp):
    """Kept capability (legal alias): the kernel reads an offset view of the declared input."""
    return {"y": _affine_of(inp["x"][1:])}


_compiled_inplace = None


def compiled_inplace_then_kernel(inp):
    """Kept capability: an Inductor-compiled in-place op on x (AOTAutograd announces the raw-pointer write by a
    version increment before the compiled kernel runs), then a kernel reads x."""
    global _compiled_inplace
    if _compiled_inplace is None:
        _compiled_inplace = torch.compile(lambda t: t.mul_(2.0))
    x = inp["x"]
    _compiled_inplace(x)
    return {"y": _affine_of(x)}


# ---------------------------------------------------------------------------------------------------------------------
# Batch 1 (precision controller): the three-level summation counterexample.

@triton.jit
def _row_sum(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(y_ptr + r, tl.sum(tl.load(x_ptr + r * N + offs), axis=0))


def row_sum(inp):
    x = inp["x"]
    y = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    _row_sum[(x.shape[0],)](x, y, N=x.shape[1])
    return {"y": y}


def three_level_rows(seed, shape, dtype):
    """Rows whose exact sum is a small value s ~ 2^-100 under cancelling pairs +-2^e (e in [100, 120], one pair at
    2^120): with S = sum |x| ~ 2^125 and n = 128, the SumK tail gamma_254^K S is ~2^-10 (K = 3) and ~2^-100 (K = 5),
    both far above 1/8 ulp(s) = 2^-126, and ~2^-235 (K = 8).  Levels 1 and 2 resolve no element; level 3 all."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows, n = shape
    a = np.zeros(shape)
    for r in range(rows):
        e = rng.integers(100, 121, n // 2 - 1)
        e[0] = 120
        signs = rng.choice([-1.0, 1.0], n // 2 - 1)
        mags = signs * np.exp2(e.astype(np.float64))
        pairs = np.stack([mags, -mags], 1).reshape(-1)
        small = float(np.float32(rng.uniform(1.0, 2.0) * 2.0 ** -100))
        a[r] = rng.permutation(np.concatenate([pairs, [small, 0.0]]))
    return torch.tensor(a, dtype=torch.float32, device="cuda")
