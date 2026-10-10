"""Batch 1 hold-out calls (protocol.md): structures and implementation lineages that took no part in development.
Declared to ``kernel_analyzer.measure`` as ``holdout_calls.py:<function>``.  Written before the acceptance run; a
change after the freeze is a manual intervention and is recorded in the results."""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK = 256


# --------------------------------------------------------------------------------------------- kernels

@gluon.jit
def _scale_gluon(x_ptr, y_ptr, n, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    m = offs < n
    gl.store(y_ptr + offs, gl.load(x_ptr + offs, mask=m) * 5.0 - 2.0, mask=m)


@triton.jit
def _affine2d(x_ptr, y_ptr, R, C, sxr, sxc, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    c = tl.arange(0, BLOCK_C)[None, :]
    m = (r < R) & (c < C)
    v = tl.load(x_ptr + r * sxr + c * sxc, mask=m)
    tl.store(y_ptr + r * C + c, v * 3.0 + 1.0, mask=m)


@triton.jit
def _affine1d(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 3.0 + 1.0, mask=m)


@triton.jit
def _rtz_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m) * 3.0 + 1.0
    tl.store(y_ptr + offs, v.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.float32), mask=m)


@triton.jit
def _rne_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m) * 3.0 + 1.0
    tl.store(y_ptr + offs, v.to(tl.bfloat16).to(tl.float32), mask=m)


@triton.jit
def _sum_axis1(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(y_ptr + tl.arange(0, R), tl.sum(tl.load(x_ptr + r * C + c), axis=1))


@triton.jit
def _rev_cumsum(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(y_ptr + r * N + offs, tl.cumsum(tl.load(x_ptr + r * N + offs), 0, reverse=True))


@triton.jit
def _exp1d(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=m)), mask=m)


@triton.jit
def _row_sum64(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(y_ptr + r, tl.sum(tl.load(x_ptr + r * N + offs), axis=0))


@triton.jit
def _contended_int(x_ptr, z_ptr, y_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_add(z_ptr, tl.load(x_ptr + pid))
    tl.store(y_ptr + pid, old)


# --------------------------------------------------------------------------------------------- helpers

def _gluon_of(t):
    y = torch.empty_like(t)
    n = t.numel()
    _scale_gluon[(triton.cdiv(n, BLOCK),)](t, y, n, BLOCK=BLOCK, num_warps=4)
    return y


def _affine2d_of(v):
    R, C = v.shape
    y = torch.empty((R, C), device=v.device, dtype=v.dtype)
    bc = triton.next_power_of_2(C)
    _affine2d[(triton.cdiv(R, 16),)](v, y, R, C, v.stride(0), v.stride(1), BLOCK_R=16, BLOCK_C=bc)
    return y


def _affine_of(t):
    y = torch.empty_like(t)
    n = t.numel()
    _affine1d[(triton.cdiv(n, BLOCK),)](t, y, n, BLOCK=BLOCK)
    return y


_compiled = {}


def _compiled_affine(t):
    f = _compiled.get("affine")
    if f is None:
        f = _compiled["affine"] = torch.compile(lambda u: u * 3.0 + 1.0)
    return f(t)


# --------------------------------------------------------------------------------------------- provenance items

def gluon_inplace_tiny_add(inp):
    """H-P1a: x + 2^-25 rounds back to x for x in [1, 2) (every byte unchanged), then a Gluon kernel reads x."""
    x = inp["x"]
    x.add_(2.0 ** -25)
    return {"y": _gluon_of(x)}


def gluon_inplace_clamp_noop(inp):
    """H-P1b: clamp to [0, 10] of x in [1, 2) is the identity on these values, then a Gluon kernel reads x."""
    x = inp["x"]
    x.clamp_(0.0, 10.0)
    return {"y": _gluon_of(x)}


def strided_transposed_input(inp):
    """H-P2: a 2D strided kernel reads the transposed view of the declared input."""
    return {"y": _affine2d_of(inp["x"].t())}


def unfold_overlapping_read(inp):
    """H-P3: a 2D strided kernel reads overlapping windows of the declared input (unfold 8, step 4)."""
    return {"y": _affine2d_of(inp["x"].unfold(0, 8, 4))}


def where_copy(inp):
    """H-P4: elements of x or exact zeros (torch.where), then a kernel reads them."""
    x = inp["x"]
    return {"y": _affine_of(torch.where(x > 1.5, x, torch.zeros_like(x)))}


def cat_copy(inp):
    """H-P5: x followed by exact ones (torch.cat)."""
    x = inp["x"]
    return {"y": _affine_of(torch.cat([x, torch.ones(64, device=x.device, dtype=x.dtype)]))}


def index_put_partial_over_computed(inp):
    """H-P6: 3x (rounded) with every other element set to 0.5 through index_put_: half the bytes stay computed."""
    x = inp["x"]
    b = x * 3.0
    idx = torch.arange(0, b.numel(), 2, device=x.device)
    b.index_put_((idx,), torch.full((idx.numel(),), 0.5, device=x.device, dtype=x.dtype))
    return {"y": _affine_of(b)}


def index_put_copy(inp):
    """H-P7: zeros with every element replaced by the matching element of x through index_put_ (an exact copy that
    the producer rules do not list as a copy)."""
    x = inp["x"]
    b = torch.zeros_like(x)
    idx = torch.arange(b.numel(), device=x.device)
    b.index_put_((idx,), x[idx])
    return {"y": _affine_of(b)}


def compiled_reader_cross_input(inp):
    """H-P8: .data write of w * (1 + 2^-30) (rounds to w's bytes) into x, then an Inductor-compiled reader."""
    x, w = inp["x"], inp["w"]
    x.data.copy_(w * (1 + 2.0 ** -30))
    return {"y": _compiled_affine(x)}


def compiled_reader_clean(inp):
    """H-P9: an Inductor-compiled reader of the declared input."""
    return {"y": _compiled_affine(inp["x"])}


def inference_input(seed, shape, dtype):
    """State source: the input is an inference-mode tensor (no version counter)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.inference_mode():
        return (torch.rand(tuple(shape), generator=g, dtype=torch.float32) + 1.0).to("cuda")


def inference_clean(inp):
    """H-P10: a kernel reads an inference-mode input (no version evidence)."""
    return {"y": _affine_of(inp["x"])}


def inference_tiny_add(inp):
    """H-P11: x + 2^-25 inside inference mode on an inference-mode input (bytes unchanged, no version counter)."""
    x = inp["x"]
    with torch.inference_mode():
        x.add_(2.0 ** -25)
    return {"y": _affine_of(x)}


def region_write_other_part(inp):
    """H-P12: launch 1 writes all of b; ATen adds 1 to the second half; launch 2 reads only the first half."""
    x = inp["x"]
    n = x.numel()
    b = torch.empty(2 * n, device=x.device, dtype=x.dtype)
    _affine1d[(triton.cdiv(2 * n, BLOCK),)](torch.cat([x, x]), b, 2 * n, BLOCK=BLOCK)
    b[n:].add_(1.0)
    return {"y": _affine_of(b[:n])}


def exact_scale_between_launches(inp):
    """H-P13: t = 3x + 1 (launch 1), t *= 2 (exact in floating point), launch 2 reads t."""
    t = _affine_of(inp["x"])
    t.mul_(2.0)
    return {"y": _affine_of(t)}


# --------------------------------------------------------------------------------------------- precision items

def pairs_rows(seed, shape, e_lo, e_hi, small_exp, rng_tag):
    """Rows of cancelling pairs +-2^e (e in [e_lo, e_hi], one pair at e_hi) plus one small value u 2^small_exp,
    u in [1, 2), shuffled: the exact row sum is the small value."""
    rng = np.random.default_rng([int(seed), rng_tag])
    rows, n = shape
    a = np.zeros(shape)
    for r in range(rows):
        e = rng.integers(e_lo, e_hi + 1, n // 2 - 1)
        e[0] = e_hi
        mags = rng.choice([-1.0, 1.0], n // 2 - 1) * np.exp2(e.astype(np.float64))
        small = float(np.float32(rng.uniform(1.0, 2.0) * 2.0 ** small_exp))
        a[r] = rng.permutation(np.concatenate([np.stack([mags, -mags], 1).reshape(-1), [small, 0.0]]))
    return a


def axis1_rows(seed, shape, dtype):
    """H-R1 source: [8, 256], pairs up to 2^90, small value ~2^-60."""
    return torch.tensor(pairs_rows(seed, shape, 70, 90, -60, 11), dtype=torch.float32, device="cuda")


def sum_axis1(inp):
    x = inp["x"]
    R, C = x.shape
    y = torch.empty(R, device=x.device, dtype=x.dtype)
    _sum_axis1[(1,)](x, y, R=R, C=C)
    return {"y": y}


def cancel_rows_rev(seed, shape, dtype):
    """H-R2 source: alternating +-2^22 plus values in [0.25, 4)."""
    rng = np.random.default_rng([int(seed), 12])
    big = np.where(np.arange(shape[-1]) % 2 == 0, 2.0 ** 22, -2.0 ** 22)
    return torch.tensor(big + rng.uniform(0.25, 4.0, shape), dtype=torch.float32, device="cuda")


def rev_cumsum(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    _rev_cumsum[(x.shape[0],)](x, y, N=x.shape[1])
    return {"y": y}


def exp1d(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    _exp1d[(triton.cdiv(x.numel(), BLOCK),)](x, y, x.numel(), BLOCK=BLOCK)
    return {"y": y}


def row_sum64(inp):
    x = inp["x"]
    y = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    _row_sum64[(x.shape[0],)](x, y, N=x.shape[1])
    return {"y": y}


def contended_int(inp):
    x = inp["x"]
    z = torch.zeros(1, device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)
    _contended_int[(x.numel(),)](x, z, y)
    return {"y": y}


# --------------------------------------------------------------------------------------------- bias items

def rtz_bf16(inp):
    """H-Q1: bf16 rounding toward zero of 3x + 1 > 0: every residual K - G is <= 2^-21 and its mean is about
    -2^-6 (analytic sign: magnitudes pulled toward zero)."""
    x = inp["x"]
    y = torch.empty_like(x)
    _rtz_bf16[(triton.cdiv(x.numel(), BLOCK),)](x, y, x.numel(), BLOCK=BLOCK)
    return {"y": y}


def rne_bf16(inp):
    """H-Q2: bf16 rounding to nearest even of 3x + 1: |K - G| <= 2^-6 + 2^-21 per element."""
    x = inp["x"]
    y = torch.empty_like(x)
    _rne_bf16[(triton.cdiv(x.numel(), BLOCK),)](x, y, x.numel(), BLOCK=BLOCK)
    return {"y": y}
