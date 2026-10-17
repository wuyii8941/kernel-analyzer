"""Shared helpers of group C_act_norm_loss for the cross-entropy contracts CE_EXT and FLCE.

The cross-entropy numerics are the v2.4 phase1 reference (specification_v2/algorithm_specs/phase1/spec_cross_entropy.py,
Decimal intervals with directed rounding), the same code the reviewed CE pilot uses. This module only adds what the
pilot adapter does not do: strict parsing of K-dimensional nested tensors, flattening of the positions (n, d_1..d_K)
into rows of C logits (class axis = dim 1), and the reshaping of the results. Every result is converted exactly into
kernel_spec values (Fraction / mpmath interval) by kernel_spec.v24.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from . import v24
from .num import Invalid, NotEstablished, Undefined, integer, rational, to_iv

ce = v24.module("spec_cross_entropy")


# ------------------------------------------------------------------ strict nested tensors
def tensor(x, what, item=rational):
    """(shape, flat values in C order) of a rectangular nested list; scalars are rank 0. Ragged or empty lists are
    invalid; every element goes through `item` (exact rational or exact integer)."""
    if not isinstance(x, (list, tuple)):
        return (), [item(x, what)]
    if not x:
        raise Invalid(f"{what}: empty dimension")
    subs = [tensor(e, f"{what}[{i}]", item) for i, e in enumerate(x)]
    shape0 = subs[0][0]
    if any(s != shape0 for s, _ in subs):
        raise Invalid(f"{what}: ragged nesting")
    return (len(x),) + shape0, [v for _, f in subs for v in f]


def nest(flat, shape):
    """Nested lists of `flat` (C order) with the given shape; rank 0 gives the single element."""
    if not shape:
        assert len(flat) == 1
        return flat[0]
    step = len(flat) // shape[0]
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def prod(xs):
    out = 1
    for v in xs:
        out *= v
    return out


# ------------------------------------------------------------------ positions <-> rows
def rows_of(flat, N, C, D):
    """Rows of C values for the N*D positions (n, j) of a tensor laid out as (N, C, d_1..d_K), D = prod(d)."""
    return [[flat[(n * C + c) * D + j] for c in range(C)] for n in range(N) for j in range(D)]


def unrows(rows, N, C, D):
    flat = [None] * (N * C * D)
    for n in range(N):
        for j in range(D):
            for c in range(C):
                flat[(n * C + c) * D + j] = rows[n * D + j][c]
    return flat


# ------------------------------------------------------------------ v2.4 cross entropy on rows
def ce_rows(rows, targets, weights, eps, ignore_index, reduction, prob):
    """loss (list for 'none', else scalar) and the gradient rows of that loss w.r.t. the logits (for 'none': of the
    sum of the row losses), as kernel_spec values. Raises Undefined for a zero mean mass and NotEstablished when the
    v2.4 reference declines."""
    z = ce.cross_entropy(rows, targets, weights=weights, ignore_index=-10 ** 9 if prob else ignore_index,
                         reduction=reduction, label_smoothing=eps, reading="R_A", prob_target=prob)
    if not z.get("established", False):
        raise NotEstablished(z.get("reason", "v2.4 cross entropy did not establish an enclosure"))
    if z["nan"] or not z["grad_defined"]:
        raise Undefined("mean normalization mass is zero")
    return v24.tree(z["loss"]), v24.tree(z["grad"])


def scale(g, u):
    """g * u for a kernel_spec value g and an exact rational u (exact stays exact)."""
    if isinstance(g, Fraction):
        return g * u
    return g * to_iv(u)


def iv_sum(xs):
    acc = to_iv(0)
    for v in xs:
        acc = acc + to_iv(v)
    return acc


def check_reduction(red):
    if red not in ("sum", "mean", "none"):
        raise Invalid(f"reduction {red!r}")
    return red


def check_eps(e):
    eps = rational(e, "eps")
    if not 0 <= eps <= 1:
        raise Invalid("eps (label_smoothing) outside [0, 1]")
    return eps


def index_targets(t, shape, C, ii, what="target"):
    tshape, tflat = tensor(t, what, item=integer)
    if tshape != shape:
        raise Invalid(f"{what}: shape {tshape}, expected {shape}")
    for v in tflat:
        if v != ii and not 0 <= v < C:
            raise Invalid(f"{what} {v} outside [0, {C}) and not the ignore_index")
    return tflat


__all__ = ["tensor", "nest", "prod", "rows_of", "unrows", "ce_rows", "scale", "iv_sum", "check_reduction",
           "check_eps", "index_targets", "iv"]
