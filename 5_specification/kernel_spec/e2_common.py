"""Helpers shared by the E2_program contracts (POOL, INDEXREDUCE, SCHED, CLIP, SCALER, ACCUM, LABELS, CKPT).

Tensor inputs are nested lists whose leaves are integers, 'p/q' strings, 'inf', '-inf' or 'nan' (the last only where a
contract admits NaN inputs). Values become `real.R` (exact rationals or +-inf); NaN becomes the marker NAN, which is
not a real number and never takes part in real arithmetic. Shapes are checked strictly (no ragged lists, no silent
truncation). Outputs are plain values for the checker: Fractions, '+inf' / '-inf', frozensets (allowed sets), and
frozenset() for an element at which a declared IEEE profile requires NaN (no real value is admissible there).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .num import Invalid, integer, rational
from .real import NEG, POS, R


class _NaN:
    __slots__ = ()

    def __repr__(self):
        return "NaN"


NAN = _NaN()
NAN_REQUIRED = frozenset()          # output element: NaN required by the profile (empty set of admissible reals)


def is_nan(v):
    return v is NAN


def scalar(x, what="value", allow_nan=False):
    """One tensor element: R (exact rational or infinity) or NAN."""
    if isinstance(x, bool):
        raise Invalid(f"{what}: boolean is not a number")
    if isinstance(x, str):
        s = x.strip()
        if s in ("inf", "+inf"):
            return R(POS)
        if s == "-inf":
            return R(NEG)
        if s == "nan":
            if not allow_nan:
                raise Invalid(f"{what}: NaN is not admitted here")
            return NAN
    if isinstance(x, float) and x != x:
        if not allow_nan:
            raise Invalid(f"{what}: NaN is not admitted here")
        return NAN
    if isinstance(x, float) and x in (float("inf"), float("-inf")):
        return R(POS if x > 0 else NEG)
    return R(rational(x, what))


def shape_of(xs, what="tensor"):
    shape = []
    t = xs
    while isinstance(t, (list, tuple)):
        shape.append(len(t))
        if not t:
            break
        t = t[0]
    return tuple(shape)


def tensor(xs, what="tensor", ndim=None, allow_nan=False, item=None):
    """numpy object array of R / NAN (or of item(x) when given) with a strictly rectangular shape."""
    if not isinstance(xs, (list, tuple)):
        raise Invalid(f"{what}: not a nested list")
    shape = shape_of(xs)
    if ndim is not None and len(shape) not in (ndim if isinstance(ndim, (tuple, list, set)) else (ndim,)):
        raise Invalid(f"{what}: {len(shape)} dimensions, expected {ndim}")
    out = np.empty(shape, dtype=object)

    def fill(node, idx, depth):
        if depth == len(shape):
            if isinstance(node, (list, tuple)):
                raise Invalid(f"{what}: ragged nesting at {idx}")
            out[idx] = item(node, f"{what}{list(idx)}") if item else scalar(node, f"{what}{list(idx)}", allow_nan)
            return
        if not isinstance(node, (list, tuple)) or len(node) != shape[depth]:
            raise Invalid(f"{what}: ragged at {idx}")
        for i, v in enumerate(node):
            fill(v, idx + (i,), depth + 1)

    fill(xs, (), 0)
    return out


def plain(v):
    """Checker value of one element."""
    if v is NAN:
        return NAN_REQUIRED
    if isinstance(v, R):
        return v.v
    if isinstance(v, (int, Fraction)) and not isinstance(v, bool):
        return Fraction(v)
    return v


def nested(arr, fn=plain):
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            return fn(arr[()])
        return [nested(a, fn) for a in arr]
    return fn(arr)


def ints(v, n, what, lo=None):
    """An int or a one-element / n-element sequence of ints as an n-tuple (one element repeated)."""
    if isinstance(v, (list, tuple)):
        vals = [integer(x, what, lo) for x in v]
        if len(vals) == 1:
            vals = vals * n
        if len(vals) != n:
            raise Invalid(f"{what}: {len(v)} values for {n} dimensions")
        return tuple(vals)
    return (integer(v, what, lo),) * n


def flag(v, what):
    if not isinstance(v, bool):
        raise Invalid(f"{what}: expected a boolean, got {v!r}")
    return v
