"""Adapter for contracts/LAYOUT.yaml. Reference: kernel_spec.aten; each mutant replaces one operation by an
independent implementation of the misreading."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "LAYOUT.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _view_colmajor(a, shape):
    a = A.as_ten(a)
    return A.Ten(np.array(a.a.reshape(-1, order="F")).reshape(shape, order="F"), a.dtype)


def _permute_inverse(a, dims):
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    return A.Ten(np.transpose(A.as_ten(a).a, inv), A.as_ten(a).dtype)


def _slice_no_step(a, dim=0, start=None, end=None, step=1):
    return A.slice_(a, dim, start, end, 1)


def _split_drop_last(a, size, dim=0):
    a = A.as_ten(a)
    n = a.a.shape[dim]
    return [A.slice_(a, dim, i, i + size) for i in range(0, n - size + 1, size)]


def _pad_forward(a, pad, value=0):
    a = A.as_ten(a)
    k = len(pad) // 2
    # apply the pairs to dimensions 0, 1, ... instead of the last dimension backwards
    perm = list(range(a.a.ndim))[::-1]
    t = A.permute(a, perm)
    t = A.constant_pad_nd(t, pad, value)
    return A.permute(t, perm)


def _arange_inclusive(start, end=None, step=1, dtype=None, **kw):
    if end is None:
        start, end = 0, start
    vals, x = [], Fraction(start)
    while x <= Fraction(end):
        vals.append(x)
        x += Fraction(step)
    d = "i64" if all(isinstance(v, int) for v in (start, end, step)) else "f32"
    return A.tensor([int(v) if d == "i64" else v for v in vals], d)


def _linspace_exclusive(start, end, steps, dtype=None, **kw):
    s, e = Fraction(start), Fraction(end)
    return A.tensor([s + i * (e - s) / steps for i in range(steps)], dtype or "f32")


def _triu_flipped(a, diagonal=0):
    return A.triu(a, -diagonal)


def _squeeze_all(a, dims=None):
    return A.squeeze(a)


def _m(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "LY.M1": _m(**{"aten.view": _view_colmajor}),
    "LY.M2": _m(**{"aten.permute": _permute_inverse}),
    "LY.M3": _m(**{"aten.slice": _slice_no_step}),
    "LY.M4": _m(**{"aten.split": _split_drop_last}),
    "LY.M5": _m(**{"aten.constant_pad_nd": _pad_forward}),
    "LY.M6": _m(**{"aten.arange": _arange_inclusive}),
    "LY.M7": _m(**{"aten.linspace": _linspace_exclusive}),
    "LY.M8": _m(**{"aten.triu": _triu_flipped}),
    "LY.M9": _m(**{"aten.squeeze": _squeeze_all}),
}
READINGS = {}
