"""Adapter for contracts/ARITH.yaml. The reference is kernel_spec.aten; each mutant replaces one operation by an
independent implementation of the misreading named in the contract."""
from __future__ import annotations

import math
from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "ARITH.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _values(t):
    return [x.v if isinstance(x, R) else x for x in t.a.reshape(-1)]


def _div_floor_as_trunc(a, b, rounding_mode=None):
    if rounding_mode == "floor":
        rounding_mode = "trunc"
    return A.div(a, b, rounding_mode=rounding_mode)


def _rsub_unswapped(a, b, alpha=1):
    x = [Fraction(v) for v in _values(A.as_ten(a))]
    return A.tensor([v - alpha * Fraction(b) for v in x], A.as_ten(a).dtype)


def _clamp_min_wins(a, min=None, max=None):
    x = [Fraction(v) for v in _values(A.as_ten(a))]
    out = []
    for v in x:
        if max is not None:
            v = v if v < Fraction(max) else Fraction(max)
        if min is not None:
            v = v if v > Fraction(min) else Fraction(min)
        out.append(v)
    return A.tensor(out, A.as_ten(a).dtype)


def _sign_zero_one(a):
    return A.tensor([1 if Fraction(v) >= 0 else -1 for v in _values(A.as_ten(a))], A.as_ten(a).dtype)


def _pow_zero_zero(a, b):
    return A.tensor([0 if (Fraction(v) == 0) else Fraction(v) ** int(b) for v in _values(A.as_ten(a))], A.as_ten(a).dtype)


def _add_no_alpha(a, b, alpha=1):
    return A.add(a, b)


def _sub_no_alpha(a, b, alpha=1):
    return A.sub(a, b)


def _div_int_truncates(a, b, rounding_mode=None):
    x, y = _values(A.as_ten(a)), _values(A.as_ten(b))
    return A.tensor([math.trunc(Fraction(u) / Fraction(v)) for u, v in zip(x, y)], "f32")


def _where_swapped(c, a, b):
    return A.where(c, b, a)


def _masked_fill_inverted(a, m, v):
    return A.where(m, a, A.as_ten(v, A.as_ten(a)))


def _mut(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "AR.M1": _mut(**{"aten.div": _div_floor_as_trunc}),
    "AR.M2": _mut(**{"aten.rsub": _rsub_unswapped}),
    "AR.M3": _mut(**{"aten.clamp": _clamp_min_wins}),
    "AR.M4": _mut(**{"aten.sign": _sign_zero_one}),
    "AR.M5": _mut(**{"aten.pow": _pow_zero_zero}),
    "AR.M6": _mut(**{"aten.add": _add_no_alpha, "aten.sub": _sub_no_alpha}),
    "AR.M7": _mut(**{"aten.div": _div_int_truncates}),
    "AR.M8": _mut(**{"aten.where": _where_swapped}),
    "AR.M9": _mut(**{"aten.masked_fill": _masked_fill_inverted}),
}
READINGS = {}
