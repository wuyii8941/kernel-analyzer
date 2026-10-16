"""Adapter for contracts/CAST.yaml. Reference: kernel_spec.aten.convert_element_type on the v2.4 rounding oracle.
The mutants use an independent binary rounding routine written here, each with one misreading."""
from __future__ import annotations

import math
from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.real import NEG, POS, R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CAST.yaml").read_text())["covers"]
reference = make_reference(COVERS)

FMT = {"f32": (24, -126, 127), "f16": (11, -14, 15), "bf16": (8, -126, 127), "f64": (53, -1022, 1023)}


def _round(q: Fraction, d: str, mode="rne", subnormals=True, saturate=False):
    p, emin, emax = FMT[d]
    if q == 0:
        return Fraction(0)
    s, a = (1 if q > 0 else -1), abs(q)
    e = math.floor(math.log2(a))
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    if e < emin and not subnormals:
        return Fraction(0)
    e = max(e, emin)
    ulp = Fraction(2) ** (e - p + 1)
    n, r = divmod(a, ulp)
    if mode == "rne" and (2 * r > ulp or (2 * r == ulp and n % 2 == 1)):
        n += 1
    elif mode == "away" and 2 * r >= ulp:
        n += 1
    v = n * ulp
    vmax = (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax
    if v > vmax:
        return s * vmax if saturate else (POS if s > 0 else NEG)
    return s * v


def _conv(**kw):
    def convert(a, dtype=None):
        a, d = A.as_ten(a), A.dt(dtype)
        if d in FMT:
            return A.unary(lambda x: R(_round(Fraction(x.v) if isinstance(x, R) else Fraction(int(x)), d, **kw)), a,
                           out_dtype=d)
        return A.convert_element_type(a, d)
    return convert


def _int_round_nearest(a, dtype):
    a, d = A.as_ten(a), A.dt(dtype)
    if d in A.INT:
        return A.unary(lambda x: round(Fraction(x.v)), a, out_dtype=d)
    return A.convert_element_type(a, d)


def _int_to_float_trunc(a, dtype):
    a, d = A.as_ten(a), A.dt(dtype)
    if a.dtype in A.INT and d in FMT:
        p = FMT[d][0]

        def tr(x):
            x = int(x)
            n = abs(x).bit_length()
            drop = max(0, n - p)
            return R((abs(x) >> drop << drop) * (1 if x >= 0 else -1))
        return A.unary(tr, a, out_dtype=d)
    return A.convert_element_type(a, d)


def _m(fn):
    return make_reference(COVERS, {"prims.convert_element_type": fn, "aten._to_copy": lambda a, dtype=None, **k: fn(a, dtype)})


MUTANTS = {
    "CA.M1": _m(_conv(mode="trunc")),
    "CA.M2": _m(_conv(mode="away")),
    "CA.M3": _m(_conv(saturate=True)),
    "CA.M4": _m(_int_round_nearest),
    "CA.M5": _m(_conv(subnormals=False)),
    "CA.M6": _m(_int_to_float_trunc),
}
READINGS = {}
