"""Conversion of a real value to a floating-point format (the CAST contract), on top of the v2.4 rounding oracle
(specification_v2/oracle_rounding.py, written from the IEEE 754 definitions and cross-checked against numpy).

Round-to-nearest-even is monotone, so an enclosure [a, b] converts to the enclosure [round(a), round(b)]; when both
endpoints round to the same number the result is exact.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from .num import Undefined
from .real import NEG, POS, R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "specification_v2"))
import oracle_rounding as _oracle  # noqa: E402

FMT = {"f64": "f64", "f32": "f32", "f16": "f16", "bf16": "bf16", "f8e4m3": "f8e4m3fn", "f8e5m2": "f8e5m2"}


def _round_exact(q: Fraction, fmt: str):
    v = _oracle.round_to(q, fmt)
    if v == "inf":
        return R(POS)
    if v == "-inf":
        return R(NEG)
    if v == "nan":
        raise Undefined(f"{float(q):.6g} overflows {fmt}, a format without infinities")
    return R(v)


def round_to(x, dtype: str) -> R:
    """x rounded to nearest-even in the format of dtype; overflow gives the infinity of that sign where the format has
    one (IEEE 754), and is undefined where it has none."""
    x = x if isinstance(x, R) else R(x)
    fmt = FMT[dtype]
    if not x.finite:
        if fmt == "f8e4m3fn":
            raise Undefined("an infinity has no value in float8_e4m3fn")
        return x
    a, b = x.bounds()
    ra, rb = _round_exact(a, fmt), _round_exact(b, fmt)
    if a == b or (ra.exact and rb.exact and ra.v == rb.v):
        return ra
    if not (ra.finite and rb.finite):
        from .num import NotEstablished
        raise NotEstablished("enclosure straddles the overflow threshold of the format")
    from mpmath import iv
    from .num import to_iv
    return R(iv.mpf([to_iv(ra.v).a, to_iv(rb.v).b]))
