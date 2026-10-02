"""Vectorized float64 interval arithmetic with rigorous directed rounding.

Endpoints are float64 arrays.  ``+ - * / sqrt`` use error-free
transformations (TwoSum, Dekker's TwoProduct) to decide the rounding
direction exactly, so an exact result keeps width 0 and an inexact one is
widened by one ulp on the correct side only.  Elementary functions call MPFR
at 53-bit precision with directed rounding element by element.  Reductions
and dot products use the classical a-priori bound ``gamma_n * sum |x|``.

All functions assume finite inputs; callers handle special values first.
"""

from __future__ import annotations

import math

import gmpy2
import numpy as np

from .numbers import ELEMENTARY as EXACT_ELEMENTARY
from .numbers import DomainError, Interval as ExactInterval

INF = np.inf
U = 2.0 ** -53
_SPLIT = 134217729.0  # 2**27 + 1
_TINY = 2.0 ** -960
_HUGE = 2.0 ** 996


def down(x):
    return np.nextafter(x, -INF)


def up(x):
    return np.nextafter(x, INF)


def two_sum(a, b):
    s = a + b
    bp = s - a
    ap = s - bp
    return s, (a - ap) + (b - bp)


def _split(a):
    c = _SPLIT * a
    h = c - (c - a)
    return h, a - h


def two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _directed(value, err_sign_down_needed, err_sign_up_needed, unreliable):
    lo = np.where(err_sign_down_needed, down(value), value)
    hi = np.where(err_sign_up_needed, up(value), value)
    lo = np.where(unreliable, down(value), lo)
    hi = np.where(unreliable, up(value), hi)
    return lo, hi


def add_bounds(a, b):
    """(RD(a+b), RU(a+b)) elementwise."""

    with np.errstate(all="ignore"):
        s, e = two_sum(a, b)
        bad = ~np.isfinite(e)
        return _directed(s, e < 0, e > 0, bad)


def mul_bounds(a, b):
    with np.errstate(all="ignore"):
        p, e = two_prod(a, b)
        bad = ~np.isfinite(e) | (np.abs(p) < _TINY) | (np.abs(a) > _HUGE) | (np.abs(b) > _HUGE)
        bad &= (a != 0) & (b != 0)  # a zero factor gives an exact zero product
        return _directed(p, e < 0, e > 0, bad)


def div_bounds(a, b):
    with np.errstate(all="ignore"):
        q = a / b
        p, e = two_prod(q, b)
        r = (a - p) - e
        sign = np.sign(r) * np.sign(b)
        bad = ~np.isfinite(r) | (np.abs(q) < _TINY) | (np.abs(q) > _HUGE) | (np.abs(b) > _HUGE)
        bad &= a != 0  # 0 / b is exactly 0
        return _directed(q, sign < 0, sign > 0, bad)


def sqrt_bounds(a):
    with np.errstate(all="ignore"):
        s = np.sqrt(a)
        p, e = two_prod(s, s)
        r = (a - p) - e
        bad = ~np.isfinite(r) | (s < _TINY)
        lo, hi = _directed(s, r < 0, r > 0, bad)
        return np.where(a == 0, 0.0, lo), np.where(a == 0, 0.0, hi)


# ---------------------------------------------------------------------------
# Interval operations on (lo, hi) pairs
# ---------------------------------------------------------------------------


def iadd(alo, ahi, blo, bhi):
    return add_bounds(alo, blo)[0], add_bounds(ahi, bhi)[1]


def isub(alo, ahi, blo, bhi):
    return add_bounds(alo, -bhi)[0], add_bounds(ahi, -blo)[1]


def imul(alo, ahi, blo, bhi, same=False):
    if same:
        return isquare(alo, ahi)
    cands_lo, cands_hi = [], []
    for x in (alo, ahi):
        for y in (blo, bhi):
            lo, hi = mul_bounds(x, y)
            cands_lo.append(lo)
            cands_hi.append(hi)
    return np.minimum.reduce(cands_lo), np.maximum.reduce(cands_hi)


def isquare(lo, hi):
    l2 = mul_bounds(lo, lo)
    h2 = mul_bounds(hi, hi)
    out_lo = np.minimum(l2[0], h2[0])
    out_hi = np.maximum(l2[1], h2[1])
    straddle = (lo <= 0) & (hi >= 0)
    return np.where(straddle, 0.0, out_lo), out_hi


def idiv(alo, ahi, blo, bhi):
    """Requires 0 not in [blo, bhi] (checked by the caller)."""

    cands_lo, cands_hi = [], []
    for x in (alo, ahi):
        for y in (blo, bhi):
            lo, hi = div_bounds(x, y)
            cands_lo.append(lo)
            cands_hi.append(hi)
    return np.minimum.reduce(cands_lo), np.maximum.reduce(cands_hi)


def isqrt(lo, hi):
    return sqrt_bounds(lo)[0], sqrt_bounds(hi)[1]


def ifma(alo, ahi, blo, bhi, clo, chi):
    point = (alo == ahi) & (blo == bhi) & (clo == chi)
    # Point inputs: a*b + c with p + e exact, then two directed additions.
    with np.errstate(all="ignore"):
        p, e = two_prod(alo, blo)
        bad = ~np.isfinite(e) | (np.abs(p) < _TINY)
        s_lo = add_bounds(add_bounds(p, clo)[0], np.where(bad, 0.0, e))[0]
        s_hi = add_bounds(add_bounds(p, chi)[1], np.where(bad, 0.0, e))[1]
        s_lo = np.where(bad, down(down(s_lo)), s_lo)
        s_hi = np.where(bad, up(up(s_hi)), s_hi)
    mlo, mhi = imul(alo, ahi, blo, bhi)
    glo, ghi = iadd(mlo, mhi, clo, chi)
    return np.where(point, s_lo, glo), np.where(point, s_hi, ghi)


# ---------------------------------------------------------------------------
# Elementary functions via MPFR
# ---------------------------------------------------------------------------

_MONOTONE = {
    "exp": (gmpy2.exp, +1, None, False), "exp2": (gmpy2.exp2, +1, None, False),
    "exp10": (gmpy2.exp10, +1, None, False), "expm1": (gmpy2.expm1, +1, None, False),
    "log": (gmpy2.log, +1, 0.0, True), "log2": (gmpy2.log2, +1, 0.0, True),
    "log10": (gmpy2.log10, +1, 0.0, True), "log1p": (gmpy2.log1p, +1, -1.0, True),
    "sqrt": (gmpy2.sqrt, +1, 0.0, False), "rsqrt": (gmpy2.rec_sqrt, -1, 0.0, True),
    "tanh": (gmpy2.tanh, +1, None, False), "erf": (gmpy2.erf, +1, None, False),
    "erfc": (gmpy2.erfc, -1, None, False), "atan": (gmpy2.atan, +1, None, False),
    "asinh": (gmpy2.asinh, +1, None, False), "sinh": (gmpy2.sinh, +1, None, False),
    "cbrt": (gmpy2.cbrt, +1, None, False), "asin": (gmpy2.asin, +1, -1.0, False),
    "acos": (gmpy2.acos, -1, -1.0, False), "atanh": (gmpy2.atanh, +1, -1.0, True),
    "acosh": (gmpy2.acosh, +1, 1.0, False),
}
_UPPER_DOMAIN = {"asin": (1.0, False), "acos": (1.0, False), "atanh": (1.0, True)}

_CTX_DOWN = gmpy2.context(precision=53, round=gmpy2.RoundDown)
_CTX_UP = gmpy2.context(precision=53, round=gmpy2.RoundUp)
_SUBNORMAL = 2.0 ** -1021


def _mpfr_scalar(fn, x: float, ctx, direction: int) -> float:
    with ctx:
        r = fn(gmpy2.mpfr(x))
    if gmpy2.is_nan(r) or gmpy2.is_infinite(r):
        raise DomainError(f"{fn.__name__}({x}) is not finite")
    f = float(r)  # exact unless the result is subnormal in float64
    if abs(f) < _SUBNORMAL:
        f = math.nextafter(f, -math.inf if direction < 0 else math.inf)
    return f


def elementary_bounds(name: str, lo: np.ndarray, hi: np.ndarray):
    """Rigorous enclosure of ``name`` over [lo, hi]; returns (lo, hi, ok_mask)."""

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    out_lo = np.zeros_like(lo)
    out_hi = np.zeros_like(hi)
    ok = np.ones(lo.shape, dtype=bool)
    if name in _MONOTONE:
        fn, direction, dom_lo, open_lo = _MONOTONE[name]
        upper = _UPPER_DOMAIN.get(name)
        flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
        rlo, rhi, rok = out_lo.reshape(-1), out_hi.reshape(-1), ok.reshape(-1)
        for i in range(flat_lo.size):
            a, b = float(flat_lo[i]), float(flat_hi[i])
            if dom_lo is not None and (a < dom_lo or (open_lo and a == dom_lo)):
                rok[i] = False
                continue
            if upper is not None and (b > upper[0] or (upper[1] and b == upper[0])):
                rok[i] = False
                continue
            try:
                if direction > 0:
                    rlo[i] = _mpfr_scalar(fn, a, _CTX_DOWN, -1)
                    rhi[i] = _mpfr_scalar(fn, b, _CTX_UP, +1)
                else:
                    rlo[i] = _mpfr_scalar(fn, b, _CTX_DOWN, -1)
                    rhi[i] = _mpfr_scalar(fn, a, _CTX_UP, +1)
            except DomainError:
                rok[i] = False
        return out_lo, out_hi, ok
    if name in ("sin", "cos", "tan", "cosh"):
        exact = EXACT_ELEMENTARY.get(name)
        flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
        rlo, rhi, rok = out_lo.reshape(-1), out_hi.reshape(-1), ok.reshape(-1)
        for i in range(flat_lo.size):
            try:
                if name == "cosh":
                    r = _cosh_interval(float(flat_lo[i]), float(flat_hi[i]))
                elif name == "tan":
                    r = _tan_interval(float(flat_lo[i]), float(flat_hi[i]))
                else:
                    r = exact(ExactInterval(float(flat_lo[i]), float(flat_hi[i])))
                rlo[i] = rational_down(r.lo)
                rhi[i] = rational_up(r.hi)
            except DomainError:
                rok[i] = False
        return out_lo, out_hi, ok
    raise KeyError(name)


def _cosh_interval(a: float, b: float) -> ExactInterval:
    from gmpy2 import mpq

    with gmpy2.context(precision=256, round=gmpy2.RoundUp):
        hi = mpq(max(gmpy2.cosh(gmpy2.mpfr(a)), gmpy2.cosh(gmpy2.mpfr(b))))
    with gmpy2.context(precision=256, round=gmpy2.RoundDown):
        lo = mpq(1) if a <= 0 <= b else mpq(min(gmpy2.cosh(gmpy2.mpfr(a)), gmpy2.cosh(gmpy2.mpfr(b))))
    return ExactInterval(lo, hi)


def _tan_interval(a: float, b: float) -> ExactInterval:
    from gmpy2 import mpq

    # tan is increasing between poles at pi/2 + k pi; reject intervals that may contain a pole.
    with gmpy2.context(precision=256):
        pi = gmpy2.const_pi()
        k_a = gmpy2.floor((gmpy2.mpfr(a) - pi / 2) / pi)
        k_b = gmpy2.floor((gmpy2.mpfr(b) - pi / 2) / pi)
    if k_a != k_b:
        raise DomainError("tan interval may contain a pole")
    with gmpy2.context(precision=256, round=gmpy2.RoundDown):
        lo = mpq(gmpy2.tan(gmpy2.mpfr(a)))
    with gmpy2.context(precision=256, round=gmpy2.RoundUp):
        hi = mpq(gmpy2.tan(gmpy2.mpfr(b)))
    return ExactInterval(lo, hi)


def rational_down(q) -> float:
    f = float(q)
    from gmpy2 import mpq

    if mpq(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def rational_up(q) -> float:
    f = float(q)
    from gmpy2 import mpq

    if mpq(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def pow_bounds(xlo, xhi, ylo, yhi):
    """x**y for x > 0 via corners (y*ln x is bilinear on the box); else not ok."""

    shape = np.broadcast(xlo, ylo).shape
    out_lo = np.zeros(shape)
    out_hi = np.zeros(shape)
    ok = np.ones(shape, dtype=bool)
    xlo, xhi, ylo, yhi = (np.broadcast_to(np.asarray(v, dtype=np.float64), shape).reshape(-1)
                          for v in (xlo, xhi, ylo, yhi))
    rlo, rhi, rok = out_lo.reshape(-1), out_hi.reshape(-1), ok.reshape(-1)
    for i in range(xlo.size):
        if xlo[i] <= 0:
            if xlo[i] == xhi[i] == 0 and ylo[i] == yhi[i] and ylo[i] > 0:
                rlo[i] = rhi[i] = 0.0
                continue
            rok[i] = False
            continue
        corners_lo, corners_hi = [], []
        try:
            for x in (float(xlo[i]), float(xhi[i])):
                for y in (float(ylo[i]), float(yhi[i])):
                    with _CTX_DOWN:
                        a = gmpy2.mpfr(x) ** gmpy2.mpfr(y)
                    with _CTX_UP:
                        b = gmpy2.mpfr(x) ** gmpy2.mpfr(y)
                    if gmpy2.is_infinite(a) or gmpy2.is_infinite(b):
                        raise DomainError("pow overflow")
                    corners_lo.append(float(a))
                    corners_hi.append(float(b))
        except DomainError:
            rok[i] = False
            continue
        rlo[i] = math.nextafter(min(corners_lo), -math.inf) if min(corners_lo) < _SUBNORMAL else min(corners_lo)
        rhi[i] = max(corners_hi)
    return out_lo, out_hi, ok


# ---------------------------------------------------------------------------
# Reductions and dot products
# ---------------------------------------------------------------------------


def gamma(n) -> float:
    n = max(int(n), 1)
    return n * U / (1 - n * U)


def isum(lo, hi, axis):
    """Enclosure of the exact sum along ``axis``."""

    n = lo.shape[axis]
    g = gamma(n) * (1 + 4 * U)
    with np.errstate(all="ignore"):
        s_lo = np.sum(lo, axis=axis)
        s_hi = np.sum(hi, axis=axis)
        b_lo = np.sum(np.abs(lo), axis=axis) * g
        b_hi = np.sum(np.abs(hi), axis=axis) * g
    return down(s_lo - b_lo), up(s_hi + b_hi)


def icumsum(lo, hi, axis, reverse=False):
    if reverse:
        lo, hi = np.flip(lo, axis), np.flip(hi, axis)
    n = lo.shape[axis]
    g = gamma(n) * (1 + 4 * U)
    with np.errstate(all="ignore"):
        s_lo = np.cumsum(lo, axis=axis)
        s_hi = np.cumsum(hi, axis=axis)
        b_lo = np.cumsum(np.abs(lo), axis=axis) * g
        b_hi = np.cumsum(np.abs(hi), axis=axis) * g
    out = down(s_lo - b_lo), up(s_hi + b_hi)
    if reverse:
        out = np.flip(out[0], axis), np.flip(out[1], axis)
    return out


def idot(alo, ahi, blo, bhi):
    """Enclosure of A @ B over interval matrices (midpoint-radius form)."""

    k = alo.shape[-1]
    with np.errstate(all="ignore"):
        am = (alo + ahi) * 0.5
        bm = (blo + bhi) * 0.5
        ar = up(np.maximum(up(ahi - am), up(am - alo)))
        br = up(np.maximum(up(bhi - bm), up(bm - blo)))
        point_a = bool(np.all(alo == ahi))
        point_b = bool(np.all(blo == bhi))
        mid = am @ bm
        g = gamma(k + 2) * (1 + 8 * U)
        abs_prod = np.abs(am) @ np.abs(bm)
        rad = abs_prod * g
        if not (point_a and point_b):
            r = (np.abs(am) @ br + ar @ np.abs(bm) + ar @ br) * (1 + g)
            rad = rad + r
        rad = up(rad + 2.0 ** -1070 * k)
    return down(mid - rad), up(mid + rad)


# ---------------------------------------------------------------------------
# Rounding to storage formats (vectorized round-to-nearest-even)
# ---------------------------------------------------------------------------

FLOAT_FORMATS = {
    "f64": (53, -1022, 1023, True),
    "f32": (24, -126, 127, True),
    "f16": (11, -14, 15, True),
    "bf16": (8, -126, 127, True),
    "tf32": (11, -126, 127, True),
    "f8E5M2": (3, -14, 15, True),
}


def round_nearest_even(x: np.ndarray, fmt: str):
    """RN-even of float64 values to ``fmt``; returns (values, overflow_mask)."""

    precision, emin, emax, has_inf = FLOAT_FORMATS[fmt]
    x = np.asarray(x, dtype=np.float64)
    if fmt == "f64":
        return x.copy(), np.zeros(x.shape, dtype=bool)
    with np.errstate(all="ignore"):
        mant, exp = np.frexp(x)  # x = mant * 2**exp, 0.5 <= |mant| < 1
        e = exp - 1  # x = m * 2**e with 1 <= |m| < 2
        e_eff = np.maximum(e, emin)
        scale = (precision - 1) - e_eff
        scaled = np.ldexp(x, scale)
        rounded = np.ldexp(np.rint(scaled), -scale)  # rint is round-half-even
        limit = np.ldexp(1.0, emax + 1)
        overflow = np.abs(rounded) >= limit
    rounded = np.where(x == 0, x, rounded)
    return rounded, overflow & (x != 0)
