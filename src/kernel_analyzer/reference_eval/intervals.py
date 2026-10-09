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


def _mpfr_point(fn, x: float):
    """Enclosure of fn(x) at a point with one MPFR call.

    Rounding down gives the largest float64 <= fn(x); if MPFR reports the
    result inexact, fn(x) lies strictly between it and the next float64.
    """

    with _CTX_POINT:
        active = gmpy2.get_context()
        active.clear_flags()
        r = fn(gmpy2.mpfr(x))
        inexact = active.inexact
    if gmpy2.is_nan(r) or gmpy2.is_infinite(r):
        raise DomainError(f"{fn.__name__}({x}) is not finite")
    f = float(r)
    if not inexact and (f == 0 or abs(f) >= _SUBNORMAL):
        return f, f
    if abs(f) < _SUBNORMAL:  # float() of a subnormal result is round-to-nearest: widen both sides
        return math.nextafter(f, -math.inf), math.nextafter(f, math.inf)
    return f, math.nextafter(f, math.inf)


_CTX_POINT = gmpy2.context(precision=53, round=gmpy2.RoundDown)


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
                if a == b:
                    rlo[i], rhi[i] = _mpfr_point(fn, a)
                elif direction > 0:
                    rlo[i] = _mpfr_scalar(fn, a, _CTX_DOWN, -1)
                    rhi[i] = _mpfr_scalar(fn, b, _CTX_UP, +1)
                else:
                    rlo[i] = _mpfr_scalar(fn, b, _CTX_DOWN, -1)
                    rhi[i] = _mpfr_scalar(fn, a, _CTX_UP, +1)
            except (DomainError, OverflowError):  # 4.0: a result beyond the float64 range is not established
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
            except (DomainError, OverflowError):  # 4.0: a result beyond the float64 range is not established
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
        rhi[i] = math.nextafter(max(corners_hi), math.inf) if abs(max(corners_hi)) < _SUBNORMAL else max(corners_hi)
    return out_lo, out_hi, ok


# ---------------------------------------------------------------------------
# Reductions and dot products
# ---------------------------------------------------------------------------


def gamma(n) -> float:
    n = max(int(n), 1)
    return n * U / (1 - n * U)


def _vecsum_last(p):
    """One error-free transformation pass along the last axis (Ogita, Rump, Oishi 2005, Algorithm 4.3): afterwards
    p[..., -1] is the floating sum and the other entries carry the exact errors; the exact sum is unchanged."""

    for i in range(1, p.shape[-1]):
        s, e = two_sum(p[..., i], p[..., i - 1])
        p[..., i] = s
        p[..., i - 1] = e
    return p


def sum_k(x, axis=-1, K=3):
    """SumK (Ogita, Rump, Oishi 2005, Algorithm 4.8): (res, err) with |sum(x) - res| <= err rigorously
    (Proposition 4.10: |res - s| <= (u + 3 gamma_{n-1}^2) |s| + gamma_{2n-2}^K S, S = sum |x_i|; valid without
    overflow and with gradual underflow, 4 n u <= 1).  Final pass: numpy's pairwise sum of the first n - 1 terms
    (error within the gamma bound of recursive summation), then p_n added last with one rounding, which is the
    structure the proof uses.  Returns (res, err, ok): ok is False where a non-finite value appeared (callers fall
    back to the gamma bound there)."""

    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = x.shape[-1]
    if n == 0:
        z = np.zeros(x.shape[:-1])
        return z, z, np.ones(x.shape[:-1], dtype=bool)
    with np.errstate(all="ignore"):
        S = up(np.sum(np.abs(x), axis=-1) * (1 + gamma(n) * (1 + 4 * U)))
        p = x.copy()
        for _ in range(K - 1):
            _vecsum_last(p)
        # the proof needs the dominant term p_n added last, with a single rounding: the first n - 1 terms in any
        # order (pairwise error <= gamma_{n-2} sum |p_i|), then + p_n.  A pairwise sum over all n terms buries p_n
        # several roundings deep and breaks the u|s| term (audit 2026-10-08, counterexample in tests).
        res = np.sum(p[..., :-1], axis=-1) + p[..., -1]
        g1 = gamma(max(n - 1, 1))
        gk = gamma(max(2 * n - 2, 1)) ** K
        c1 = U + 3 * g1 * g1
        tail = gk * S
        abs_s = (np.abs(res) + tail) / (1 - c1)
        err = up((c1 * abs_s + tail) * (1 + 8 * U))
        ok = np.isfinite(res) & np.isfinite(err) & np.all(np.isfinite(p), axis=-1)
    return res, err, ok


def _accumulation_mode():
    """KA_ACCUMULATION=gamma restores the tool-2.3 gamma_n bounds (for the before / after width comparison only)."""
    import os
    return os.environ.get("KA_ACCUMULATION", "exact")


def isum(lo, hi, axis):
    """Enclosure of the exact sum along ``axis``: SumK on each endpoint array (accumulation adds only the last-bit
    rounding of the result; tool 3.0); the gamma_n bound where SumK met a non-finite value."""

    if _accumulation_mode() == "gamma":
        return isum_gamma(lo, hi, axis)
    r_lo, e_lo, ok_lo = sum_k(lo, axis)
    r_hi, e_hi, ok_hi = sum_k(hi, axis)
    g_lo, g_hi = isum_gamma(lo, hi, axis)
    with np.errstate(all="ignore"):
        out_lo = np.where(ok_lo, down(r_lo - e_lo), g_lo)
        out_hi = np.where(ok_hi, up(r_hi + e_hi), g_hi)
    return np.maximum(out_lo, g_lo), np.minimum(out_hi, g_hi)


def isum_gamma(lo, hi, axis):
    """Enclosure of the exact sum along ``axis`` by the gamma_n bound (tool <= 2.3)."""

    n = lo.shape[axis]
    g = gamma(n) * (1 + 4 * U)
    with np.errstate(all="ignore"):
        s_lo = np.sum(lo, axis=axis)
        s_hi = np.sum(hi, axis=axis)
        b_lo = np.sum(np.abs(lo), axis=axis) * g
        b_hi = np.sum(np.abs(hi), axis=axis) * g
    return down(s_lo - b_lo), up(s_hi + b_hi)


def fsum_bounds(lo, hi, axis=-1):
    """Strict outward enclosure of the exact sums of the endpoint arrays along ``axis``: each endpoint array
    is summed exactly (math.fsum, correctly rounded) and the result moved one ulp outward."""

    lo_m = np.moveaxis(np.asarray(lo, dtype=np.float64), axis, -1)
    hi_m = np.moveaxis(np.asarray(hi, dtype=np.float64), axis, -1)
    shape = lo_m.shape[:-1]
    flat_lo = lo_m.reshape(-1, lo_m.shape[-1])
    flat_hi = hi_m.reshape(-1, hi_m.shape[-1])
    s_lo = np.array([math.fsum(r.tolist()) for r in flat_lo], dtype=np.float64).reshape(shape)
    s_hi = np.array([math.fsum(r.tolist()) for r in flat_hi], dtype=np.float64).reshape(shape)
    return np.nextafter(s_lo, -np.inf), np.nextafter(s_hi, np.inf)


def project_bounds(lo, hi, w):
    """Strict outward enclosure of sum(e * w) over the last axis for every e in the box [lo, hi] and the
    finite direction w (broadcast against lo): directed products, then :func:`fsum_bounds`."""

    w = np.broadcast_to(np.asarray(w, dtype=np.float64), np.shape(lo))
    a_lo, a_hi = mul_bounds(np.asarray(lo, dtype=np.float64), w)
    b_lo, b_hi = mul_bounds(np.asarray(hi, dtype=np.float64), w)
    return fsum_bounds(np.minimum(a_lo, b_lo), np.maximum(a_hi, b_hi), axis=-1)


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


def dot_k(a, b, K=3, max_elems=1 << 24):
    """DotK: (res, err, ok) for the exact A @ B of float64 matrices (..., m, k) @ (..., k, n): every product split
    exactly by two_prod, the 2k terms summed by SumK.  Underflowing products are not error-free in two_prod; the
    caller adds k * 2^-1070 (as the gamma version does).  ok False where a non-finite value appeared."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    if a.ndim != 2 or b.ndim != 2 or m * k * n > max_elems:
        return None
    with np.errstate(all="ignore"):
        prod = a[:, :, None] * b[None, :, :]
        _, e = two_prod(a[:, :, None], b[None, :, :])
        terms = np.concatenate([prod, e], axis=1)            # (m, 2k, n)
        res, err, ok = sum_k(terms, axis=1, K=K)
        ok &= np.all(np.isfinite(e), axis=1)
    return res, err, ok


def idot(alo, ahi, blo, bhi):
    """Enclosure of A @ B over interval matrices (midpoint-radius form); the midpoint product by DotK where the
    shapes allow (tool 3.0), the gamma bound otherwise."""

    out = idot_gamma(alo, ahi, blo, bhi)
    if np.asarray(alo).ndim != 2 or np.asarray(blo).ndim != 2 or _accumulation_mode() == "gamma":
        return out
    k = alo.shape[-1]
    with np.errstate(all="ignore"):
        am = (alo + ahi) * 0.5
        bm = (blo + bhi) * 0.5
        point = bool(np.all(alo == ahi)) and bool(np.all(blo == bhi)) and bool(np.all(am == alo)) and \
            bool(np.all(bm == blo))
        dk = dot_k(am, bm)
        if dk is None:
            return out
        res, err, ok = dk
        rad = err + 2.0 ** -1070 * k
        if not point:
            ar = up(np.maximum(up(ahi - am), up(am - alo)))
            br = up(np.maximum(up(bhi - bm), up(bm - blo)))
            g = gamma(k + 2) * (1 + 8 * U)
            rad = rad + (np.abs(am) @ br + ar @ np.abs(bm) + ar @ br) * (1 + g)
        rad = up(rad)
        lo = np.where(ok, down(res - rad), out[0])
        hi = np.where(ok, up(res + rad), out[1])
    return np.maximum(lo, out[0]), np.minimum(hi, out[1])


def idot_gamma(alo, ahi, blo, bhi):
    """Enclosure of A @ B over interval matrices (midpoint-radius form; tool <= 2.3)."""

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

# fmt -> (precision including the implicit bit, minimum normal exponent, maximum exponent, has infinity)
FLOAT_FORMATS = {
    "f64": (53, -1022, 1023, True),
    "f32": (24, -126, 127, True),
    "f16": (11, -14, 15, True),
    "bf16": (8, -126, 127, True),
    "tf32": (11, -126, 127, True),
    "f8E5M2": (3, -14, 15, True),
    # OCP / MLIR finite formats: no infinity; the largest exponent field also holds normal values,
    # so the largest finite value is set explicitly below (the remaining encodings are NaN).
    "f8E4M3FN": (4, -6, 8, False),
    "f8E4M3FNUZ": (4, -7, 7, False),
    "f8E5M2FNUZ": (3, -15, 15, False),
    "f8E4M3B11FNUZ": (4, -10, 4, False),
    "f4E2M1FN": (2, 0, 2, False),
}
# Largest finite value where it differs from (2 - 2**(1 - precision)) * 2**emax.
_MAX_FINITE = {"f8E4M3FN": 448.0}


def max_finite(fmt: str) -> float:
    precision, _, emax, _ = FLOAT_FORMATS[fmt]
    return _MAX_FINITE.get(fmt, (2.0 - 2.0 ** (1 - precision)) * 2.0 ** emax)


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
        overflow = np.abs(rounded) > max_finite(fmt)
    rounded = np.where(x == 0, x, rounded)
    return rounded, overflow & (x != 0)


def round_directed(x: np.ndarray, fmt: str, mode: str):
    """Round float64 values to ``fmt`` in IEEE mode ``mode``.

    mode: "rtne" (nearest even), "rtz" (toward zero), "rd" (toward -inf), "ru" (toward +inf).
    Returns (values, positive_infinity_mask, negative_infinity_mask); finite overflow results
    saturate at the largest finite value as IEEE prescribes for the directed modes.  For a format
    without infinity the two masks mark every overflow beyond the largest finite value: whether the
    implementation saturates or produces NaN is not part of the format, so callers treat it as
    not established.
    """

    precision, emin, emax, has_inf = FLOAT_FORMATS[fmt]
    x = np.asarray(x, dtype=np.float64)
    if mode == "rtne":
        values, overflow = round_nearest_even(x, fmt)
        return values, overflow & (x > 0), overflow & (x < 0)
    fn = {"rtz": np.trunc, "rd": np.floor, "ru": np.ceil}[mode]
    with np.errstate(all="ignore"):
        _, exp = np.frexp(x)
        scale = (precision - 1) - np.maximum(exp - 1, emin)
        rounded = np.ldexp(fn(np.ldexp(x, scale)), -scale)
    rounded = np.where(x == 0, x, rounded)
    top = max_finite(fmt)
    big = np.abs(rounded) > top
    if has_inf:
        pos_inf = big & (rounded > 0) & (mode == "ru")
        neg_inf = big & (rounded < 0) & (mode == "rd")
    else:
        pos_inf, neg_inf = big & (rounded > 0), big & (rounded < 0)
    rounded = np.where(big & ~pos_inf & ~neg_inf, np.sign(rounded) * top, rounded)
    return rounded, pos_inf, neg_inf
