"""Interval helpers on mpmath.iv (200-bit, ~60 decimal digits) for the phase-2 specs that need cos/sin/tanh/erf.

Rigorous (mpmath.iv): +,-,*,/, exp, log, sqrt, cos, sin.
tanh(x) = (e^{2x} - 1)/(e^{2x} + 1) built from iv.exp (monotone, rigorous). For large |x| the quotient is evaluated as
sign(x) * (1 - 2/(e^{2|x|} + 1)) to avoid overflow.
erf: mpmath.iv has no erf. erf(x) is computed as a 120-digit point value and widened by 10^-60 * (1 + |erf|);
this is a declared high-precision approximation, NOT a rigorous enclosure, and every spec using it says so.
Values are converted from exact rationals via numerator/denominator intervals (rigorous).
"""
from fractions import Fraction as F
import mpmath as mp

iv = mp.iv
iv.prec = 200
mp.mp.dps = 120


class SpecNotEstablished(Exception):
    pass


class SpecInputError(ValueError):
    pass


_IVTYPE = type(iv.mpf(0))


def I(v):
    """exact rational / int / float / interval -> mpmath interval enclosing it."""
    if isinstance(v, _IVTYPE):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise SpecInputError("non-finite input outside the spec's scope")
        return iv.mpf(v)                                  # floats are exact in binary
    if isinstance(v, int):
        return iv.mpf(v)
    f = F(v)
    return iv.mpf(f.numerator) / iv.mpf(f.denominator)


def isum(items):
    acc = iv.mpf(0)
    for it in items:
        acc = acc + it
    return acc


def tanh(x):
    x = I(x)
    if x.b <= 0:
        e = iv.exp(2 * x)
        return (e - 1) / (e + 1)
    if x.a >= 0:
        e = iv.exp(-2 * x)
        return (1 - e) / (1 + e)
    lo = tanh(iv.mpf([x.a, 0])); hi = tanh(iv.mpf([0, x.b]))
    return iv.mpf([lo.a, hi.b])


def sigmoid(x):
    x = I(x)
    if x.a >= 0:
        e = iv.exp(-x)
        return 1 / (1 + e)
    if x.b <= 0:
        e = iv.exp(x)
        return e / (1 + e)
    lo = sigmoid(iv.mpf([x.a, 0])); hi = sigmoid(iv.mpf([0, x.b]))
    return iv.mpf([lo.a, hi.b])


def erf_approx(x):
    """point value at 120 digits widened by 1e-60 relative: declared approximation, not an enclosure."""
    x = I(x)
    lo = mp.erf(mp.mpf(x.a)); hi = mp.erf(mp.mpf(x.b))
    pad = mp.mpf(10) ** -60
    return iv.mpf([lo - pad * (1 + abs(lo)), hi + pad * (1 + abs(hi))])


def contains(x, v):
    if isinstance(v, F):
        v = mp.mpf(v.numerator) / mp.mpf(v.denominator)
    return x.a <= mp.mpf(v) <= x.b


def overlaps(x, y):
    return x.a <= y.b and y.a <= x.b


def softmax_row(xs):
    """rigorous softmax / log-softmax of a row of intervals: stable form with m = max of upper bounds."""
    xs = [I(v) for v in xs]
    m = max(v.b for v in xs)
    ts = [v - m for v in xs]
    es = [iv.exp(t) for t in ts]
    s = isum(es)
    lns = iv.log(s)
    return [e / s for e in es], [t - lns for t in ts]
