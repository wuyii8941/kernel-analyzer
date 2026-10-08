"""Rigorous interval arithmetic shared by the phase-2 specs (same design as spec_cross_entropy v0.4, plus sqrt).

Fixed decimal context: PREC digits, Emin/Emax = -/+999999, traps on Overflow/Underflow/Subnormal/Clamped/Invalid/DivByZero.
+,-,*,/ use directed rounding on the endpoints; sqrt() is correctly rounded by decimal and widened by one ulp.
Products/quotients that underflow below the normal range are enclosed by [0, smallest normal] with the exact sign.
Anything else leaving the normal range raises SpecNotEstablished (the spec declines rather than returning a wrong box).
"""
from decimal import (Decimal, Context, localcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN,
                     Overflow, Underflow, Subnormal, Clamped, InvalidOperation, DivisionByZero)
from fractions import Fraction
import math

PREC = 60
_TRAPS = [Overflow, Underflow, Subnormal, Clamped, InvalidOperation, DivisionByZero]
_SMALLEST_NORMAL = Decimal("1E-999999")


def _ctx(rounding=ROUND_HALF_EVEN, prec=PREC):
    c = Context(prec=prec, rounding=rounding, Emin=-999999, Emax=999999)
    for t in _TRAPS:
        c.traps[t] = True
    return c


class SpecNotEstablished(Exception):
    """the spec declines for this condition."""


class SpecInputError(ValueError):
    """illegal or unsupported input: refused, not answered."""


def _ulp(d):
    with localcontext(_ctx()):
        if d == 0:
            raise SpecNotEstablished("correctly rounded result is zero: true value below the normal range")
        return Decimal(10) ** (d.adjusted() - PREC + 1)


def _down(d):
    with localcontext(_ctx(ROUND_FLOOR, PREC + 5)):
        return d - _ulp(d)


def _up(d):
    with localcontext(_ctx(ROUND_CEILING, PREC + 5)):
        return d + _ulp(d)


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = lo
        self.hi = lo if hi is None else hi

    @staticmethod
    def exact(v):
        if isinstance(v, Interval):
            return v
        if isinstance(v, Decimal):
            return Interval(v, v)
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                raise SpecInputError("non-finite value outside the spec's scope")
            d = Decimal(v)
            return Interval(d, d)
        f = Fraction(v)
        n, d = Decimal(f.numerator), Decimal(f.denominator)
        with localcontext(_ctx(ROUND_FLOOR)):
            lo = n / d
        with localcontext(_ctx(ROUND_CEILING)):
            hi = n / d
        return Interval(lo, hi)

    def _bin(self, other, op, signed_tiny=False):
        other = Interval.exact(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]

        def ev(a, b, side):
            try:
                with localcontext(_ctx(ROUND_FLOOR if side == "lo" else ROUND_CEILING)):
                    return op(a, b)
            except (Underflow, Subnormal):
                if not signed_tiny:
                    raise
                sign = (1 if a >= 0 else -1) * (1 if b >= 0 else -1)
                if sign >= 0:
                    return Decimal(0) if side == "lo" else _SMALLEST_NORMAL
                return -_SMALLEST_NORMAL if side == "lo" else Decimal(0)

        return Interval(min(ev(a, b, "lo") for a, b in pairs), max(ev(a, b, "hi") for a, b in pairs))

    def __add__(self, o): return self._bin(o, lambda a, b: a + b)
    def __radd__(self, o): return Interval.exact(o) + self
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __rsub__(self, o): return Interval.exact(o) - self
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b, signed_tiny=True)
    def __rmul__(self, o): return Interval.exact(o) * self

    def __truediv__(self, o):
        o = Interval.exact(o)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("interval division by an interval containing zero")
        return self._bin(o, lambda a, b: a / b, signed_tiny=True)

    def __rtruediv__(self, o): return Interval.exact(o) / self
    def __neg__(self): return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def sqrt(self):
        if self.lo < 0:
            raise SpecNotEstablished("sqrt of an interval with a negative endpoint")
        with localcontext(_ctx()):
            lo, hi = self.lo.sqrt(), self.hi.sqrt()
        lo_w = Decimal(0) if lo == 0 else _down(lo)
        return Interval(lo_w, _up(hi) if hi != 0 else Decimal(0))

    def contains(self, v):
        v = Decimal(v) if not isinstance(v, Decimal) else v
        return self.lo <= v <= self.hi

    def overlaps(self, other):
        return self.lo <= other.hi and other.lo <= self.hi

    def width(self):
        with localcontext(_ctx(ROUND_CEILING, PREC + 5)):
            return self.hi - self.lo

    def mid(self):
        with localcontext(_ctx(ROUND_HALF_EVEN, PREC + 5)):
            return (self.lo + self.hi) / 2

    def __repr__(self): return f"[{self.lo}, {self.hi}]"


def isum(items):
    acc = Interval(Decimal(0), Decimal(0))
    for it in items:
        acc = acc + it
    return acc


def imax(a, b):
    """max of two intervals (monotone): [max(lo), max(hi)]."""
    return Interval(max(a.lo, b.lo), max(a.hi, b.hi))
