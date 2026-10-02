"""Exact and enclosing numbers for the reference evaluator.

Point inputs of ``+ - * /`` and ``fma`` are binary rationals, so they are
evaluated exactly with ``gmpy2.mpq`` (width 0).  Elementary functions are
enclosed by MPFR evaluations rounded down and up; their results are intervals
even for point inputs, because sqrt(2) or e is not a rational number.

Every interval operation returns a superset of ``{g(x) : x in I}``.  This is
the inclusion property the composed reference relies on.
"""

from __future__ import annotations

from typing import Callable, Union

import gmpy2
from gmpy2 import mpfr, mpq

# Working precision of MPFR enclosures.  It only affects interval width, never
# the validity of the enclosure.
MPFR_PRECISION = 256


class DomainError(ValueError):
    """The real-valued semantics is undefined on (part of) the input."""


class Interval:
    """Closed interval ``[lo, hi]`` with exact rational endpoints."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        lo = _to_mpq(lo)
        hi = lo if hi is None else _to_mpq(hi)
        if lo > hi:
            raise ValueError(f"empty interval [{lo}, {hi}]")
        self.lo = lo
        self.hi = hi

    @property
    def width(self):
        return self.hi - self.lo

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    def contains(self, value) -> bool:
        value = _to_mpq(value)
        return self.lo <= value <= self.hi

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __add__(self, other):
        other = as_interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        other = as_interval(other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        other = as_interval(other)
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval(min(products), max(products))

    def __truediv__(self, other):
        other = as_interval(other)
        if other.contains_zero():
            raise DomainError("division by an interval containing zero")
        return self * Interval(1 / other.hi, 1 / other.lo)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return as_interval(other) - self

    def square(self) -> "Interval":
        """x*x without the dependency problem of ``self * self``."""

        lo2, hi2 = self.lo * self.lo, self.hi * self.hi
        if self.contains_zero():
            return Interval(0, max(lo2, hi2))
        return Interval(min(lo2, hi2), max(lo2, hi2))

    def __eq__(self, other):
        return isinstance(other, Interval) and self.lo == other.lo and self.hi == other.hi

    def __hash__(self):
        return hash((self.lo, self.hi))

    def __repr__(self):
        if self.is_point:
            return f"Interval({float(self.lo)!r})"
        return f"Interval[{float(self.lo)!r}, {float(self.hi)!r}]"


def _to_mpq(value):
    if isinstance(value, Interval):
        raise TypeError("expected a number, got an interval")
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise DomainError(f"{value} is not a real number")
    # gmpy2 converts binary floats exactly.
    return mpq(value)


def as_interval(value) -> Interval:
    return value if isinstance(value, Interval) else Interval(value)


Number = Union[Interval, int, float]


# ---------------------------------------------------------------------------
# Elementary functions
# ---------------------------------------------------------------------------


def _mpfr_eval(fn: Callable, q, round_mode):
    context = gmpy2.context(precision=MPFR_PRECISION, round=round_mode)
    with context:
        result = fn(mpfr(q))
    if gmpy2.is_nan(result) or gmpy2.is_infinite(result):
        raise DomainError(f"{fn.__name__} has no finite real value at {q}")
    return mpq(result)


def _increasing(fn: Callable, domain_lo=None, open_lo=False) -> Callable[[Interval], Interval]:
    """Enclosure of a non-decreasing function on an interval.

    For a non-decreasing ``f``, ``RD(f(RD(lo)))`` is a lower bound: the
    rounded-down argument is at most ``lo`` and MPFR's directed rounding is
    correct.  The upper bound is symmetric.
    """

    def enclosure(x: Interval) -> Interval:
        if domain_lo is not None:
            if x.lo < domain_lo or (open_lo and x.lo == domain_lo):
                raise DomainError(f"{fn.__name__} undefined below its domain at {x}")
        return Interval(_mpfr_eval(fn, x.lo, gmpy2.RoundDown), _mpfr_eval(fn, x.hi, gmpy2.RoundUp))

    enclosure.__name__ = fn.__name__
    return enclosure


def _decreasing(fn: Callable, domain_lo=None, open_lo=False) -> Callable[[Interval], Interval]:
    def enclosure(x: Interval) -> Interval:
        if domain_lo is not None:
            if x.lo < domain_lo or (open_lo and x.lo == domain_lo):
                raise DomainError(f"{fn.__name__} undefined below its domain at {x}")
        return Interval(_mpfr_eval(fn, x.hi, gmpy2.RoundDown), _mpfr_eval(fn, x.lo, gmpy2.RoundUp))

    enclosure.__name__ = fn.__name__
    return enclosure


def _pi_bounds():
    lo = _mpfr_eval(lambda _: gmpy2.const_pi(), 0, gmpy2.RoundDown)
    hi = _mpfr_eval(lambda _: gmpy2.const_pi(), 0, gmpy2.RoundUp)
    return lo, hi


def _periodic_extremum_inside(x: Interval, phase_num: int, phase_den: int) -> bool:
    """Whether ``x`` may contain a point ``pi * (phase + 2k)``.

    ``phase = phase_num / phase_den``.  Uses a rigorous enclosure of pi, so the
    answer can be a false positive (wider interval) but never a false negative.
    """

    pi_lo, pi_hi = _pi_bounds()
    phase = mpq(phase_num, phase_den)
    # Candidate k values bracket x / (2 pi) - phase / 2.
    k_lo = gmpy2.floor(x.lo / (2 * pi_hi) - phase / 2) - 1
    k_hi = gmpy2.ceil(x.hi / (2 * pi_lo) - phase / 2) + 1
    k = int(k_lo)
    while k <= int(k_hi):
        factor = phase + 2 * k
        a, b = sorted((factor * pi_lo, factor * pi_hi))
        if a <= x.hi and b >= x.lo:
            return True
        k += 1
    return False


def _lipschitz_endpoint(fn: Callable, q) -> Interval:
    """Enclose ``fn(q)`` for a 1-Lipschitz ``fn`` even if ``q`` is not an MPFR number."""

    q_lo = _mpfr_eval(lambda t: t, q, gmpy2.RoundDown)
    q_hi = _mpfr_eval(lambda t: t, q, gmpy2.RoundUp)
    slack = q_hi - q_lo
    values = (
        _mpfr_eval(fn, q_lo, gmpy2.RoundDown),
        _mpfr_eval(fn, q_lo, gmpy2.RoundUp),
        _mpfr_eval(fn, q_hi, gmpy2.RoundDown),
        _mpfr_eval(fn, q_hi, gmpy2.RoundUp),
    )
    return Interval(min(values) - slack, max(values) + slack)


def _sin_cos(fn: Callable, max_phase, min_phase) -> Callable[[Interval], Interval]:
    """Enclosure of sin/cos, including extrema inside the interval."""

    def enclosure(x: Interval) -> Interval:
        pi_lo, _ = _pi_bounds()
        if x.width >= 2 * pi_lo:
            return Interval(-1, 1)
        result = _lipschitz_endpoint(fn, x.lo).hull(_lipschitz_endpoint(fn, x.hi))
        lo, hi = result.lo, result.hi
        if _periodic_extremum_inside(x, *max_phase):
            hi = mpq(1)
        if _periodic_extremum_inside(x, *min_phase):
            lo = mpq(-1)
        return Interval(max(lo, -1), min(hi, 1))

    enclosure.__name__ = fn.__name__
    return enclosure


def _rsqrt(t):
    return gmpy2.rec_sqrt(t)


_rsqrt.__name__ = "rsqrt"

ELEMENTARY = {
    "exp": _increasing(gmpy2.exp),
    "exp2": _increasing(gmpy2.exp2),
    "log": _increasing(gmpy2.log, domain_lo=0, open_lo=True),
    "log2": _increasing(gmpy2.log2, domain_lo=0, open_lo=True),
    "log1p": _increasing(gmpy2.log1p, domain_lo=-1, open_lo=True),
    "sqrt": _increasing(gmpy2.sqrt, domain_lo=0),
    "rsqrt": _decreasing(_rsqrt, domain_lo=0, open_lo=True),
    "tanh": _increasing(gmpy2.tanh),
    "erf": _increasing(gmpy2.erf),
    "atan": _increasing(gmpy2.atan),
    # sin has maxima at pi/2 + 2k pi and minima at -pi/2 + 2k pi; cos at 2k pi and pi + 2k pi.
    "sin": _sin_cos(gmpy2.sin, (1, 2), (-1, 2)),
    "cos": _sin_cos(gmpy2.cos, (0, 1), (1, 1)),
}


# Derivatives used by forward-mode differentiation, expressed with the same
# enclosures so tangents are intervals as well.
def elementary_derivative(name: str, x: Interval) -> Interval:
    if name == "exp":
        return ELEMENTARY["exp"](x)
    if name == "exp2":
        ln2 = ELEMENTARY["log"](Interval(2))
        return ELEMENTARY["exp2"](x) * ln2
    if name == "log":
        return Interval(1) / x
    if name == "log2":
        return Interval(1) / (x * ELEMENTARY["log"](Interval(2)))
    if name == "log1p":
        return Interval(1) / (x + 1)
    if name == "sqrt":
        if x.lo <= 0:
            raise DomainError("sqrt is not differentiable at 0")
        return Interval(1, 2) / ELEMENTARY["sqrt"](x) * Interval(mpq(1, 2))
    if name == "rsqrt":
        r = ELEMENTARY["rsqrt"](x)
        return -(r * r * r) * Interval(mpq(1, 2))
    if name == "tanh":
        t = ELEMENTARY["tanh"](x)
        return Interval(1) - t.square()
    if name == "sin":
        return ELEMENTARY["cos"](x)
    if name == "cos":
        return -ELEMENTARY["sin"](x)
    if name == "atan":
        return Interval(1) / (Interval(1) + x.square())
    if name == "erf":
        pi_lo, pi_hi = _pi_bounds()
        two_over_sqrt_pi = Interval(2) / ELEMENTARY["sqrt"](Interval(pi_lo, pi_hi))
        return two_over_sqrt_pi * ELEMENTARY["exp"](-x.square())
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Rounding to binary formats
# ---------------------------------------------------------------------------

# (precision including the implicit bit, minimum normal exponent, maximum exponent)
FORMATS = {
    "fp64": (53, -1022, 1023),
    "fp32": (24, -126, 127),
    "tf32": (11, -126, 127),
    "fp16": (11, -14, 15),
    "bf16": (8, -126, 127),
}


class Overflow(ArithmeticError):
    """Round-to-nearest overflowed to an infinity."""

    def __init__(self, sign: int):
        super().__init__("overflow to infinity")
        self.sign = sign


def _floor_log2(a) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if mpq(2) ** e > a:
        e -= 1
    elif mpq(2) ** (e + 1) <= a:
        e += 1
    return e


def round_nearest_even(value, fmt: str):
    """Round a rational to ``fmt`` with IEEE round-to-nearest-even.

    Subnormals are kept (no flush to zero).  Raises :class:`Overflow` when the
    rounded value is not finite.
    """

    precision, emin, emax = FORMATS[fmt]
    q = _to_mpq(value)
    if q == 0:
        return mpq(0)
    sign = -1 if q < 0 else 1
    a = abs(q)
    exponent = max(_floor_log2(a), emin)
    ulp = mpq(2) ** (exponent - (precision - 1))
    scaled = a / ulp
    n = gmpy2.floor(scaled)
    remainder = scaled - n
    n = int(n)
    if remainder > mpq(1, 2) or (remainder == mpq(1, 2) and n % 2 == 1):
        n += 1
    result = n * ulp
    if result >= mpq(2) ** (emax + 1):
        raise Overflow(sign)
    return sign * result


def round_interval(x: Interval, fmt: str) -> Interval:
    # Round-to-nearest is monotone, so the endpoints suffice.
    return Interval(round_nearest_even(x.lo, fmt), round_nearest_even(x.hi, fmt))


def is_representable(value, fmt: str) -> bool:
    try:
        return round_nearest_even(value, fmt) == _to_mpq(value)
    except Overflow:
        return False
