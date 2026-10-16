"""Real values for the references: exact rationals, rigorous enclosures, and the two infinities.

`R` wraps one of
- a Fraction (exact),
- an mpmath interval (a rigorous enclosure, outward rounded, 256 bits),
- +inf / -inf (exact extended reals, used only where the operation has that value as its limit).
Arithmetic on R follows the real numbers. Where the real operation has no value -- 0/0, inf - inf, 0 * inf, x/0, log of a
negative number, sqrt of a negative number -- it raises Undefined; a contract maps that to domain_undefined. When an
enclosure is too wide to decide a comparison or a discontinuous function (sign, floor, a branch), it raises
NotEstablished rather than guessing.

There is no NaN and no signed zero: the reals have neither. A contract that needs IEEE special values says so and
handles them itself.
"""
from __future__ import annotations

import math
from fractions import Fraction

from mpmath import iv, libmp

from .num import IV, NotEstablished, Undefined, ends, rational, to_iv

POS, NEG = "+inf", "-inf"


def _is_inf(v):
    return isinstance(v, str)


class R:
    __slots__ = ("v",)

    def __init__(self, v):
        if isinstance(v, R):
            v = v.v
        elif isinstance(v, bool):
            raise TypeError("booleans are not reals")
        elif isinstance(v, (int, Fraction)):
            v = Fraction(v)
        elif isinstance(v, float):
            if math.isnan(v):
                raise Undefined("NaN is not a real number")
            v = POS if v == math.inf else NEG if v == -math.inf else Fraction(v)
        elif isinstance(v, str) and v in (POS, NEG):
            pass
        elif isinstance(v, IV):
            a, b = v._mpi_
            if a == libmp.fninf or b == libmp.finf or a == libmp.fnan or b == libmp.fnan:
                raise NotEstablished("enclosure with an infinite endpoint")
        else:
            v = rational(v)
        self.v = v

    # -------------------------------------------------------------- inspection
    @property
    def exact(self):
        return not isinstance(self.v, IV)

    @property
    def finite(self):
        return not _is_inf(self.v)

    def bounds(self):
        """Exact endpoints (Fractions, or the infinity strings)."""
        if _is_inf(self.v):
            return self.v, self.v
        return ends(self.v)

    def sign(self):
        """-1, 0 or 1; NotEstablished when an enclosure contains 0 in its interior or at an endpoint without being 0."""
        if self.v == POS:
            return 1
        if self.v == NEG:
            return -1
        a, b = self.bounds()
        if a > 0:
            return 1
        if b < 0:
            return -1
        if a == b == 0:
            return 0
        raise NotEstablished("sign of an enclosure that contains zero")

    def value(self):
        """The plain value for comparison with expected values: Fraction, IV or an infinity string."""
        return self.v

    def __repr__(self):
        if _is_inf(self.v):
            return self.v
        a, b = self.bounds()
        return str(a) if a == b and not isinstance(self.v, IV) else f"[{float(a):.17g}, {float(b):.17g}]"

    # -------------------------------------------------------------- arithmetic
    def __neg__(self):
        if self.v == POS:
            return R(NEG)
        if self.v == NEG:
            return R(POS)
        return R(-self.v)

    def __add__(self, o):
        o = R(o)
        if _is_inf(self.v) or _is_inf(o.v):
            s = {x.v for x in (self, o) if _is_inf(x.v)}
            if len(s) == 2:
                raise Undefined("inf - inf")
            return R(s.pop())
        if isinstance(self.v, Fraction) and isinstance(o.v, Fraction):
            return R(self.v + o.v)
        return R(to_iv(self.v) + to_iv(o.v))

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-R(o))

    def __rsub__(self, o):
        return R(o) + (-self)

    def __mul__(self, o):
        o = R(o)
        if _is_inf(self.v) or _is_inf(o.v):
            s1, s2 = self.sign(), o.sign()
            if s1 == 0 or s2 == 0:
                raise Undefined("0 * inf")
            return R(POS if s1 * s2 > 0 else NEG)
        if isinstance(self.v, Fraction) and isinstance(o.v, Fraction):
            return R(self.v * o.v)
        return R(to_iv(self.v) * to_iv(o.v))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = R(o)
        if _is_inf(o.v):
            if _is_inf(self.v):
                raise Undefined("inf / inf")
            return R(0)
        s = o.sign()                       # NotEstablished if an enclosure of the divisor contains 0
        if s == 0:
            raise Undefined("division by zero")
        if _is_inf(self.v):
            return R(self.v) if s > 0 else -R(self.v)
        if isinstance(self.v, Fraction) and isinstance(o.v, Fraction):
            return R(self.v / o.v)
        return R(to_iv(self.v) / to_iv(o.v))

    def __rtruediv__(self, o):
        return R(o) / self

    def __pow__(self, k):
        """Integer powers exactly; other exponents through exp(k log x) for x > 0."""
        k = R(k)
        if isinstance(k.v, Fraction) and k.v.denominator == 1:
            n = k.v.numerator
            if n == 0:
                return R(1)                # x**0 = 1, including 0**0 (torch.pow documents 1)
            if _is_inf(self.v):
                if n < 0:
                    return R(0)
                return R(POS) if self.v == POS or n % 2 == 0 else R(NEG)
            if n < 0:
                return R(1) / (self ** (-n))
            if isinstance(self.v, Fraction):
                return R(self.v ** n)
            return R(to_iv(self.v) ** n)
        s = self.sign()
        if s < 0:
            raise Undefined("non-integer power of a negative number")
        if s == 0:
            ks = k.sign()
            if ks > 0:
                return R(0)
            raise Undefined("zero to a non-positive non-integer power")
        return exp(k * log(self))

    # -------------------------------------------------------------- comparison (decided or NotEstablished)
    def _cmp(self, o):
        o = R(o)
        rank = lambda v: 1 if v == POS else -1 if v == NEG else 0
        if _is_inf(self.v) or _is_inf(o.v):
            a, b = rank(self.v), rank(o.v)
            if a != b:
                return (a > b) - (a < b)
            if a != 0:
                return 0
        return (self - o).sign()

    def __lt__(self, o):
        return self._cmp(o) < 0

    def __le__(self, o):
        return self._cmp(o) <= 0

    def __gt__(self, o):
        return self._cmp(o) > 0

    def __ge__(self, o):
        return self._cmp(o) >= 0

    def same(self, o):
        """Equality as a decided fact (exact values), else NotEstablished."""
        return self._cmp(o) == 0

    def __hash__(self):
        return hash(self.v) if self.exact else id(self)


# ------------------------------------------------------------------ functions
def _iv(x):
    return to_iv(x.v)


def hull(a: R, b: R) -> R:
    lo, hi = min_(a, b), max_(a, b)
    if not (lo.finite and hi.finite):
        raise NotEstablished("hull with an infinity")
    return R(iv.mpf([_iv(lo).a, _iv(hi).b]))


def max_(a, b):
    a, b = R(a), R(b)
    try:
        return a if a >= b else b
    except NotEstablished:
        return R(iv.mpf([max(_iv(a).a, _iv(b).a), max(_iv(a).b, _iv(b).b)]))


def min_(a, b):
    a, b = R(a), R(b)
    try:
        return a if a <= b else b
    except NotEstablished:
        return R(iv.mpf([min(_iv(a).a, _iv(b).a), min(_iv(a).b, _iv(b).b)]))


def abs_(x):
    x = R(x)
    if not x.finite:
        return R(POS)
    try:
        return x if x.sign() >= 0 else -x
    except NotEstablished:
        return R(abs(_iv(x)))


def exp(x):
    x = R(x)
    if x.v == POS:
        return R(POS)
    if x.v == NEG:
        return R(0)
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(1)
    return R(iv.exp(_iv(x)))


def log(x):
    x = R(x)
    if x.v == POS:
        return R(POS)
    s = x.sign()
    if s < 0:
        raise Undefined("log of a negative number")
    if s == 0:
        return R(NEG)                       # the limit; contracts that exclude it check before calling
    if isinstance(x.v, Fraction) and x.v == 1:
        return R(0)
    return R(iv.log(_iv(x)))


def log1p(x):
    return log(R(1) + x)


def expm1(x):
    return exp(x) - 1


def sqrt(x):
    x = R(x)
    if x.v == POS:
        return R(POS)
    s = x.sign()
    if s < 0:
        raise Undefined("sqrt of a negative number")
    if s == 0:
        return R(0)
    if isinstance(x.v, Fraction):
        n, d = x.v.numerator, x.v.denominator
        rn, rd = math.isqrt(n), math.isqrt(d)
        if rn * rn == n and rd * rd == d:
            return R(Fraction(rn, rd))
    return R(iv.sqrt(_iv(x)))


def rsqrt(x):
    return R(1) / sqrt(x)


def tanh(x):
    x = R(x)
    if x.v == POS:
        return R(1)
    if x.v == NEG:
        return R(-1)
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(0)
    return R(1 - 2 / (iv.exp(2 * _iv(x)) + 1))


def sigmoid(x):
    x = R(x)
    if x.v == POS:
        return R(1)
    if x.v == NEG:
        return R(0)
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(Fraction(1, 2))
    return R(1 / (1 + iv.exp(-_iv(x))))


def erf(x):
    from .expr import iv_erf
    x = R(x)
    if x.v == POS:
        return R(1)
    if x.v == NEG:
        return R(-1)
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(0)
    return R(iv_erf(_iv(x)))


def sin(x):
    x = R(x)
    if not x.finite:
        raise Undefined("sin of an infinity")
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(0)
    return R(iv.sin(_iv(x)))


def cos(x):
    x = R(x)
    if not x.finite:
        raise Undefined("cos of an infinity")
    if isinstance(x.v, Fraction) and x.v == 0:
        return R(1)
    return R(iv.cos(_iv(x)))


def floor(x):
    x = R(x)
    if not x.finite:
        return x
    a, b = x.bounds()
    fa, fb = math.floor(a), math.floor(b)
    if fa != fb:
        raise NotEstablished("floor of an enclosure that straddles an integer")
    return R(fa)


def ceil(x):
    x = R(x)
    if not x.finite:
        return x
    a, b = x.bounds()
    ca, cb = math.ceil(a), math.ceil(b)
    if ca != cb:
        raise NotEstablished("ceil of an enclosure that straddles an integer")
    return R(ca)


def trunc(x):
    x = R(x)
    return floor(x) if x.sign() >= 0 else ceil(x)


def sign(x):
    return R(R(x).sign())


def relu(x):
    x = R(x)
    try:
        return x if x.sign() > 0 else R(0)
    except NotEstablished:
        return max_(x, R(0))


def softplus(x, beta=1, threshold=20):
    """torch.nn.functional.softplus: (1/beta) log(1 + exp(beta x)); for beta x > threshold it returns x (documented
    reversion to the linear function, part of the operator's definition)."""
    x, beta = R(x), R(beta)
    bx = beta * x
    if bx > R(threshold):
        return x
    return log1p(exp(bx)) / beta


def lift(x):
    return x if isinstance(x, R) else R(x)
