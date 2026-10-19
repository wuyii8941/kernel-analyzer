"""Values shared by the contract checker, the adapters and the tool.

Exact values are Fractions (and bools / ints for discrete outputs). Enclosures are mpmath interval values (iv) at
256 bits; every conversion into an enclosure rounds outward (iv.mpf of an integer, then an iv division), and every
conversion out of one is exact (libmp.to_rational). Display strings never take part in arithmetic.

Outcomes of a reference evaluation are raised as typed exceptions by the adapters and turned into outcomes by
`evaluate`; any other exception is an implementation error, never a mathematical outcome.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from mpmath import iv, libmp

iv.prec = 256
IV = type(iv.mpf(0))

OUTCOMES = ("value", "domain_undefined", "invalid_input", "unsupported", "not_established", "implementation_error")
GUARANTEES = ("exact", "rigorous_enclosure", "allowed_set", "declared_approximation")


class Invalid(Exception):
    """invalid_input: the input violates an explicit shape / type / range condition of the contract."""


class Unsupported(Exception):
    """unsupported: legal for the API but outside every profile of the contract."""


class Undefined(Exception):
    """domain_undefined: legal input, but the contract's finite-real target is undefined (e.g. a zero mean mass)."""


class NotEstablished(Exception):
    """not_established: the reference could not reach the declared guarantee within its budget."""


@dataclass
class Result:
    outcome: str
    fields: dict | None = None
    note: str = ""
    labels: dict = field(default_factory=dict)


def evaluate(fn, *args, **kw) -> Result:
    """Run one adapter call; typed exceptions become outcomes, everything else is an implementation error."""
    try:
        r = fn(*args, **kw)
    except Invalid as e:
        return Result("invalid_input", note=str(e))
    except Unsupported as e:
        return Result("unsupported", note=str(e))
    except Undefined as e:
        return Result("domain_undefined", note=str(e))
    except NotEstablished as e:
        return Result("not_established", note=str(e))
    except Exception as e:  # noqa: BLE001 -- deliberately not a mathematical outcome
        return Result("implementation_error", note=f"{type(e).__name__}: {e}")
    if not isinstance(r, Result) or r.outcome not in OUTCOMES:
        return Result("implementation_error", note=f"adapter returned {type(r).__name__}, not a Result")
    return r


# ---------------------------------------------------------------- strict inputs
_RAT = re.compile(r"^\s*-?\d+(\s*/\s*\d+)?\s*$")


def require_fields(value, fields, what="input"):
    """Validate a complete named input record before any field lookup or computation."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Invalid(f"{what}: expected a mapping with string field names")
    missing = set(fields) - set(value)
    if missing:
        raise Invalid(f"{what}: missing required fields {sorted(missing)}")
    return value


def rational(x, what="value") -> Fraction:
    """The exactly received rational value. Ints, Fractions, 'p/q' strings and finite floats (their exact binary
    value) are accepted; bools, decimals-as-strings, NaN and infinities are invalid."""
    if isinstance(x, bool):
        raise Invalid(f"{what}: boolean is not a number")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            raise Invalid(f"{what}: non-finite value {x}")
        return Fraction(x)
    if isinstance(x, str) and _RAT.match(x):
        num, _, den = x.replace(" ", "").partition("/")
        if den and int(den) == 0:
            raise Invalid(f"{what}: zero denominator")
        return Fraction(int(num), int(den) if den else 1)
    raise Invalid(f"{what}: {x!r} is not an exact rational")


def integer(x, what="value", lo=None) -> int:
    """An exact integer; a rational with a nonzero fractional part is invalid, never truncated."""
    if isinstance(x, bool):
        raise Invalid(f"{what}: boolean is not an integer")
    if isinstance(x, int):
        v = x
    else:
        q = rational(x, what)
        if q.denominator != 1:
            raise Invalid(f"{what}: {x!r} is not an integer")
        v = q.numerator
    if lo is not None and v < lo:
        raise Invalid(f"{what}: {v} < {lo}")
    return v


def vector(xs, what="vector", n=None, item=rational) -> list:
    if not isinstance(xs, (list, tuple)):
        raise Invalid(f"{what}: not a list")
    if n is not None and len(xs) != n:
        raise Invalid(f"{what}: length {len(xs)}, expected {n}")
    return [item(x, f"{what}[{i}]") for i, x in enumerate(xs)]


def matrix(xs, what="matrix", rows=None, cols=None, item=rational) -> list:
    if not isinstance(xs, (list, tuple)) or not xs:
        raise Invalid(f"{what}: not a non-empty list of rows")
    width = len(xs[0]) if isinstance(xs[0], (list, tuple)) else None
    if rows is not None and len(xs) != rows:
        raise Invalid(f"{what}: {len(xs)} rows, expected {rows}")
    out = []
    for i, r in enumerate(xs):
        if not isinstance(r, (list, tuple)) or len(r) != width:
            raise Invalid(f"{what}: ragged row {i}")
        out.append(vector(r, f"{what}[{i}]", cols, item))
    return out


# ---------------------------------------------------------------- enclosures
def to_iv(x) -> IV:
    """Outward-rounded enclosure of an exact value (or an enclosure unchanged)."""
    if isinstance(x, IV):
        return x
    if isinstance(x, bool):
        raise TypeError("boolean has no enclosure")
    if isinstance(x, Decimal):
        x = Fraction(x)
    q = rational(x)
    return iv.mpf(q.numerator) / q.denominator


def ends(x) -> tuple[Fraction, Fraction]:
    """Exact endpoints of an enclosure (a point for exact values)."""
    if isinstance(x, IV):
        a, b = x._mpi_
        for e in (a, b):
            if e in (libmp.fnan, libmp.finf, libmp.fninf):
                raise ValueError("enclosure with a non-finite endpoint")
        return _exact_or_widened(a, "lo"), _exact_or_widened(b, "hi")
    q = rational(x)
    return q, q


_TINY_EXP = -(1 << 20)      # below 2^-1048576 an endpoint is widened outward instead of being written out exactly


def _exact_or_widened(e, side):
    """Exact rational endpoint; an endpoint of astronomically small magnitude (e.g. exp(-1e30)) is widened outward to 0
    or to +-2^-1048576, which keeps the enclosure rigorous and representable."""
    sign, man, exp, bc = e
    if man and exp + bc < _TINY_EXP:
        neg = bool(sign)
        if side == "lo":
            return -Fraction(1, 1 << (1 << 20)) if neg else Fraction(0)
        return Fraction(0) if neg else Fraction(1, 1 << (1 << 20))
    p, q = libmp.to_rational(e)
    return Fraction(int(p), int(q))     # int(): under the gmpy2 backend these are gmpy2.mpz, which Fraction keeps as is


def hull(lo, hi) -> IV:
    """Enclosure of the closed range [lo, hi] given two exact values or enclosures (lo <= hi required)."""
    a, b = to_iv(lo), to_iv(hi)
    if ends(a)[0] > ends(b)[1]:
        raise ValueError("inverted range")
    return iv.mpf([a.a, b.b])


def from_interval(lo, hi) -> IV:
    """Enclosure from exact endpoints of another interval implementation (Decimal, Fraction, int)."""
    lo, hi = Fraction(lo), Fraction(hi)
    if lo > hi:
        raise ValueError(f"inverted interval [{lo}, {hi}]")
    return hull(lo, hi)


def width(x) -> Fraction:
    a, b = ends(x)
    return b - a
