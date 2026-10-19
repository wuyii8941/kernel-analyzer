"""Input helpers shared by the group D2_special contracts (GDR, SSU, LORA, SAMPLING, SORT, BSR).

Inputs are validated before any arithmetic: nested lists must be rectangular with the stated rank, numbers must be
exact rationals (integers or 'p/q' strings; a binary float is its exact value), infinities only where the caller allows
them. Nothing is truncated, padded or broadcast silently. Results are nested lists of plain values (Fraction, mpmath
interval, '+inf' / '-inf') so that kernel_spec.compare can compare them with the expected values.
"""
from __future__ import annotations

from fractions import Fraction

from .num import Invalid, integer, rational, require_fields
from .real import NEG, POS, R


def shape_of(x, what):
    """Shape of a rectangular nested list (a scalar has shape ())."""
    if not isinstance(x, (list, tuple)):
        return ()
    if not x:
        return (0,)
    subs = [shape_of(e, what) for e in x]
    if any(s != subs[0] for s in subs):
        raise Invalid(f"{what}: ragged nested list")
    return (len(x),) + subs[0]


def leaf(v, what, allow_inf=False):
    if isinstance(v, bool):
        raise Invalid(f"{what}: boolean where a number is required")
    if isinstance(v, str) and v.strip() in ("inf", "+inf", "-inf"):
        if not allow_inf:
            raise Invalid(f"{what}: infinity not allowed here")
        return R(NEG if v.strip() == "-inf" else POS)
    return R(rational(v, what))


def tensor(x, rank, what, allow_inf=False, shape=None):
    """Nested list of R values with exactly `rank` dimensions (and the given shape, None entries free)."""
    shp = shape_of(x, what)
    if len(shp) != rank:
        raise Invalid(f"{what}: rank {len(shp)}, expected {rank}")
    if shape is not None:
        for d, (a, b) in enumerate(zip(shp, shape)):
            if b is not None and a != b:
                raise Invalid(f"{what}: dimension {d} has size {a}, expected {b}")

    def conv(e, depth):
        if depth == rank:
            return leaf(e, what, allow_inf)
        return [conv(s, depth + 1) for s in e]
    return conv(x, 0), shp


def ints(x, rank, what, lo=None):
    shp = shape_of(x, what)
    if len(shp) != rank:
        raise Invalid(f"{what}: rank {len(shp)}, expected {rank}")

    def conv(e, depth):
        if depth == rank:
            return integer(e, what, lo=lo)
        return [conv(s, depth + 1) for s in e]
    return conv(x, 0), shp


def flag(inp, key, default=False):
    v = inp.get(key, default)
    if not isinstance(v, bool):
        raise Invalid(f"{key}: expected a boolean")
    return v


def check_keys(inp, allowed, required=()):
    require_fields(inp, required)
    extra = set(inp) - set(allowed)
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")


def plain(x):
    """Nested lists of R -> nested lists of plain values."""
    if isinstance(x, list):
        return [plain(e) for e in x]
    if isinstance(x, R):
        return x.v
    if isinstance(x, (frozenset, set)):
        return frozenset(x)
    return x


def zeros(*shape):
    if not shape:
        return R(0)
    return [zeros(*shape[1:]) for _ in range(shape[0])]


def rsum(xs):
    s = R(0)
    for x in xs:
        s = s + x
    return s


def frac(x):
    return x.v if isinstance(x, R) else Fraction(x)
