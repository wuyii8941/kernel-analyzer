"""Input helpers shared by the issuer-task contracts (BLIND_V1, BLIND_V2, STRUCTURE; group F_issuer).

Strict: every value is the exactly received rational (kernel_spec.num.rational); shapes are checked, never repaired.
"""
from __future__ import annotations

from fractions import Fraction

from .num import Invalid, Unsupported, integer, rational
from .real import R


def keys(inp, want, what):
    if not isinstance(inp, dict) or set(inp) != set(want):
        got = sorted(inp) if isinstance(inp, dict) else type(inp).__name__
        raise Invalid(f"{what} takes exactly {sorted(want)}, got {got}")


def profile(p, allowed):
    if p not in allowed:
        raise Unsupported(f"profile {p!r}")


def tensor(x, what, item=rational):
    """(shape, flat values) of a nested list; ragged lists are invalid."""
    def shape_of(v):
        if isinstance(v, (list, tuple)):
            if not v:
                return (0,)
            subs = [shape_of(e) for e in v]
            if any(s != subs[0] for s in subs):
                raise Invalid(f"{what}: ragged nested list")
            return (len(v),) + subs[0]
        return ()

    def flat(v, out):
        if isinstance(v, (list, tuple)):
            for e in v:
                flat(e, out)
        else:
            out.append(item(v, what))
        return out
    shp = shape_of(x)
    if not shp:
        raise Invalid(f"{what}: expected a tensor (nested list), got a scalar")
    return shp, flat(x, [])


def nest(flat, shape):
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0] if shape[0] else 0
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def rank(x, what, n, item=rational):
    shp, v = tensor(x, what, item)
    if len(shp) != n:
        raise Invalid(f"{what}: rank {len(shp)}, expected {n}")
    return shp, v


def mat(x, what, item=rational):
    """Rank-2 list of rows (R >= 1); rows may be empty (D = 0) -- the contract decides what that means."""
    if not isinstance(x, (list, tuple)) or not x:
        raise Invalid(f"{what}: expected a non-empty list of rows")
    if not all(isinstance(r, (list, tuple)) for r in x):
        raise Invalid(f"{what}: expected rank 2")
    shp, v = rank(x, what, 2, item)
    return [v[i * shp[1]:(i + 1) * shp[1]] for i in range(shp[0])]


def vec(x, what, n=None, item=rational):
    shp, v = rank(x, what, 1, item)
    if n is not None and shp[0] != n:
        raise Invalid(f"{what}: length {shp[0]}, expected {n}")
    return v


def scalar(x, what):
    if isinstance(x, (list, tuple, dict)):
        raise Invalid(f"{what}: expected a scalar")
    return rational(x, what)


def count(x, what, lo=None):
    return integer(x, what, lo)


def v(r):
    """Plain value of an R (Fraction or enclosure) for the comparison."""
    return r.v if isinstance(r, R) else r


__all__ = ["keys", "profile", "tensor", "nest", "rank", "mat", "vec", "scalar", "count", "v", "Fraction"]
