"""Adapter for contracts/CLIP.yaml. The reference is kernel_spec.e2_clip; the mutants are an independent
implementation (Fractions, float infinities / NaN and mpmath intervals), each switching on one misreading."""
from __future__ import annotations

import math
from fractions import Fraction

from mpmath import iv

from kernel_spec import e2_clip
from kernel_spec.num import Invalid, Result, Undefined, to_iv


def reference(profile, inp):
    return Result("value", e2_clip.evaluate(profile, inp))


READINGS = {"minimum_number": lambda p, i: Result("value", e2_clip.evaluate(p, i, min_reading="minimum_number"))}


# ------------------------------------------------------------------ independent implementation for the mutants
def _v(x):
    if isinstance(x, str) and x.strip() in ("inf", "-inf", "nan"):
        return float(x.strip())
    return Fraction(x)


def _flat(t):
    if isinstance(t, list):
        for e in t:
            yield from _flat(e)
    else:
        yield t


def _map(t, fn):
    return [_map(e, fn) for e in t] if isinstance(t, list) else fn(t)


def _out(x):
    if isinstance(x, float):
        return frozenset() if x != x else ("+inf" if x > 0 else "-inf")
    return x


def _mul(g, c, mis):
    if isinstance(c, float) and c != c:
        return float("nan")
    if isinstance(g, float):
        if g != g:
            return g
        if c == 0:
            return 0 if mis == "zero_times_inf" else float("nan")
        return g
    return g * c if isinstance(c, Fraction) else to_iv(g) * c


def _norm(vals, p, mis):
    if mis == "always_l2":
        p = 2
    if any(isinstance(v, float) and v != v for v in vals):
        return float("nan")
    if any(isinstance(v, float) for v in vals):
        return float("inf")
    if p == "inf":
        return max(vals) if mis == "max_no_abs" else max(abs(v) for v in vals)
    s = sum(abs(v) ** Fraction(p) if Fraction(p).denominator == 1 else to_iv(abs(v)) ** to_iv(Fraction(p)) for v in vals)
    if Fraction(p) == 1:
        return s
    return to_iv(s) ** (to_iv(1) / to_iv(Fraction(p)))


def _independent(profile, inp, mis):
    grads = inp["grads"]
    if mis == "truncate_after_none" and None in grads:
        grads = grads[:grads.index(None)]
    G = [None if g is None else _map(g, _v) for g in grads]
    allv = [v for g in G if g is not None for v in _flat(g)]
    if profile == "real" and any(isinstance(v, float) and v != v for v in allv):
        raise Undefined("NaN")
    if inp["op"] == "clip_grad_value":
        c = Fraction(inp["clip_value"])
        lo = -math.inf if mis == "upper_only" else -c
        fn = lambda v: v if (isinstance(v, float) and v != v) else min(max(v, lo), c)   # noqa: E731
        return Result("value", {"total_norm": [], "grads": [[] if g is None else _map(g, lambda v: _out(fn(v))) for g in G]})
    p = inp.get("norm_type", 2)
    p = "inf" if p == "inf" else Fraction(p)
    if mis == "sum_of_norms":
        N = sum(_norm(list(_flat(g)), p, mis) for g in G if g is not None)
    else:
        N = _norm(allv, p, mis)
    nonfinite = isinstance(N, float)
    if nonfinite and inp.get("error_if_nonfinite") and mis != "ignore_error":
        raise Invalid("documented error")
    if inp["op"] == "get_total_norm":
        return Result("value", {"total_norm": _out(N), "grads": [[] if g is None else _map(g, _out) for g in G]})
    mx = Fraction(inp["max_norm"])
    eps = 0 if mis == "no_eps" else Fraction(1, 10 ** 6)

    def coef(n):
        if isinstance(n, float):
            if mis == "nonfinite_coef_one":
                return Fraction(1)
            return float("nan") if n != n else Fraction(0)
        if isinstance(n, Fraction) and n + eps == 0:
            return Fraction(1)
        c = to_iv(mx) / (to_iv(n) + to_iv(eps))
        if isinstance(n, Fraction):
            c = mx / (n + eps)
            return c if (mis == "no_clamp" or c < 1) else Fraction(1)
        return c if (mis == "no_clamp" or c.b < 1) else (Fraction(1) if c.a >= 1 else iv.mpf([c.a, min(c.b, 1)]))
    if profile == "real" and nonfinite:
        if any(isinstance(v, float) for v in allv):
            raise Undefined("0 * inf")
    out = []
    for g in G:
        if g is None:
            out.append([])
            continue
        c = coef(_norm(list(_flat(g)), p, mis)) if mis == "per_param" else coef(N)
        out.append(_map(g, lambda v: _out(_mul(v, c, mis))))
    return Result("value", {"total_norm": _out(N), "grads": out})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "CLIP.M1": _m("no_eps"), "CLIP.M2": _m("no_clamp"), "CLIP.M3": _m("sum_of_norms"), "CLIP.M4": _m("always_l2"),
    "CLIP.M5": _m("per_param"), "CLIP.M6": _m("upper_only"), "CLIP.M7": _m("nonfinite_coef_one"),
    "CLIP.M8": _m("zero_times_inf"), "CLIP.M9": _m("ignore_error"), "CLIP.M10": _m("max_no_abs"),
    "CLIP.M11": _m("truncate_after_none"),
}
