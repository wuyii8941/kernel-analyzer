"""Adapter for contracts/SCALER.yaml. The reference is kernel_spec.e2_train.scaler_iteration; the mutants are an
independent implementation (Fractions and float specials), each switching on one misreading."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import e2_train
from kernel_spec.num import Result


def reference(profile, inp):
    return Result("value", e2_train.scaler_iteration(profile, inp))


READINGS = {}


def _v(x):
    return float(x) if isinstance(x, str) and x.strip() in ("inf", "-inf", "nan") else Fraction(x)


def _map(t, fn):
    return [_map(e, fn) for e in t] if isinstance(t, list) else fn(t)


def _flat(t):
    if isinstance(t, list):
        for e in t:
            yield from _flat(e)
    else:
        yield t


def _out(x):
    if isinstance(x, float):
        return frozenset() if x != x else ("+inf" if x > 0 else "-inf")
    return x


def _independent(profile, inp, mis):
    S = Fraction(inp.get("scale", 65536))
    gf, bf = Fraction(inp.get("growth_factor", 2)), Fraction(inp.get("backoff_factor", Fraction(1, 2)))
    gi, tr = int(inp.get("growth_interval", 2000)), int(inp.get("growth_tracker", 0))
    G = [_map(g, _v) for g in inp["grads"]]
    vals = list(_flat(G))
    if not inp.get("enabled", True):
        found = mis == "disabled_checks" and any(isinstance(v, float) for v in vals)
        return Result("value", {"unscaled": [_map(g, _out) for g in G], "found_inf": found, "stepped": not found,
                                "scale": S, "growth_tracker": tr})
    if mis == "inf_only":
        found = any(isinstance(v, float) and v == v for v in vals)
    else:
        found = any(isinstance(v, float) for v in vals)
    div = (lambda v: v * S) if mis == "multiply" else (lambda v: v if isinstance(v, float) else v / S)
    if mis == "no_unscale_when_skipped" and found:
        div = lambda v: v                                           # noqa: E731
    unscaled = [_map(g, lambda v: _out(div(v))) for g in G]
    stepped = True if mis == "step_anyway" else not found
    if found:
        S2 = S if mis == "no_backoff" else S * bf
        tr2 = tr if mis == "no_reset_on_skip" else 0
    else:
        if mis == "late_growth":
            grow, nxt = tr == gi, tr + 1
        else:
            grow, nxt = tr + 1 == gi, tr + 1
        S2 = S * gf if grow else S
        tr2 = (nxt if mis == "no_reset_on_growth" else 0) if grow else nxt
    return Result("value", {"unscaled": unscaled, "found_inf": found, "stepped": stepped, "scale": S2,
                            "growth_tracker": tr2})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "SCL.M1": _m("step_anyway"), "SCL.M2": _m("inf_only"), "SCL.M3": _m("no_backoff"), "SCL.M4": _m("no_reset_on_skip"),
    "SCL.M5": _m("no_reset_on_growth"), "SCL.M6": _m("late_growth"), "SCL.M7": _m("multiply"),
    "SCL.M8": _m("no_unscale_when_skipped"), "SCL.M9": _m("disabled_checks"),
}
