"""Adapter for contracts/RPROP.yaml: one torch.optim.Rprop step (PyTorch 2.10 box) from an explicit pre-state
(theta, prev, eta, t) to the full post-state; exact rational arithmetic. Mutants and readings are separate code."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer


def _inputs(profile, inp):
    if profile != "torch210":
        raise Unsupported(f"profile {profile!r}")
    maximize = flag(inp, "maximize")
    st = state_dict(inp, {"theta", "prev", "eta", "t"})
    theta = vec(st["theta"], "theta")
    d = len(theta)
    prev, eta = vec(st["prev"], "prev", d), vec(st["eta"], "eta", d)
    if any(e < 0 for e in eta):
        raise Invalid("negative step size")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    hp = hyper(inp, ("etaminus", "etaplus", "step_min", "step_max"), nonneg=("step_min", "step_max"))
    if not (0 < hp["etaminus"] < 1 < hp["etaplus"]):
        raise Invalid("etas must satisfy 0 < etaminus < 1 < etaplus")
    return theta, prev, eta, t, g, hp, maximize


def _sign(x):
    return (x > 0) - (x < 0)


def reference(profile, inp):
    theta, prev, eta, t, g, hp, maximize = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "prev": prev, "eta": eta, "t": t})
    lo, hi = hp["step_min"], hp["step_max"]
    th1, p1, e1 = [], [], []
    for x, pi, ei, gi in zip(theta, prev, eta, g):
        gi = -gi if maximize else gi
        prod = pi * gi
        if prod > 0:
            en = min(ei * hp["etaplus"], hi)
        elif prod < 0:
            en = max(ei * hp["etaminus"], lo)
            gi = Fraction(0)
        else:
            if not (lo <= ei <= hi):
                raise Undefined("zero sign product with a step size outside [step_min, step_max]: box and reference "
                                "program differ (decision RP.CLAMP)")
            en = ei
        th1.append(x - en * _sign(gi)), p1.append(gi), e1.append(en)
    return Result("value", {"theta": th1, "prev": p1, "eta": e1, "t": t + 1})


# ------------------------------------------------------------------ independent mutants and readings
def _mut(profile, inp, mis):
    theta, prev, eta, t, g, hp, maximize = _inputs(profile, inp)
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "prev": prev, "eta": eta, "t": t})
        g = [Fraction(0)] * len(theta)
    em, ep, lo, hi = hp["etaminus"], hp["etaplus"], hp["step_min"], hp["step_max"]
    out = {"theta": [], "prev": [], "eta": [], "t": t if mis == "step_not_counted" else t + 1}
    for x, pi, ei, gi in zip(theta, prev, eta, g):
        if maximize and mis != "ascent_at_update":
            gi = -gi
        s = _sign(pi * gi)
        if mis == "zero_as_increase" and s == 0:
            s = 1
        if s > 0:
            en = ei * ep if mis == "no_clamp" else min(ei * ep, hi)
        elif s < 0:
            en = ei * em if mis == "no_clamp" else max(ei * em, lo)
        elif mis == "reading_program_clamp":
            en = min(max(ei, lo), hi)
        else:
            en = ei
        gz = gi if (s >= 0 or mis == "no_zeroing") else Fraction(0)
        step = en * gz if mis == "magnitude_not_sign" else en * _sign(gz)
        if mis == "ascent_at_update" and maximize:
            out["theta"].append(x + step)
        else:
            out["theta"].append(x - step)
        out["prev"].append(gi if mis == "prev_not_zeroed" else gz)
        out["eta"].append(en)
    return Result("value", out)


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "RP.M1": _m("prev_not_zeroed"),
    "RP.M2": _m("no_zeroing"),
    "RP.M3": _m("zero_as_increase"),
    "RP.M4": _m("no_clamp"),
    "RP.M5": _m("magnitude_not_sign"),
    "RP.M6": _m("ascent_at_update"),
    "RP.M7": _m("none_as_zero"),
    "RP.M8": _m("step_not_counted"),
}
READINGS = {"program_clamp": _m("reading_program_clamp"), "box_literal": _m("box_literal")}
