"""Adapter for contracts/ADAMW.yaml: one AdamW step from an explicit pre-state (theta, m, v, t) to the full
post-state. The reference is the v2.4 phase2 adam_step on a prefilled state; the mutants are independent."""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import v24
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational, to_iv, vector

opt = v24.module("spec_optimizers")


def _inputs(profile, inp):
    if profile != "base":
        raise Unsupported(f"profile {profile!r}")
    if inp.get("maximize") is not False:
        raise Unsupported("maximize=True is outside this contract") if inp.get("maximize") is True else \
            Invalid("maximize must be a boolean")
    if inp.get("amsgrad") is not False:
        raise Unsupported("amsgrad=True is outside this contract") if inp.get("amsgrad") is True else \
            Invalid("amsgrad must be a boolean")
    st = inp["state"]
    if not isinstance(st, dict) or set(st) != {"theta", "m", "v", "t"}:
        raise Invalid("state must have exactly theta, m, v, t")
    theta = vector(st["theta"], "theta")
    d = len(theta)
    if d < 1:
        raise Invalid("empty parameter")
    m, v = vector(st["m"], "m", d), vector(st["v"], "v", d)
    if any(x < 0 for x in v):
        raise Invalid("negative second moment")
    t = integer(st["t"], "t", lo=0)
    g = None if inp["gradient"] is None else vector(inp["gradient"], "gradient", d)
    hp = {k: rational(inp[k], k) for k in ("lr", "beta1", "beta2", "eps", "weight_decay")}
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas outside [0, 1)")
    if min(hp["lr"], hp["eps"], hp["weight_decay"]) < 0:
        raise Invalid("negative lr, eps or weight_decay")
    return theta, m, v, t, g, hp


def reference(profile, inp):
    theta, m, v, t, g, hp = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
    for vi, gi in zip(v, g):
        if hp["eps"] == 0 and hp["beta2"] * vi + (1 - hp["beta2"]) * gi * gi == 0:
            raise Undefined("sqrt(v_hat) + eps = 0")
    state = opt.AdamState(len(theta))
    state.m, state.v, state.t = list(m), list(v), t
    out = opt.adam_step(theta, g, state, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"],
                        amsgrad=False, maximize=False, decoupled=True)
    return Result("value", {"theta": v24.tree(out), "m": v24.tree(state.m), "v": v24.tree(state.v), "t": state.t})


# ------------------------------------------------------------------ independent mutants
def _step(profile, inp, misreading):
    theta, m, v, t, g, hp = _inputs(profile, inp)
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    if g is None:
        if misreading != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
        g = [Fraction(0)] * len(theta)
    if misreading == "ignore_warm_state":
        m, v, t = [Fraction(0)] * len(theta), [Fraction(0)] * len(theta), 0
    if eps == 0 and any(b2 * vi + (1 - b2) * gi * gi == 0 for vi, gi in zip(v, g)):
        raise Undefined("sqrt(v_hat) + eps = 0")
    t1 = t + 1
    out, m1, v1 = [], [], []
    for x, mi, vi, gi in zip(theta, m, v, g):
        if misreading == "coupled_l2":
            gi = gi + wd * x
        mn, vn = b1 * mi + (1 - b1) * gi, b2 * vi + (1 - b2) * gi * gi
        mh, vh = (mn, vn) if misreading == "no_bias_correction" else (mn / (1 - b1 ** t1), vn / (1 - b2 ** t1))
        den = iv.sqrt(to_iv(vh + eps)) if misreading == "eps_inside_sqrt" else iv.sqrt(to_iv(vh)) + to_iv(eps)
        step = to_iv(lr * mh) / den
        if misreading == "coupled_l2":
            new = to_iv(x) - step
        elif misreading == "decay_updated_theta":
            new = (to_iv(x) - step) * to_iv(1 - lr * wd)
        else:
            new = to_iv(x * (1 - lr * wd)) - step
        out.append(new), m1.append(mn), v1.append(vn)
    if misreading == "post_state_not_returned":
        return Result("value", {"theta": out, "m": m, "v": v, "t": t})
    return Result("value", {"theta": out, "m": m1, "v": v1, "t": t1})


def _m(kind):
    return lambda profile, inp: _step(profile, inp, kind)


MUTANTS = {
    "AW.M1": _m("decay_updated_theta"),
    "AW.M2": _m("coupled_l2"),
    "AW.M3": _m("no_bias_correction"),
    "AW.M4": _m("eps_inside_sqrt"),
    "AW.M5": _m("ignore_warm_state"),
    "AW.M6": _m("post_state_not_returned"),
    "AW.M7": _m("none_as_zero"),
}
READINGS = {}
