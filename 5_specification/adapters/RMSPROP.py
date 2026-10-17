"""Adapter for contracts/RMSPROP.yaml: one torch.optim.RMSprop step from an explicit pre-state (theta, v, gave, b, t)
to the full post-state.

The reference is the v2.4 phase-2 rmsprop_step on a prefilled state (v2.3 fix: the centered path squares gave); the
mutants and the diagnostic reading are independent implementations (kernel_spec.real)."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer
from kernel_spec.real import R, sqrt

opt = v24.module("spec_optimizers")
KEYS = {"theta", "v", "gave", "b", "t"}


def _inputs(profile, inp):
    if profile != "torch210":
        raise Unsupported(f"profile {profile!r}")
    maximize, centered = flag(inp, "maximize"), flag(inp, "centered")
    st = state_dict(inp, KEYS)
    theta = vec(st["theta"], "theta")
    d = len(theta)
    v, gave, b = vec(st["v"], "v", d), vec(st["gave"], "gave", d), vec(st["b"], "b", d)
    if any(x < 0 for x in v):
        raise Invalid("negative square average")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    hp = hyper(inp, ("lr", "alpha", "eps", "weight_decay", "momentum"),
               nonneg=("lr", "alpha", "eps", "weight_decay", "momentum"))
    return theta, v, gave, b, t, g, hp, maximize, centered


def _check_domain(theta, v, gave, g, hp, maximize, centered):
    a, eps, wd = hp["alpha"], hp["eps"], hp["weight_decay"]
    for x, vi, ai, gi in zip(theta, v, gave, g):
        gi = (-gi if maximize else gi) + wd * x
        vn = a * vi + (1 - a) * gi * gi
        vt = vn - (a * ai + (1 - a) * gi) ** 2 if centered else vn
        if vt < 0:
            raise Undefined("negative centered variance estimate (sqrt undefined)")
        if vt == 0 and eps == 0:
            raise Undefined("sqrt(v~) + eps = 0")


def reference(profile, inp):
    theta, v, gave, b, t, g, hp, maximize, centered = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "v": v, "gave": gave, "b": b, "t": t})
    _check_domain(theta, v, gave, g, hp, maximize, centered)
    state = opt.RMSpropState(len(theta))
    state.v, state.gave, state.b = list(v), list(gave), list(b)
    out = opt.rmsprop_step(theta, g, state, hp["lr"], hp["alpha"], hp["eps"], hp["weight_decay"], hp["momentum"],
                           centered=centered, maximize=maximize)
    return Result("value", {"theta": v24.tree(out), "v": v24.tree(state.v), "gave": v24.tree(state.gave),
                            "b": v24.tree(state.b), "t": t + 1})


# ------------------------------------------------------------------ independent mutants and reading
def _step(profile, inp, mis):
    theta, v, gave, b, t, g, hp, maximize, centered = _inputs(profile, inp)
    lr, a, eps, wd, mu = (hp[k] for k in ("lr", "alpha", "eps", "weight_decay", "momentum"))
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "v": v, "gave": gave, "b": b, "t": t})
        g = [Fraction(0)] * len(theta)
    th1, v1, a1, b1 = [], [], [], []
    for x, vi, ai, bi, gi in zip(theta, v, gave, b, g):
        if mis == "maximize_after_decay":
            gi = gi + wd * x
            gi = -gi if maximize else gi
        elif mis == "ascent_at_update":
            gi = gi + wd * x
        else:
            gi = (-gi if maximize else gi) + wd * x
        vn = a * vi + (1 - a) * gi * gi
        an = a * ai + (1 - a) * gi if centered else ai
        if centered and mis == "centered_ignored":
            vt = vn
        elif centered and mis == "centered_old_gave":
            vt = vn - ai * ai
        elif centered and mis == "centered_no_square":
            vt = vn - an
        elif centered:
            vt = vn - an * an
        else:
            vt = vn
        if vt < 0 or (vt == 0 and eps == 0):
            raise Undefined("denominator undefined")
        den = sqrt(R(vt + eps)) if mis == "eps_inside_sqrt" else sqrt(R(vt)) + R(eps)
        q = R(gi) / den
        if mu > 0:
            bn = R(bi) * R(mu) + (q * R(lr) if mis == "lr_in_buffer" else q)
            delta = bn if mis == "lr_in_buffer" else bn * R(lr)
        else:
            bn = bi
            delta = q * R(lr)
        th1.append(R(x) + delta if (mis == "ascent_at_update" and maximize) else R(x) - delta)
        v1.append(vn), a1.append(an), b1.append(bn)
    return Result("value", {"theta": th1, "v": v1, "gave": a1, "b": b1, "t": t if mis == "step_not_counted" else t + 1})


def _m(kind):
    return lambda profile, inp: _step(profile, inp, kind)


MUTANTS = {
    "RM.M1": _m("eps_inside_sqrt"),
    "RM.M2": _m("centered_old_gave"),
    "RM.M3": _m("centered_ignored"),
    "RM.M4": _m("maximize_after_decay"),
    "RM.M5": _m("lr_in_buffer"),
    "RM.M6": _m("none_as_zero"),
    "RM.M7": _m("centered_no_square"),
    "RM.M8": _m("step_not_counted"),
}
READINGS = {"ascent_at_update": _m("ascent_at_update")}
