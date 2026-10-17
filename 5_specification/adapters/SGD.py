"""Adapter for contracts/SGD.yaml: one torch.optim.SGD step from an explicit pre-state (theta, b) to the full
post-state. b = [] means that no momentum buffer exists yet (first step of this parameter, or momentum = 0).

The reference is the v2.4 phase-2 sgd_step (v2 order: negate, then decay, then momentum) on a prefilled buffer; the
mutants are independent implementations, each with one misreading."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict
from kernel_spec.num import Invalid, Result, Unsupported, vector

opt = v24.module("spec_optimizers")


def _inputs(profile, inp):
    if profile != "torch210":
        raise Unsupported(f"profile {profile!r}")
    maximize, nesterov = flag(inp, "maximize"), flag(inp, "nesterov")
    st = state_dict(inp, {"theta", "b"})
    theta = vector(st["theta"], "theta")
    if not theta:
        raise Invalid("empty parameter")
    d = len(theta)
    b = vector(st["b"], "b")
    if len(b) not in (0, d):
        raise Invalid("momentum buffer must be [] (absent) or have the parameter's length")
    g = grad_or_none(inp, d)
    hp = hyper(inp, ("lr", "momentum", "dampening", "weight_decay"), nonneg=("lr", "momentum", "weight_decay"))
    if nesterov and (hp["momentum"] <= 0 or hp["dampening"] != 0):
        raise Invalid("nesterov requires momentum > 0 and dampening = 0")
    return theta, b, g, hp, maximize, nesterov


def reference(profile, inp):
    theta, b, g, hp, maximize, nesterov = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "b": b})
    state = opt.SGDState(len(theta))
    if b:
        state.b = list(b)
    out = opt.sgd_step(theta, g, state, hp["lr"], hp["momentum"], hp["dampening"], hp["weight_decay"],
                       nesterov=nesterov, maximize=maximize)
    b_new = v24.tree(state.b) if hp["momentum"] != 0 else b
    return Result("value", {"theta": v24.tree(out), "b": b_new})


# ------------------------------------------------------------------ independent mutants
def _step(profile, inp, mis):
    wanted = inp.get("nesterov")
    if mis == "nesterov_unchecked" and wanted is True:   # precondition not enforced: compute nesterov anyway
        inp = dict(inp, nesterov=False)
    theta, b, g, hp, maximize, nesterov = _inputs(profile, inp)
    if mis == "nesterov_unchecked" and wanted is True:
        nesterov = True
    lr, mu, tau, wd = hp["lr"], hp["momentum"], hp["dampening"], hp["weight_decay"]
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "b": b})
        g = [Fraction(0)] * len(theta)
    out, bn = [], []
    for i, (x, gi) in enumerate(zip(theta, g)):
        if mis == "maximize_after_decay":
            gi = gi + wd * x
            gi = -gi if maximize else gi
        elif mis == "maximize_at_update":
            gi = gi + wd * x
        else:
            gi = (-gi if maximize else gi) + wd * x
        if mu != 0:
            if mis == "lr_in_buffer":
                bi = (mu * b[i] if b else 0) + lr * (1 - tau) * gi
                out.append(x - bi), bn.append(bi)
                continue
            if not b:
                bi = (1 - tau) * gi if mis == "dampening_first_step" else gi
            else:
                bi = mu * b[i] + (1 - tau) * gi
            bn.append(bi)
            if nesterov and mis != "nesterov_ignored":
                gi = gi + mu * bi
            else:
                gi = bi
        out.append(x + lr * gi if (mis == "maximize_at_update" and maximize) else x - lr * gi)
    if mu == 0 or mis == "buffer_not_stored":
        bn = b
    return Result("value", {"theta": out, "b": bn})


def _m(kind):
    return lambda profile, inp: _step(profile, inp, kind)


MUTANTS = {
    "SG.M1": _m("maximize_after_decay"),
    "SG.M2": _m("dampening_first_step"),
    "SG.M3": _m("nesterov_ignored"),
    "SG.M4": _m("maximize_at_update"),
    "SG.M5": _m("none_as_zero"),
    "SG.M6": _m("nesterov_unchecked"),
    "SG.M7": _m("buffer_not_stored"),
    "SG.M8": _m("lr_in_buffer"),
}
READINGS = {}
