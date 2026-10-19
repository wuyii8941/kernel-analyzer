"""Adapter for contracts/ADAM.yaml: one Adam step (coupled L2 weight decay, amsgrad, maximize) and the AdamW variants
with amsgrad or maximize, from an explicit pre-state (theta, m, v, vmax, t) to the full post-state.

The reference is the v2.4 phase-2 adam_step (amsgrad reading "main": maximum over the uncorrected v_t) on a prefilled
state. The mutants are independent implementations (kernel_spec.real), each with one misreading."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import bf16_rn, flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer
from kernel_spec.real import R, sqrt

opt = v24.module("spec_optimizers")
KEYS = {"theta", "m", "v", "vmax", "t"}


def _inputs(profile, inp):
    maximize, amsgrad, decoupled = flag(inp, "maximize"), flag(inp, "amsgrad"), flag(inp, "decoupled")
    if profile == "adam":
        if decoupled:
            raise Unsupported("decoupled weight decay belongs to profile adamw_variants (or ADAMW)")
    elif profile == "adamw_variants":
        if not decoupled:
            raise Unsupported("coupled L2 weight decay belongs to profile adam")
        if not (amsgrad or maximize):
            raise Unsupported("AdamW with amsgrad=False and maximize=False is contract ADAMW, profile base")
    else:
        raise Unsupported(f"profile {profile!r}")
    st = state_dict(inp, KEYS)
    theta = vec(st["theta"], "theta")
    d = len(theta)
    m, v, vmax = vec(st["m"], "m", d), vec(st["v"], "v", d), vec(st["vmax"], "vmax", d)
    if any(x < 0 for x in v + vmax):
        raise Invalid("negative second moment or vmax")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"), nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas outside [0, 1)")
    return theta, m, v, vmax, t, g, hp, maximize, amsgrad, decoupled


def reference(profile, inp):
    theta, m, v, vmax, t, g, hp, maximize, amsgrad, decoupled = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": t})
    b2, eps, wd = hp["beta2"], hp["eps"], hp["weight_decay"]
    for i in range(len(theta)):                       # domain check: sqrt(v_hat) + eps == 0
        gi = (-g[i] if maximize else g[i]) + (0 if decoupled else wd * theta[i])
        vn = b2 * v[i] + (1 - b2) * gi * gi
        if eps == 0 and (max(vn, vmax[i]) if amsgrad else vn) == 0:
            raise Undefined("sqrt(v_hat) + eps = 0")
    state = opt.AdamState(len(theta))
    state.m, state.v, state.vmax, state.t = list(m), list(v), list(vmax), t
    out = opt.adam_step(theta, g, state, hp["lr"], hp["beta1"], b2, eps, wd, amsgrad=amsgrad, maximize=maximize,
                        decoupled=decoupled)
    return Result("value", {"theta": v24.tree(out), "m": v24.tree(state.m), "v": v24.tree(state.v),
                            "vmax": v24.tree(state.vmax), "t": state.t})


# ------------------------------------------------------------------ independent mutants
def _step(profile, inp, mis):
    theta, m, v, vmax, t, g, hp, maximize, amsgrad, decoupled = _inputs(profile, inp)
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": t})
        g = [Fraction(0)] * len(theta)
    if mis == "decoupled_swapped":
        decoupled = not decoupled
    t1 = t + 1
    th1, m1, v1, x1 = [], [], [], []
    for x, mi, vi, xi, gi in zip(theta, m, v, vmax, g):
        if mis == "maximize_after_decay":
            gi = gi + (0 if decoupled else wd * x)
            gi = -gi if maximize else gi
        elif mis == "maximize_at_update":
            gi = gi + (0 if decoupled else wd * x)
        else:
            gi = -gi if maximize else gi
            gi = gi + (0 if decoupled else wd * x)
        mn, vn = b1 * mi + (1 - b1) * gi, b2 * vi + (1 - b2) * gi * gi
        bc1, bc2 = 1 - b1 ** t1, 1 - b2 ** t1
        xn = xi
        if amsgrad and mis == "amsgrad_corrected_max":
            xn = max(xi, vn / bc2)
            vh = xn
        elif amsgrad and mis != "amsgrad_ignored":
            xn = max(xi, vn)
            vh = xn / bc2
        else:
            vh = vn / bc2
        if mis == "vmax_not_stored":
            xn = xi
        if eps == 0 and vh == 0:
            raise Undefined("sqrt(v_hat) + eps = 0")
        step = R(lr * mn / bc1) / (sqrt(R(vh)) + R(eps))
        base = R(x * (1 - lr * wd)) if decoupled else R(x)
        new = base + step if mis == "maximize_at_update" and maximize else base - step
        th1.append(new), m1.append(mn), v1.append(vn), x1.append(xn)
    if mis == "bf16_storage":                          # B005: every stored tensor rounded to bfloat16 (nearest)
        mid = [Fraction(*_mid(z)) for z in th1]
        return Result("value", {"theta": [bf16_rn(z) for z in mid], "m": [bf16_rn(z) for z in m1],
                                "v": [bf16_rn(z) for z in v1], "vmax": [bf16_rn(z) for z in x1], "t": t1})
    return Result("value", {"theta": th1, "m": m1, "v": v1, "vmax": x1, "t": t1})


def _mid(z):
    lo, hi = z.bounds()
    q = (lo + hi) / 2
    return q.numerator, q.denominator


def _m(kind):
    return lambda profile, inp: _step(profile, inp, kind)


MUTANTS = {
    "AD.M1": _m("amsgrad_corrected_max"),
    "AD.M2": _m("amsgrad_ignored"),
    "AD.M3": _m("vmax_not_stored"),
    "AD.M4": _m("maximize_after_decay"),
    "AD.M5": _m("maximize_at_update"),
    "AD.M6": _m("decoupled_swapped"),
    "AD.M7": _m("decoupled_swapped"),
    "AD.M8": _m("none_as_zero"),
    "AD.M9": _m("bf16_storage"),
}
READINGS = {}
