"""Adapter for contracts/RADAM.yaml: one torch.optim.RAdam step (PyTorch 2.10 box) from an explicit pre-state
(theta, m, v, t) to the full post-state. rho_t is an exact rational; the rectification factor and the adaptive term are
strict enclosures (kernel_spec.real). The mutants are separate implementations, each with one misreading."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer
from kernel_spec.real import R, sqrt


def _inputs(profile, inp):
    if profile != "torch210":
        raise Unsupported(f"profile {profile!r}")
    maximize, decoupled = flag(inp, "maximize"), flag(inp, "decoupled")
    st = state_dict(inp, {"theta", "m", "v", "t"})
    theta = vec(st["theta"], "theta")
    d = len(theta)
    m, v = vec(st["m"], "m", d), vec(st["v"], "v", d)
    if any(x < 0 for x in v):
        raise Invalid("negative second moment")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"), nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas outside [0, 1)")
    return theta, m, v, t, g, hp, maximize, decoupled


def reference(profile, inp):
    theta, m, v, t, g, hp, maximize, decoupled = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    t1 = t + 1
    rho_inf = 2 / (1 - b2) - 1
    rho = rho_inf - 2 * t1 * b2 ** t1 / (1 - b2 ** t1)
    rect = rho > 5
    if rect:
        r = sqrt(R((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)))
        c2 = sqrt(R(1 - b2 ** t1))
    th1, m1, v1 = [], [], []
    for x, mi, vi, gi in zip(theta, m, v, g):
        if maximize:
            gi = -gi
        xt = x
        if wd != 0:
            if decoupled:
                xt = xt - lr * wd * xt
            else:
                gi = gi + wd * xt
        mn = b1 * mi + (1 - b1) * gi
        vn = b2 * vi + (1 - b2) * gi * gi
        mh = mn / (1 - b1 ** t1)
        if rect:
            if vn == 0 and eps == 0:
                raise Undefined("sqrt(v_t) + eps = 0 in the rectified branch")
            new = R(xt) - R(lr * mh) * r * (c2 / (sqrt(R(vn)) + R(eps)))
        else:
            new = xt - lr * mh
        th1.append(new), m1.append(mn), v1.append(vn)
    return Result("value", {"theta": th1, "m": m1, "v": v1, "t": t1})


# ------------------------------------------------------------------ independent mutants
def _rho32(b2, t1):
    """B012: rho_t evaluated in binary32 arithmetic (as the compiled / capturable path does)."""
    f = np.float32
    b, tt = f(float(b2)), f(t1)
    ri = f(2) / (f(1) - b) - f(1)
    bt = b ** tt
    return Fraction(float(ri - f(2) * tt * bt / (f(1) - bt))), Fraction(float(ri))


def _mut(profile, inp, mis):
    theta, m, v, t, g, hp, maximize, decoupled = _inputs(profile, inp)
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
        g = [Fraction(0)] * len(theta)
    if mis == "decoupled_swapped":
        decoupled = not decoupled
    t1 = t + 1
    if mis == "rho_binary32":
        rho, rho_inf = _rho32(b2, t1)
    else:
        rho_inf = Fraction(2) / (1 - b2) - 1
        rho = rho_inf - 2 * t1 * b2 ** t1 / (1 - b2 ** t1)
    threshold = 4 if mis == "threshold_4" else 5
    out, mo, vo = [], [], []
    for x, mi, vi, gi in zip(theta, m, v, g):
        if mis == "maximize_after_decay":
            gi = gi + (0 if decoupled else wd * x)
            gi = -gi if maximize else gi
        else:
            gi = (-gi if maximize else gi) + (0 if decoupled else wd * x)
        base = x * (1 - lr * wd) if decoupled else x
        mn, vn = b1 * mi + (1 - b1) * gi, b2 * vi + (1 - b2) * gi * gi
        mh = mn / (1 - b1 ** t1)
        if rho > threshold or mis == "always_adaptive":
            if mis == "always_adaptive":
                rr = R(1)
            else:
                rr = sqrt(R((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)))
            if vn == 0 and eps == 0:
                raise Undefined("zero denominator")
            if mis in ("bias_corrected_v", "always_adaptive"):
                ad = R(1) / (sqrt(R(vn / (1 - b2 ** t1))) + R(eps))
            else:
                ad = sqrt(R(1 - b2 ** t1)) / (sqrt(R(vn)) + R(eps))
            new = R(base) - R(lr * mh) * rr * ad
        else:
            new = R(base - lr * mh)
        out.append(new), mo.append(mn), vo.append(vn)
    return Result("value", {"theta": out, "m": mo, "v": vo, "t": t1})


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "RA.M1": _m("rho_binary32"),
    "RA.M2": _m("threshold_4"),
    "RA.M3": _m("always_adaptive"),
    "RA.M4": _m("bias_corrected_v"),
    "RA.M5": _m("decoupled_swapped"),
    "RA.M6": _m("maximize_after_decay"),
    "RA.M7": _m("none_as_zero"),
}
READINGS = {}
