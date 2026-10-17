"""Adapter for contracts/SROUND.yaml: torchao 0.16 _AdamW(bf16_stochastic_round=True), one step from an explicit
pre-state (bf16 theta, m, v, t) and the 16 random bits of each element to the full post-state.

theta_new is the stochastic rounding to bfloat16 of the real AdamW result y, conditioned on the random bits r:
rounded away from zero iff r < frac(y) * 2^16, frac(y) = (|y| - |trunc_bf16(y)|) / spacing. y comes from the v2.4
phase-2 adam_step (decoupled); the mutants are separate code."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import bf16_rn, bf16_toward_zero, flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, NotEstablished, Result, Undefined, Unsupported, integer
from kernel_spec.real import R, sqrt

opt = v24.module("spec_optimizers")
TWO16 = 1 << 16


def _inputs(profile, inp):
    if profile != "torchao016":
        raise Unsupported(f"profile {profile!r}")
    if flag(inp, "amsgrad", default=False):
        raise Unsupported("amsgrad=True was not tested with stochastic rounding")
    st = state_dict(inp, {"theta", "m", "v", "t"})
    theta = vec(st["theta"], "theta")
    if any(bf16_rn(x) != x for x in theta):
        raise Invalid("theta must hold bfloat16 values (the rounding applies to bf16 parameters)")
    d = len(theta)
    m, v = vec(st["m"], "m", d), vec(st["v"], "v", d)
    if any(x < 0 for x in v):
        raise Invalid("negative second moment")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    rnd = inp.get("rand16")
    if not isinstance(rnd, list) or len(rnd) != d:
        raise Invalid("rand16 must give one integer per element")
    rnd = [integer(r, "rand16", lo=0) for r in rnd]
    if any(r >= TWO16 for r in rnd):
        raise Invalid("rand16 values must lie in [0, 2^16)")
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"), nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas outside [0, 1)")
    return theta, m, v, t, g, rnd, hp


def exact_sqrt(q):
    """The rational square root of q if it is rational, else None."""
    from math import isqrt
    q = Fraction(q)
    if q < 0:
        return None
    a, b = isqrt(q.numerator), isqrt(q.denominator)
    return Fraction(a, b) if a * a == q.numerator and b * b == q.denominator else None


def stochastic_round(y, r):
    """bfloat16 value of the real y (exact or enclosure) rounded away from zero iff r < frac*2^16."""
    lo, hi = R(y).bounds()
    if lo == hi == 0:
        return Fraction(0)
    if lo <= 0 <= hi:
        raise NotEstablished("enclosure of the update contains 0")
    a_lo, u = bf16_toward_zero(lo)
    a_hi, u2 = bf16_toward_zero(hi)
    if a_lo != a_hi:
        raise NotEstablished("enclosure crosses a bfloat16 value")
    x_lo, x_hi = sorted(((abs(lo) - abs(a_lo)) / u * TWO16, (abs(hi) - abs(a_lo)) / u * TWO16))
    if r < x_lo:
        away = True
    elif r >= x_hi:
        away = False
    else:
        raise NotEstablished("random bits at the rounding threshold within the enclosure")
    sgn = 1 if lo > 0 else -1
    return a_lo + sgn * u if away else a_lo


def reference(profile, inp):
    theta, m, v, t, g, rnd, hp = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
    b2, eps = hp["beta2"], hp["eps"]
    if eps == 0 and any(b2 * vi + (1 - b2) * gi * gi == 0 for vi, gi in zip(v, g)):
        raise Undefined("sqrt(v_hat) + eps = 0")
    state = opt.AdamState(len(theta))
    state.m, state.v, state.t = list(m), list(v), t
    y = opt.adam_step(theta, g, state, hp["lr"], hp["beta1"], b2, eps, hp["weight_decay"], amsgrad=False,
                      maximize=False, decoupled=True)
    out = []
    for i, (yi, r) in enumerate(zip(y, rnd)):        # an exactly representable update stays exact (perfect squares)
        vh = state.v[i] / (1 - b2 ** state.t)
        root = exact_sqrt(vh)
        if root is not None and root + eps != 0:
            lr, b1, wd = hp["lr"], hp["beta1"], hp["weight_decay"]
            yi = theta[i] * (1 - lr * wd) - lr * (state.m[i] / (1 - b1 ** state.t)) / (root + eps)
        out.append(stochastic_round(R(v24.value(yi)), r))
    return Result("value", {"theta": out, "m": v24.tree(state.m), "v": v24.tree(state.v), "t": state.t})


# ------------------------------------------------------------------ independent mutants
def _mut(profile, inp, mis):
    theta, m, v, t, g, rnd, hp = _inputs(profile, inp)
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    if g is None:
        if mis != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
        g = [Fraction(0)] * len(theta)
    t1 = t + 1
    th, mo, vo = [], [], []
    for x, mi, vi, gi, r in zip(theta, m, v, g, rnd):
        if mis == "coupled_l2":
            gi = gi + wd * x
            base = x
        else:
            base = x * (1 - lr * wd)
        mn, vn = b1 * mi + (1 - b1) * gi, b2 * vi + (1 - b2) * gi * gi
        if eps == 0 and vn == 0:
            raise Undefined("zero denominator")
        y = R(base) - R(lr * mn / (1 - b1 ** t1)) / (sqrt(R(vn / (1 - b2 ** t1))) + R(eps))
        lo, hi = y.bounds()
        mid = (lo + hi) / 2
        if mis == "round_nearest":
            th.append(bf16_rn(mid))
        else:
            if mis == "floor_not_toward_zero" and mid < 0:
                a, u = bf16_toward_zero(mid)
                a = a - u if a != mid else a          # round down (toward -inf) as the truncation
                frac = (mid - a) / u
                th.append(a + u if r < frac * TWO16 else a)
            else:
                a, u = bf16_toward_zero(mid)
                frac = (abs(mid) - abs(a)) / u if u else Fraction(0)
                sgn = 1 if mid > 0 else -1
                if mis == "inverted_probability":
                    away = r >= frac * TWO16
                elif mis == "le_threshold":
                    away = r <= frac * TWO16
                else:
                    away = r < frac * TWO16
                th.append(a + sgn * u if away else a)
        if mis == "states_bf16":                       # B009: states rounded to nearest bf16
            mn, vn = bf16_rn(mn), bf16_rn(vn)
        mo.append(mn), vo.append(vn)
    return Result("value", {"theta": th, "m": mo, "v": vo, "t": t1})


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "SR.M1": _m("round_nearest"),
    "SR.M2": _m("inverted_probability"),
    "SR.M3": _m("floor_not_toward_zero"),
    "SR.M4": _m("le_threshold"),
    "SR.M5": _m("states_bf16"),
    "SR.M6": _m("coupled_l2"),
    "SR.M7": _m("none_as_zero"),
}
READINGS = {}
