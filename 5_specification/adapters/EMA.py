"""Adapter for contracts/EMA.yaml: one parameter-averaging update from an explicit pre-state (avg, n) to the full
post-state. AveragedModel profiles use the v2.4 phase-3 spec_ema recursions on a prefilled state; the timm ModelEmaV3
profile is computed here from its get_decay rule. Mutants are separate code."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import bf16_rn, flag, state_dict, vec
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R

ema = v24.module("spec_ema")


def _inputs(profile, inp):
    if profile not in ("swa_ema", "swa_equal", "timm_v3"):
        raise Unsupported(f"profile {profile!r}")
    st = state_dict(inp, {"avg", "n"})
    avg = vec(st["avg"], "avg")
    n = integer(st["n"], "n", lo=0)
    cur = vec(inp.get("current"), "current", len(avg))
    hp = {}
    if profile in ("swa_ema", "timm_v3"):
        hp["decay"] = rational(inp.get("decay"), "decay")
        if not 0 <= hp["decay"] <= 1:
            raise Invalid("decay outside [0, 1]")
    if profile == "timm_v3":
        s = inp.get("step", "missing")
        if s == "missing":
            raise Invalid("missing step (null for update(model) without a step)")
        hp["step"] = None if s is None else integer(s, "step")
        hp["update_after_step"] = integer(inp.get("update_after_step", 0), "update_after_step", lo=0)
        hp["use_warmup"] = flag(inp, "use_warmup", default=False)
        hp["min_decay"] = rational(inp.get("min_decay", 0), "min_decay")
        hp["warmup_gamma"] = rational(inp.get("warmup_gamma", 1), "warmup_gamma")
        hp["warmup_power"] = rational(inp.get("warmup_power", Fraction(2, 3)), "warmup_power")
        if hp["warmup_gamma"] <= 0:
            raise Invalid("warmup_gamma must be > 0")
    return avg, n, cur, hp


def timm_decay(hp):
    if hp["step"] is None:
        return hp["decay"]
    s = max(0, hp["step"] - hp["update_after_step"] - 1)
    if s <= 0:
        return Fraction(0)
    if not hp["use_warmup"]:
        return hp["decay"]
    p = hp["warmup_power"]
    base = 1 + Fraction(s) / hp["warmup_gamma"]
    w = 1 - (base ** -int(p) if p.denominator == 1 else R(base) ** R(-p))
    # max(min(w, decay), min_decay)
    from kernel_spec.real import max_, min_
    return max_(min_(R(w), R(hp["decay"])), R(hp["min_decay"])) if isinstance(w, R) else \
        max(min(w, hp["decay"]), hp["min_decay"])


def reference(profile, inp):
    avg, n, cur, hp = _inputs(profile, inp)
    if profile == "timm_v3":
        d = timm_decay(hp)
        if isinstance(d, R):
            out = [R(a) + (R(1) - d) * (R(c) - R(a)) for a, c in zip(avg, cur)]
        else:
            out = [a + (1 - d) * (c - a) for a, c in zip(avg, cur)]
        return Result("value", {"avg": out, "n": n})
    st = ema.AveragedState()
    st.n_averaged, st.params = n, list(avg)
    if profile == "swa_ema":
        out = ema.ema_update(st, cur, hp["decay"])
    else:
        out = ema.swa_update(st, cur)
    return Result("value", {"avg": v24.tree(out), "n": st.n_averaged})


# ------------------------------------------------------------------ independent mutants
def _mut(profile, inp, mis):
    avg, n, cur, hp = _inputs(profile, inp)
    if profile == "timm_v3":
        if hp["step"] is None:
            d = hp["decay"]
        else:
            s = max(0, hp["step"] - hp["update_after_step"] - (0 if mis == "timm_off_by_one" else 1))
            if s <= 0:
                d = Fraction(0)
            elif hp["use_warmup"]:
                w = 1 - (1 + Fraction(s) / hp["warmup_gamma"]) ** -int(hp["warmup_power"])
                d = w if mis == "timm_no_clamp" else max(min(w, hp["decay"]), hp["min_decay"])
            else:
                d = hp["decay"]
        if mis == "swapped_weights":
            d = 1 - d
        return Result("value", {"avg": [d * a + (1 - d) * c for a, c in zip(avg, cur)], "n": n})
    if n == 0 and mis != "no_first_copy":
        new = list(cur)
    elif profile == "swa_ema":
        d = hp["decay"]
        if mis == "swapped_weights":
            d = 1 - d
        new = [d * a + (1 - d) * c for a, c in zip(avg, cur)]
    else:
        k = n if mis == "swa_divide_by_n" else n + 1
        new = [a + (c - a) / k for a, c in zip(avg, cur)]
    if mis == "bf16_storage":
        new = [bf16_rn(x) for x in new]
    return Result("value", {"avg": new, "n": n if mis == "counter_not_advanced" else n + 1})


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "EM.M1": _m("swapped_weights"),
    "EM.M2": _m("no_first_copy"),
    "EM.M3": _m("counter_not_advanced"),
    "EM.M4": _m("swa_divide_by_n"),
    "EM.M5": _m("bf16_storage"),
    "EM.M6": _m("timm_off_by_one"),
    "EM.M7": _m("timm_no_clamp"),
}
READINGS = {}
