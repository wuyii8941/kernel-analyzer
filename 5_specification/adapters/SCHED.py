"""Adapter for contracts/SCHED.yaml. The reference is kernel_spec.e2_sched; the mutants are an independent
implementation (mpmath interval cosines throughout), each switching on one misreading named in the contract."""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import e2_sched
from kernel_spec.num import Invalid, Result, to_iv


def reference(profile, inp):
    return Result("value", e2_sched.evaluate(profile, inp))


READINGS = {
    "doc_recursion_only": lambda p, i: Result("value", e2_sched.evaluate(p, i, reading="doc_recursion_only")),
    "linear_example_shift": lambda p, i: Result("value", e2_sched.evaluate(p, i, reading="linear_example_shift")),
}


# ------------------------------------------------------------------ independent implementation for the mutants
def _q(x):
    return Fraction(x) if not isinstance(x, str) else Fraction(x.replace(" ", ""))


def _cos_pi(q):
    return iv.cos(iv.pi * to_iv(Fraction(q)))


def _independent(profile, inp, mis):
    s, n = inp["schedule"], int(inp["steps"])
    k = inp.get("resume_at")
    ext = {int(t): _q(v) for t, v in (inp.get("external_lr") or {}).items()}
    out = []
    if s == "cosine_annealing":
        b, e, T = _q(inp["base_lr"]), _q(inp.get("eta_min", 0)), int(inp["T_max"])

        def closed(t):
            tc = t
            if mis == "clamp":
                tc = min(t, T)
            elif mis == "restart":
                tc = t % T
            elif mis == "shift":
                tc = t + 1
            elif mis == "resume_restart" and k is not None and t > k:
                tc = t - k
            return to_iv(e) + to_iv(b - e) * (1 + _cos_pi(Fraction(tc, T))) / 2
        if profile == "closed_form":
            return [closed(t) for t in range(n + 1)]
        out = [to_iv(b)]
        for t in range(1, n + 1):
            prev = out[-1] if mis == "ignore_external" else (to_iv(ext[t - 1]) if t - 1 in ext else out[-1])
            if (t - 1 - T) % (2 * T) == 0:
                out.append(closed(t) if mis == "singular_closed" else prev + to_iv(b - e) * (1 - _cos_pi(Fraction(1, T))) / 2)
            else:
                out.append((1 + _cos_pi(Fraction(t, T))) / (1 + _cos_pi(Fraction(t - 1, T))) * (prev - to_iv(e)) + to_iv(e))
        return out
    if s == "linear":
        b, a, z, T = _q(inp["base_lr"]), _q(inp["start_factor"]), _q(inp.get("end_factor", 1)), int(inp["total_iters"])
        tt = (lambda t: t) if mis == "linear_no_hold" else (lambda t: min(t, T))
        if profile == "closed_form":
            return [b * (a + (z - a) * Fraction(tt(t), T)) for t in range(n + 1)]
        out = [b * a]
        for t in range(1, n + 1):
            prev = ext.get(t - 1, out[-1])
            out.append(prev if (t > T and mis != "linear_no_hold") else prev * (1 + (z - a) / (T * a + (t - 1) * (z - a))))
        return out
    if s == "one_cycle":
        mx, total = _q(inp["max_lr"]), int(inp["total_steps"])
        pct = _q(inp.get("pct_start", Fraction(3, 10)))
        init = mx if mis == "no_div" else mx / _q(inp.get("div_factor", 25))
        lo = init / _q(inp.get("final_div_factor", 10000)) if mis != "no_div" else mx / 25 / 10000
        if n > total and mis != "extrapolate":
            raise Invalid("beyond total_steps")
        e1 = pct * total - (0 if mis == "no_minus1" else 1)
        for t in range(n + 1):
            if t <= e1:
                p, a, z = Fraction(t) / e1, init, mx
            else:
                p, a, z = (t - e1) / (total - 1 - e1), mx, lo
            out.append(to_iv(z) + to_iv(a - z) / 2 * (_cos_pi(p) + 1))
        return out
    b, w, T = _q(inp["base_lr"]), int(inp["num_warmup_steps"]), int(inp["num_training_steps"])
    c = _q(inp.get("num_cycles", Fraction(1, 2)))
    if mis == "refuse_short" and T <= w:
        raise Invalid("total <= warmup refused")
    for t in range(n + 1):
        if t < w:
            out.append(b * Fraction(t + (1 if mis == "warm_from_1" else 0), max(1, w)))
        elif mis == "hold_zero" and t >= T:
            out.append(Fraction(0))
        else:
            v = (1 + _cos_pi(2 * c * Fraction(t - w, max(1, T - w)))) / 2
            out.append(to_iv(b) * iv.mpf([max(v.a, 0), max(v.b, 0)]))
    return out


def _m(kind):
    return lambda profile, inp: Result("value", {"lr": _independent(profile, inp, kind)})


MUTANTS = {
    "SCH.M1": _m("clamp"), "SCH.M2": _m("restart"), "SCH.M3": _m("shift"), "SCH.M4": _m("resume_restart"),
    "SCH.M5": _m("linear_no_hold"), "SCH.M6": _m("no_minus1"), "SCH.M7": _m("no_div"), "SCH.M8": _m("extrapolate"),
    "SCH.M9": _m("refuse_short"), "SCH.M10": _m("hold_zero"), "SCH.M11": _m("warm_from_1"),
    "SCH.M12": _m("singular_closed"), "SCH.M13": _m("ignore_external"),
}
