"""Reference for contracts/SCHED.yaml: learning-rate sequences of CosineAnnealingLR, LinearLR, OneCycleLR and
transformers.get_cosine_schedule_with_warmup (LambdaLR), as lr_0 (after construction) ... lr_N (after N step() calls).

cos(pi q) is exact where q is a multiple of 1/2 or 1/3, otherwise a rigorous mpmath enclosure.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from .num import Invalid, Undefined, Unsupported, integer, rational, to_iv
from .real import R, max_ as rmax

SCHEDULES = ("cosine_annealing", "linear", "one_cycle", "hf_cosine_warmup")
PROFILES = ("closed_form", "recursive")


def cospi(q) -> R:
    """cos(pi * q) for a rational q."""
    q = Fraction(q) % 2
    exact = {Fraction(0): 1, Fraction(1, 2): 0, Fraction(1): -1, Fraction(3, 2): 0, Fraction(1, 3): Fraction(1, 2),
             Fraction(2, 3): Fraction(-1, 2), Fraction(4, 3): Fraction(-1, 2), Fraction(5, 3): Fraction(1, 2)}
    if q in exact:
        return R(exact[q])
    return R(iv.cos(iv.pi * to_iv(q)))


def _common(inp):
    n = integer(inp["steps"], "steps", 0)
    k = inp.get("resume_at")
    if k is not None:
        k = integer(k, "resume_at", 0)
        if k > n:
            raise Invalid("resume_at exceeds steps")
    ext = {}
    for t, v in (inp.get("external_lr") or {}).items():
        t = integer(t, "external_lr step", 0)
        if t >= n:
            raise Invalid("an external lr change after the last step has no effect on the sequence")
        ext[t] = R(rational(v, "external_lr value"))
    return n, ext


def cosine(inp, profile, n, ext, doc_recursion_only=False):
    base, eta = R(rational(inp["base_lr"], "base_lr")), R(rational(inp.get("eta_min", 0), "eta_min"))
    T = integer(inp["T_max"], "T_max")
    if T < 1:
        raise Invalid("T_max is a number of iterations (>= 1)")
    if profile == "closed_form":
        return [eta + (base - eta) * (1 + cospi(Fraction(t, T))) / 2 for t in range(n + 1)]
    lrs = [base]
    for t in range(1, n + 1):
        prev = ext.get(t - 1, lrs[-1])
        if (t - 1 - T) % (2 * T) == 0:                     # 1 + cos(pi (t-1)/T) = 0: SCH.COS_SINGULAR
            if doc_recursion_only:
                raise Undefined("the documented recursion has a zero denominator at T_cur = (2k+1) T_max")
            lrs.append(prev + (base - eta) * (1 - cospi(Fraction(1, T))) / 2)
        else:
            lrs.append((1 + cospi(Fraction(t, T))) / (1 + cospi(Fraction(t - 1, T))) * (prev - eta) + eta)
    return lrs


def linear(inp, profile, n, ext, example_shift=False):
    base = R(rational(inp["base_lr"], "base_lr"))
    s, e = rational(inp.get("start_factor", Fraction(1, 3)), "start_factor"), rational(inp.get("end_factor", 1), "end_factor")
    T = integer(inp.get("total_iters", 5), "total_iters")
    if not 0 < s <= 1 or not 0 <= e <= 1:
        raise Invalid("start_factor must be in (0, 1] and end_factor in [0, 1]")
    if T < 1:
        raise Invalid("total_iters must be >= 1")
    if profile == "closed_form" or example_shift:
        sh = 1 if example_shift else 0
        return [base * (s + (e - s) * Fraction(min(t + sh, T), T)) for t in range(n + 1)]
    lrs = [base * s]
    for t in range(1, n + 1):
        prev = ext.get(t - 1, lrs[-1])
        lrs.append(prev if t > T else prev * (1 + R(e - s) / (T * s + (t - 1) * (e - s))))
    return lrs


def _anneal(kind, start, end, pct):
    if kind == "cos":
        return end + (start - end) / 2 * (cospi(pct) + 1)
    return (end - start) * pct + start


def one_cycle(inp, n):
    mx = R(rational(inp["max_lr"], "max_lr"))
    total = integer(inp["total_steps"], "total_steps")
    pct = rational(inp.get("pct_start", Fraction(3, 10)), "pct_start")
    div, fdiv = rational(inp.get("div_factor", 25), "div_factor"), rational(inp.get("final_div_factor", 10000), "final_div_factor")
    kind = inp.get("anneal_strategy", "cos")
    three = inp.get("three_phase", False)
    if total < 1 or not 0 <= pct <= 1 or kind not in ("cos", "linear") or div == 0 or fdiv == 0 or not isinstance(three, bool):
        raise Invalid("total_steps >= 1, 0 <= pct_start <= 1, anneal_strategy cos | linear, nonzero divisors required")
    if n > total:
        raise Invalid(f"step {n} exceeds total_steps {total} (the scheduler refuses it)")
    init = mx / div
    lo = init / fdiv
    if three:
        phases = [(pct * total - 1, init, mx), (2 * pct * total - 2, mx, init), (Fraction(total - 1), init, lo)]
    else:
        phases = [(pct * total - 1, init, mx), (Fraction(total - 1), mx, lo)]
    out = []
    for t in range(n + 1):
        start = Fraction(0)
        for i, (end, a, b) in enumerate(phases):
            if t <= end or i == len(phases) - 1:
                if end == start:
                    raise Undefined("phase of zero length")
                out.append(_anneal(kind, a, b, (t - start) / (end - start)))
                break
            start = end
    return out


def hf_cosine_warmup(inp, n):
    base = R(rational(inp["base_lr"], "base_lr"))
    w, T = integer(inp["num_warmup_steps"], "num_warmup_steps", 0), integer(inp["num_training_steps"], "num_training_steps", 0)
    c = rational(inp.get("num_cycles", Fraction(1, 2)), "num_cycles")
    out = []
    for t in range(n + 1):
        if t < w:
            lam = R(Fraction(t, max(1, w)))
        else:
            prog = Fraction(t - w, max(1, T - w))
            lam = rmax(R(0), (1 + cospi(2 * c * prog)) / 2)
        out.append(base * lam)
    return out


def evaluate(profile, inp, reading=None):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    sch = inp.get("schedule")
    if sch not in SCHEDULES:
        raise Unsupported(f"schedule {sch!r}")
    n, ext = _common(inp)
    if sch == "cosine_annealing":
        lrs = cosine(inp, profile, n, ext, reading == "doc_recursion_only")
    elif sch == "linear":
        lrs = linear(inp, profile, n, ext, reading == "linear_example_shift")
    elif sch == "one_cycle":
        lrs = one_cycle(inp, n)
    else:
        lrs = hf_cosine_warmup(inp, n)
    return {"lr": [v.v for v in lrs]}
