"""Adapter for contracts/STRUCTURE.yaml: the structure acceptance v1.1 issuer's task specifications f (T1..T7).

One profile per family; input is a dict of the family's tensors (nested lists of exact rationals) and scalars; the single
declared output `out` is the program's output mapping ({'mean', 'variance'} for T1, {'output'} for the others), as in the
issuer's programs/specs.py. Reference on kernel_spec.real; the T6 square root is a rigorous enclosure. The mutants are an
independent Fraction / mpmath-interval implementation of natural misreadings (the issuer's answer key for this set is not
available, see the contract).
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import f_issuer as I
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, to_iv
from kernel_spec.real import R

PROFILES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
KEYS = {"T1": ("x",), "T2": ("x", "s"), "T3": ("x", "w"), "T4": ("x", "split", "alpha", "beta", "gamma"), "T5": ("x",),
        "T6": ("x", "eps"), "T7": ("x", "w")}


def _inputs(profile, inp):
    I.profile(profile, PROFILES)
    I.keys(inp, KEYS[profile], profile)
    x = I.mat(inp["x"], "x")
    if not x[0]:
        raise Invalid("x must be a non-empty rank-2 tensor (issuer: SpecNotEstablished)")
    z = dict(x=x)
    if profile == "T2":
        z["s"] = I.scalar(inp["s"], "s")
    elif profile in ("T3", "T7"):
        z["w"] = I.vec(inp["w"], "w", len(x[0]))
    elif profile == "T4":
        sp = inp["split"]
        if isinstance(sp, bool) or not isinstance(sp, int) or not 0 <= sp <= len(x):
            raise Invalid("split must be a non-boolean integer in [0, R] (issuer: SpecNotEstablished)")
        z.update(split=sp, alpha=I.scalar(inp["alpha"], "alpha"), beta=I.scalar(inp["beta"], "beta"),
                 gamma=I.scalar(inp["gamma"], "gamma"))
    elif profile == "T6":
        eps = I.scalar(inp["eps"], "eps")
        if eps <= 0:
            raise Invalid("eps must be positive in this task contract (issuer: SpecNotEstablished)")
        z["eps"] = eps
    return z


def reference(profile, inp):
    z = _inputs(profile, inp)
    x, V = z["x"], I.v
    if profile == "T1":
        means, variances = [], []
        for row in x:
            m = sum((R(v) for v in row), R(0)) / len(row)
            means.append(V(m))
            variances.append(V(sum(((R(v) - m) * (R(v) - m) for v in row), R(0)) / len(row)))
        return Result("value", {"out": {"mean": means, "variance": variances}})
    if profile == "T2":
        out = []
        for row in x:
            acc, one = R(0), []
            for v in row:
                acc = acc + R(v) * z["s"]
                one.append(V(acc))
            out.append(one)
        return Result("value", {"out": {"output": out}})
    if profile in ("T3", "T7"):
        return Result("value", {"out": {"output": [V(sum((R(a) * b for a, b in zip(row, z["w"])), R(0))) for row in x]}})
    if profile == "T4":
        out = [[V(R(v) * z["alpha"]) for v in row] if r < z["split"] else [V(R(v) * z["beta"] + z["gamma"]) for v in row]
               for r, row in enumerate(x)]
        return Result("value", {"out": {"output": out}})
    if profile == "T5":
        out = []
        for row in x:
            m = sum((R(v) for v in row), R(0)) / len(row)
            out.append([V(R(v) - m) for v in row])
        return Result("value", {"out": {"output": out}})
    out = []
    for row in x:
        q = sum((R(v) * v for v in row), R(0)) / len(row) + z["eps"]
        r = R(1) / rl.sqrt(q)
        out.append([V(R(v) * r) for v in row])
    return Result("value", {"out": {"output": out}})


# ------------------------------------------------------------------------------------------------ mutants
def _indep(profile, inp, mis):
    z = _inputs(profile, inp)
    x = z["x"]
    if profile == "T1":
        mean, var = [], []
        for row in x:
            n = len(row)
            m = sum(row) / n
            mean.append(m)
            if mis == "unbiased":
                var.append(sum((v - m) ** 2 for v in row) / (n - 1))
            elif mis == "uncentered":
                var.append(sum(v * v for v in row) / n)
            else:
                var.append(sum((v - m) ** 2 for v in row) / n)
        return Result("value", {"out": {"mean": mean, "variance": var}})
    if profile == "T2":
        s = 1 if mis == "no_scale" else z["s"]
        out = []
        for row in x:
            pref = [sum(row[:i + 1], Fraction(0)) * s for i in range(len(row))]
            out.append([Fraction(0)] + pref[:-1] if mis == "exclusive" else pref)
        return Result("value", {"out": {"output": out}})
    if profile in ("T3", "T7"):
        dots = [sum((a * b for a, b in zip(row, z["w"])), Fraction(0)) for row in x]
        if mis == "mean":
            dots = [d / len(z["w"]) for d in dots]
        return Result("value", {"out": {"output": dots}})
    if profile == "T4":
        out = []
        for r, row in enumerate(x):
            first = r <= z["split"] if mis == "split_inclusive" else r < z["split"]
            if mis == "swapped":
                first = not first
            g_first = z["gamma"] if mis == "gamma_both" else 0
            out.append([v * z["alpha"] + g_first if first else v * z["beta"] + z["gamma"] for v in row])
        return Result("value", {"out": {"output": out}})
    if profile == "T5":
        if mis == "global_mean":
            allv = [v for row in x for v in row]
            g = sum(allv) / len(allv)
            return Result("value", {"out": {"output": [[v - g for v in row] for row in x]}})
        return Result("value", {"out": {"output": [[v - sum(row) / len(row) for v in row] for row in x]}})
    out = []
    for row in x:
        ss = sum(v * v for v in row)
        if mis == "eps_outside":
            den = iv.sqrt(to_iv(ss / len(row))) + to_iv(z["eps"])
        elif mis == "sum_not_mean":
            den = iv.sqrt(to_iv(ss + z["eps"]))
        else:
            den = iv.sqrt(to_iv(ss / len(row) + z["eps"]))
        out.append([to_iv(v) / den for v in row])
    return Result("value", {"out": {"output": out}})


def _m(mis):
    return lambda profile, inp: _indep(profile, inp, mis)


MUTANTS = {
    "ST.M1": _m("unbiased"),
    "ST.M2": _m("uncentered"),
    "ST.M3": _m("exclusive"),
    "ST.M4": _m("no_scale"),
    "ST.M5": _m("mean"),
    "ST.M6": _m("split_inclusive"),
    "ST.M7": _m("swapped"),
    "ST.M8": _m("gamma_both"),
    "ST.M9": _m("global_mean"),
    "ST.M10": _m("eps_outside"),
    "ST.M11": _m("sum_not_mean"),
}
READINGS = {}
