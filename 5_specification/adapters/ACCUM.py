"""Adapter for contracts/ACCUM.yaml. The reference is kernel_spec.e2_train.accum_window; the mutants are an
independent implementation of the misreadings named in the contract."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import e2_train
from kernel_spec.num import Invalid, Result, Undefined


def reference(profile, inp):
    return Result("value", e2_train.accum_window(profile, inp))


READINGS = {}


def _mean(xs):
    return sum(xs, Fraction(0)) / len(xs)


def _independent(profile, inp, mis):
    mbs = [(mb.get("rank", 0), [Fraction(x) for x in mb["losses"]], [[Fraction(v) for v in g] for g in mb["grads"]])
           for mb in inp["microbatches"]]
    if mis != "drop_missing" and any(len(l) != len(g) for _, l, g in mbs):
        raise Invalid("gradient count")
    if mis == "drop_missing":
        mbs = [(r, l[:len(g)], g[:len(l)]) for r, l, g in mbs]
    toks = [(l, g) for _, ls, gs in mbs for l, g in zip(ls, gs)]
    dim = len(toks[0][1]) if toks else 1
    if not toks:
        if mis == "empty_zero":
            return Result("value", {"loss": Fraction(0), "grad": [Fraction(0)] * dim})
        raise Undefined("empty window")
    if mis == "mean_of_means":
        ne = [(ls, gs) for _, ls, gs in mbs if ls]
        loss = _mean([_mean(ls) for ls, _ in ne])
        grad = [_mean([_mean([g[k] for g in gs]) for _, gs in ne]) for k in range(dim)]
    elif mis == "div_k":
        k = len(mbs)
        loss = sum((_mean(ls) / k for _, ls, _ in mbs if ls), Fraction(0))
        grad = [sum((_mean([g[j] for g in gs]) / k for _, ls, gs in mbs if ls), Fraction(0)) for j in range(dim)]
    elif mis == "rank_mean":
        ranks = sorted({r for r, _, _ in mbs})
        per = [[(l, g) for r2, ls, gs in mbs if r2 == r for l, g in zip(ls, gs)] for r in ranks]
        per = [p for p in per if p]
        loss = _mean([_mean([l for l, _ in p]) for p in per])
        grad = [_mean([_mean([g[k] for _, g in p]) for p in per]) for k in range(dim)]
    else:
        n = len(toks)
        loss = sum(l for l, _ in toks) / n
        den = len(mbs) if mis == "grad_div_k" else n
        grad = [sum(g[k] for _, g in toks) / den for k in range(dim)]
    return Result("value", {"loss": loss, "grad": grad})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {"ACC.M1": _m("mean_of_means"), "ACC.M2": _m("div_k"), "ACC.M3": _m("rank_mean"), "ACC.M4": _m("grad_div_k"),
           "ACC.M5": _m("empty_zero"), "ACC.M6": _m("drop_missing")}
