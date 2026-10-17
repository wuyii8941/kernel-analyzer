"""Adapter for contracts/INDEXREDUCE.yaml. The reference is kernel_spec.e2_index (real.R arithmetic); the mutants are
an independent implementation (plain Fractions over Python lists), each switching on one misreading named in the
contract's mutation requirements."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

from kernel_spec import e2_index
from kernel_spec.num import Result, Undefined


def reference(profile, inp):
    return Result("value", e2_index.evaluate(profile, inp))


READINGS = {"even_split": lambda profile, inp: Result("value", e2_index.evaluate(profile, inp, tie_reading="even"))}


# ------------------------------------------------------------------ independent implementation for the mutants
def _shape(t):
    s = []
    while isinstance(t, list):
        s.append(len(t))
        t = t[0] if t else None
    return s


def _get(t, i):
    for k in i:
        t = t[k]
    return t


def _build(shape, fn, prefix=()):
    return fn(prefix) if not shape else [_build(shape[1:], fn, prefix + (i,)) for i in range(shape[0])]


def _independent(profile, inp, mis):
    op = inp["op"]
    if any(isinstance(v, str) and v == "nan" for v in str(inp).split("'")):
        raise Undefined("NaN")
    S, X = inp["self"], inp["src"]
    ss, xs = _shape(S), _shape(X)
    nd = len(ss)
    dim = 0 if mis == "dim0" else inp["dim"] % nd
    red = "sum" if op == "index_add" else inp["reduce"]
    inc = True if op == "index_add" else inp.get("include_self", True)
    if mis == "ignore_include_self":
        inc = True
    alpha = Fraction(inp.get("alpha", 1)) if op == "index_add" and mis != "no_alpha" else Fraction(1)
    I = inp["index"]
    contrib = {}                                   # target -> list of (src coordinate, value)
    for q in itertools.product(*[range(n) for n in xs]):
        if op == "scatter_reduce":
            j = _get(I, q)
        elif mis == "index_selects_source":
            j = None
        else:
            j = I[q[dim]]
        if j is None:
            continue
        t = q[:dim] + (j,) + q[dim + 1:]
        contrib.setdefault(t, []).append((q, alpha * Fraction(_get(X, q))))
    if mis == "index_selects_source":            # self[i] += alpha * source[index[i]] along dim
        for i, j in enumerate(I):
            for q in itertools.product(*[range(n) for n in xs]):
                if q[dim] == j:
                    t = q[:dim] + (i,) + q[dim + 1:]
                    contrib.setdefault(t, []).append((q, alpha * Fraction(_get(X, q))))
    up = inp["upstream"]
    out, gs, gq, ties = {}, {}, {}, {}
    for p in itertools.product(*[range(n) for n in ss]):
        u = Fraction(_get(up, p))
        base = Fraction(_get(S, p))
        cs = contrib.get(p, [])
        if not cs:
            if mis == "identity_everywhere" and not inc:
                ident = {"sum": 0, "prod": 1, "amax": -math.inf, "amin": math.inf, "mean": 0}[red]
                out[p], gs[p], ties[p] = ident, Fraction(0), frozenset()
            else:
                out[p], gs[p], ties[p] = base, u, frozenset()
            continue
        if mis == "lanes8" and math.prod(ss) == 1:
            cs = cs + [cs[-1]] * ((-len(cs)) % 8)
        parts = ([((-1,), base)] if inc else []) + cs
        vals = [v for _, v in parts]
        g = {}
        if red == "sum":
            out[p] = sum(vals)
            g = {k: u for k, _ in parts}
        elif red == "prod":
            out[p] = math.prod(vals)
            if mis == "prod_divide":
                g = {k: (u * out[p] / v if v != 0 else Fraction(0)) for k, v in parts}
            else:
                g = {k: u * math.prod(w for kk, w in parts if kk != k) for k, _ in parts}
        elif red == "mean":
            n = len(cs) if (mis == "mean_div_k" and inc) else len(parts)
            out[p] = sum(vals) / n
            g = {k: (u if mis == "mean_grad_nodiv" else u / len(parts)) for k, _ in parts}
        else:
            best = max(vals) if red == "amax" else min(vals)
            T = [k for k, v in parts if v == best]
            if mis == "b020" and not inc and base == best:
                g = {k: u / (len(T) + 1) for k in T}
            elif mis == "first_tie":
                g = {T[0]: u}
            elif len(T) == 1:
                g = {T[0]: u}
            out[p], ties[p] = best, frozenset(T)
        gs[p] = g.get((-1,), Fraction(0))
        for q, _ in cs:
            gq[q] = gq.get(q, Fraction(0)) + g.get(q, Fraction(0)) * (alpha if op == "index_add" else 1)
    conv = lambda v: ("+inf" if v > 0 else "-inf") if isinstance(v, float) else Fraction(v)   # noqa: E731
    return Result("value", {
        "out": _build(ss, lambda p: conv(out[p])),
        "grad_self": _build(ss, lambda p: Fraction(gs[p])),
        "grad_src": _build(xs, lambda q: gq.get(q, Fraction(0))),
        "ties": _build(ss, lambda p: ties[p]) if red in ("amax", "amin") else []})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "IR.M1": _m("no_alpha"),
    "IR.M2": _m("ignore_include_self"),
    "IR.M3": _m("mean_div_k"),
    "IR.M4": _m("prod_divide"),
    "IR.M5": _m("first_tie"),
    "IR.M6": _m("b020"),
    "IR.M7": _m("identity_everywhere"),
    "IR.M8": _m("lanes8"),
    "IR.M9": _m("dim0"),
    "IR.M10": _m("index_selects_source"),
    "IR.M11": _m("mean_grad_nodiv"),
}
