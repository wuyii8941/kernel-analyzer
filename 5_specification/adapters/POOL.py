"""Adapter for contracts/POOL.yaml. The reference is kernel_spec.e2_pool (real.R arithmetic). The mutants are an
independent implementation (plain Fractions and float infinities, written separately from the reference), each
switching on one misreading named in the contract's mutation requirements."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

from kernel_spec import e2_pool
from kernel_spec.num import Invalid, Result, Undefined, Unsupported


def reference(profile, inp):
    return Result("value", e2_pool.evaluate(profile, inp))


READINGS = {
    "R1_kernel_divisor": lambda profile, inp: Result("value", e2_pool.evaluate(profile, inp, divisor_reading="R1")),
    "R_ignore": lambda profile, inp: Result("value", e2_pool.evaluate(profile, inp, nan_reading="ignore")),
    "shape_literal_1d": lambda profile, inp: Result("value", e2_pool.evaluate(profile, inp, literal_1d=True)),
    "min_norm": lambda profile, inp: Result("value", e2_pool.evaluate(profile, inp, min_norm=True)),
}


# ------------------------------------------------------------------ independent implementation for the mutants
INF, NAN = float("inf"), float("nan")


def _leaf(v):
    if isinstance(v, str) and v.strip() in ("inf", "+inf", "-inf", "nan"):
        return {"inf": INF, "+inf": INF, "-inf": -INF, "nan": NAN}[v.strip()]
    return Fraction(v)


def _walk(t, fn):
    return [_walk(e, fn) for e in t] if isinstance(t, list) else fn(t)


def _get(t, idx):
    for i in idx:
        t = t[i]
    return t


def _shape(t):
    s = []
    while isinstance(t, list):
        s.append(len(t))
        t = t[0]
    return s


def _tup(v, n, mis, default=None):
    if v is None:
        return default
    if isinstance(v, list):
        if len(v) == 1:
            return (v[0],) + ((1,) * (n - 1) if mis == "one_elem_first" else (v[0],) * (n - 1))
        return tuple(v)
    return (v,) * n


def _olen(L, k, s, p, d, ceil, mis):
    num = L + 2 * p - d * (k - 1) - 1
    if ceil and mis != "floor_only":
        n = -((-num) // s) + 1
        if mis != "no_correction" and (n - 1) * s >= L + p:
            n -= 1
        return n
    return num // s + 1


def _out(v):
    if isinstance(v, float):
        return frozenset() if v != v else ("+inf" if v > 0 else "-inf")
    return v


def _independent(profile, inp, mis):
    op = inp["op"]
    kind, nd = op.split("_pool")[0], int(op[-2])
    x = _walk(inp["x"], _leaf)
    shp = _shape(x)
    lead, Ls = shp[:-nd], shp[-nd:]
    flat_nan = any(isinstance(v, float) and v != v for v in _flatten(x))
    if flat_nan and (kind == "avg" or profile == "real"):
        raise Undefined("NaN input")
    if kind == "adaptive_max":
        n = inp["output_size"] if not isinstance(inp["output_size"], list) else inp["output_size"][0]
        L = Ls[0]
        ends = [((i + 1) * L) // n if mis == "adaptive_floor_end" else ((i + 1) * L + n - 1) // n for i in range(n)]
        wins = [[list(range((i * L) // n, ends[i])) for i in range(n)]]
        outs, P, k = [n], None, None
    else:
        k = _tup(inp["kernel"], nd, mis)
        s = _tup(inp.get("stride"), nd, mis, k)
        p = _tup(inp.get("padding", 0), nd, mis)
        d = _tup(inp.get("dilation", 1), nd, mis) if kind == "max" and mis != "no_dilation" else (1,) * nd
        ceil = inp.get("ceil_mode", False)
        outs = [_olen(Ls[i], k[i], s[i], p[i], d[i], ceil, mis) for i in range(nd)]
        wins = [[[s[i] * o - p[i] + d[i] * m for m in range(k[i])] for o in range(outs[i])] for i in range(nd)]
        P = [(-p[i], Ls[i] + p[i]) for i in range(nd)]
    up = inp["upstream"]
    planes = list(itertools.product(*[range(n) for n in lead]))
    grad = {}
    out, arg = {}, {}
    for pl in planes:
        for o in itertools.product(*[range(n) for n in outs]):
            cells = list(itertools.product(*[wins[i][o[i]] for i in range(nd)]))
            real = [c for c in cells if all(0 <= c[i] < Ls[i] for i in range(nd))]
            try:
                v = Fraction(_get(up, pl + o))
            except (IndexError, TypeError):
                v = Fraction(0)
            if kind == "avg":
                tot = sum((_get(x, pl + c) for c in real), Fraction(0))
                inP = sum(1 for c in cells if all(P[i][0] <= c[i] < P[i][1] for i in range(nd)))
                cip = inp.get("count_include_pad", True)
                ov = inp.get("divisor_override")
                if ov is not None and mis != "override_ignored":
                    dv = dbw = ov
                else:
                    dv = math.prod(k) if (cip and mis == "kernel_div") else (len(real) if (not cip and mis != "cip_ignored") else inP)
                    dbw = math.prod(k) if (cip and mis in ("kernel_div", "bwd_kernel_div")) else dv
                out[pl + o] = tot / dv
                for c in real:
                    grad[pl + c] = grad.get(pl + c, 0) + (v if mis == "grad_no_div" else v / dbw)
                continue
            cand = [(c, _get(x, pl + c)) for c in real]
            if mis == "zero_pad":
                cand += [(None, Fraction(0)) for c in cells if c not in real
                         and all(P[i][0] <= c[i] < P[i][1] for i in range(nd))]
            nans = [c for c, val in cand if isinstance(val, float) and val != val]
            plane_nan = any(isinstance(val, float) and val != val for val in _flatten(_get(x, pl)))
            if (mis == "nan_plane" and plane_nan) or (nans and mis != "nan_skip"):
                out[pl + o], arg[pl + o] = NAN, frozenset(nans) if nans else frozenset()
                continue
            cand = [(c, val) for c, val in cand if not (isinstance(val, float) and val != val)]
            if not cand:
                out[pl + o], arg[pl + o] = -INF, frozenset()
                if mis == "empty_next_plane":
                    tgt = ((pl[0] + 1) % lead[0],) + pl[1:] + (0,) * nd
                    grad[tgt] = grad.get(tgt, 0) + v
                continue
            best = max(val for _, val in cand)
            S = [c for c, val in cand if val == best and c is not None]
            if mis == "first_tie":
                S = S[:1]
            if mis == "neginf_empty" and best == -INF:
                S = []
            out[pl + o], arg[pl + o] = best, frozenset(S)
            if len(S) == 1:
                grad[pl + S[0]] = grad.get(pl + S[0], 0) + v
    def build(shape, fn, prefix=()):
        return fn(prefix) if not shape else [build(shape[1:], fn, prefix + (i,)) for i in range(shape[0])]
    oshape = list(lead) + list(outs)
    fields = {"out": build(oshape, lambda i: _out(out[i])),
              "grad": build(list(shp), lambda i: Fraction(grad.get(i, 0))),
              "argmax": [] if kind == "avg" else build(oshape, lambda i: arg[i])}
    return Result("value", fields)


def _flatten(t):
    if isinstance(t, list):
        for e in t:
            yield from _flatten(e)
    else:
        yield t


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "POOL.M1": _m("bwd_kernel_div"),
    "POOL.M2": _m("kernel_div"),
    "POOL.M3": _m("cip_ignored"),
    "POOL.M4": _m("override_ignored"),
    "POOL.M5": _m("no_correction"),
    "POOL.M6": _m("floor_only"),
    "POOL.M7": _m("no_dilation"),
    "POOL.M8": _m("zero_pad"),
    "POOL.M9": _m("empty_next_plane"),
    "POOL.M10": _m("first_tie"),
    "POOL.M11": _m("nan_skip"),
    "POOL.M12": _m("nan_plane"),
    "POOL.M13": _m("neginf_empty"),
    "POOL.M14": _m("adaptive_floor_end"),
    "POOL.M15": _m("one_elem_first"),
    "POOL.M16": _m("grad_no_div"),
}
