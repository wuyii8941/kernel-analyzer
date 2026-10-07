"""The independent specification (specs/phase1) evaluated on the inputs a candidate actually received.

Every function returns a dict of float64 enclosures {name: (lo, hi)} for every documented reading (and labelled error
variant) plus status flags.  Gradients are J^T v for the candidate's upstream v, computed from the specification
without finite differences: the CE spec's exact gradient intervals; avg_pool_backward (the exact adjoint); max pool
and amax/amin from the argmax sets (unique extrema only; ties go to the set checks); sum / mean / index_add / prod
from the derivative of the spec's own definitions (linear maps, and prod_grad_factors for products).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import common  # noqa: F401  (paths)
import spec_cross_entropy as sce
import spec_index_scatter as six
import spec_pooling as spo

def _ivec(vals):
    lo = np.array([common.enclose(v)[0] for v in vals], dtype=np.float64)
    hi = np.array([common.enclose(v)[1] for v in vals], dtype=np.float64)
    return lo, hi


def _readings_ce(cond, weights, eps):
    if cond["target"] == "index" and weights is not None and eps != 0:
        return ("R_A", "R_B", "R_C")
    return ("R_A",)


# ------------------------------------------------------------------------------------------------ cross-entropy

def ce(cond, logits, target, weights, eps, upstream):
    """logits [N][C] float64 (values the candidate received), target list (index) or [N][C] (prob), weights list or
    None, eps the received label_smoothing, upstream: list of N ints ('none') or one int (sum/mean)."""
    N, C = logits.shape
    red = cond["reduction"]
    prob = cond["target"] == "prob"
    tgt = [list(map(float, row)) for row in target] if prob else list(target)
    out = {"status": "ok", "readings": {}}
    for rd in _readings_ce(cond, weights, eps):
        try:
            r = sce.cross_entropy(logits.tolist(), tgt, weights=weights, ignore_index=cond["ignore_index"],
                                  reduction=red, label_smoothing=Fraction(eps), reading=rd, prob_target=prob)
        except sce.SpecInputError as exc:
            return {"status": "input_illegal", "reason": str(exc)}
        if not r.get("established", True):
            out["readings"][rd] = {"status": "not_established", "reason": r.get("reason")}
            continue
        if r["nan"]:                                          # CE-A3: undefined
            out["readings"][rd] = {"status": "undefined"}
            continue
        if red == "none":
            loss = _ivec(r["loss"])
            g = [[r["grad"][n][c] * int(upstream[n]) for c in range(C)] for n in range(N)]
        else:
            loss = _ivec([r["loss"]])
            g = [[r["grad"][n][c] * int(upstream) for c in range(C)] for n in range(N)]
        glo, ghi = _ivec([v for row in g for v in row])
        out["readings"][rd] = {"status": "ok", "loss": loss, "grad": (glo.reshape(N, C), ghi.reshape(N, C))}
    return out


# ------------------------------------------------------------------------------------------------ pooling

def _nest(x):
    return x.tolist()


def avg_pool(cond, x, upstream):
    kw = dict(kernel=list(cond["kernel"]), stride=list(cond["stride"]), padding=list(cond["padding"]),
              ceil_mode=cond["ceil_mode"], count_include_pad=cond["count_include_pad"],
              divisor_override=cond["divisor_override"])
    out = {"status": "ok", "readings": {}}
    for rd in ("R2", "R1"):
        try:
            y = spo.avg_pool(_nest(x), reading=rd, **kw)
            g = spo.avg_pool_backward(upstream.astype(np.int64).tolist(), list(x.shape), reading=rd, **kw)
        except spo.SpecInputError as exc:
            return {"status": "input_illegal", "reason": str(exc)}
        except ValueError as exc:                       # non-finite input: outside the avg spec
            return {"status": "out_of_scope", "reason": str(exc)}
        shape = np.array(y, dtype=object).shape
        out["readings"][rd] = {"status": "ok", "out": common.enclose_array(y, shape),
                               "grad": common.enclose_array(g, x.shape)}
    return out


def max_pool(cond, x, upstream):
    kw = dict(kernel=list(cond["kernel"]), stride=list(cond["stride"]), padding=list(cond["padding"]),
              dilation=list(cond["dilation"]), ceil_mode=cond["ceil_mode"])
    out = {"status": "ok", "readings": {}}
    for rd in ("R_prop", "R_ignore"):
        try:
            vals, sets = spo.max_pool(_nest(x), nan_reading=rd, **kw)
        except spo.SpecInputError as exc:
            return {"status": "input_illegal", "reason": str(exc)}
        except spo.SpecNotEstablished as exc:
            out["readings"][rd] = {"status": "not_established", "reason": str(exc)}
            continue
        flat_sets = []

        def walk(t):
            if isinstance(t, set):
                flat_sets.append(t)
            else:
                for e in t:
                    walk(e)
        walk(sets)
        oshape = upstream.shape
        lo, hi = common.enclose_array(vals, oshape)
        rec = {"status": "ok", "out": (lo, hi), "sets": flat_sets, "out_shape": oshape}
        # strict gradient only if every window has a unique argmax and no NaN participates
        has_nan = bool(np.isnan(lo).any())
        unique = all(len(s) <= 1 for s in flat_sets)
        if unique and not has_nan:
            g = np.zeros(x.shape)
            C = x.shape[0]
            per_c = len(flat_sets) // C
            for j, (v, s) in enumerate(zip(upstream.reshape(-1), flat_sets)):
                c = j // per_c
                for pos in s:
                    g[(c,) + tuple(pos)] += v
            rec["grad"] = (g, g.copy())
        else:
            rec["grad"] = None
            rec["grad_reason"] = "NaN participates" if has_nan else "ties: set checks only"
        out["readings"][rd] = rec
    return out


# ------------------------------------------------------------------------------------------------ index / scatter

def _groups(op, self_t, dim, index, source):
    if op in ("index_add", "index_reduce"):
        return six._groups_index(self_t, dim, index, source)
    return six._groups_scatter(self_t, dim, index, source)


def _src_positions(op, self_t, dim, index, source):
    """same traversal order as the spec's grouping: {target position: [source positions]}."""
    pos = {}
    nd = 1 if not isinstance(self_t[0], list) else 2
    if op in ("index_add", "index_reduce"):
        if nd == 1:
            for i, j in enumerate(index):
                pos.setdefault((j,), []).append((i,))
        else:
            R, Cn = len(self_t), len(self_t[0])
            for i, j in enumerate(index):
                if dim == 0:
                    for c in range(Cn):
                        pos.setdefault((j, c), []).append((i, c))
                else:
                    for r in range(R):
                        pos.setdefault((r, j), []).append((r, i))
    else:
        if nd == 1:
            for i, j in enumerate(index):
                pos.setdefault((j,), []).append((i,))
        else:
            for r in range(len(index)):
                for c in range(len(index[0])):
                    tgt = (index[r][c], c) if dim == 0 else (r, index[r][c])
                    pos.setdefault(tgt, []).append((r, c))
    return pos


def index_family(cond, self_np, index, source_np, dim, upstream):
    """forward (all documented readings; for mean with include_self also the labelled error variant) and the exact
    J^T v w.r.t. self and source.  amax/amin: strict gradient only at unique extrema; ties -> set check data."""
    op = cond["op"]
    red = cond.get("reduce", "sum")
    inc = cond.get("include_self", True)
    self_t, source = self_np.tolist(), source_np.tolist()
    out = {"status": "ok", "readings": {}}
    variants = [("main", {})]
    if red in ("amax", "amin"):
        variants = [("R_prop", {"nan_reading": "R_prop"}), ("R_ignore", {"nan_reading": "R_ignore"})]
    if red == "mean" and inc:
        variants.append(("error_variant_mean_divides_by_contrib_count", {"error_variant": "mean_divides_by_contrib_count"}))
    for name, kw in variants:
        try:
            if op == "index_add":
                y = six.index_add(self_t, dim, index, source, alpha=Fraction(cond["alpha"]))
            elif op == "index_reduce":
                y = six.index_reduce(self_t, dim, index, source, red, include_self=inc, **kw)
            else:
                y = six.scatter_reduce(self_t, dim, index, source, red, include_self=inc, **kw)
        except six.SpecInputError as exc:
            return {"status": "input_illegal", "reason": str(exc)}
        except six.SpecNotEstablished as exc:
            out["readings"][name] = {"status": "not_established", "reason": str(exc)}
            continue
        except ValueError as exc:
            out["readings"][name] = {"status": "out_of_scope", "reason": str(exc)}
            continue
        out["readings"][name] = {"status": "ok", "out": common.enclose_array(y, self_np.shape)}
    # gradients (J^T v), from the derivative of the definitions
    if np.isnan(self_np).any() or np.isnan(source_np).any() or not np.isfinite(source_np).all() or not np.isfinite(self_np).all():
        out["grad"] = None
        out["grad_reason"] = "non-finite inputs"
        return out
    groups = _groups(op, self_t, dim, index, source)
    srcpos = _src_positions(op, self_t, dim, index, source)
    v = upstream
    # J^T v w.r.t. self starts as the identity (positions nothing is reduced into keep their value)
    g_self = [[Fraction(int(e)) for e in row] for row in v.tolist()] if v.ndim == 2 else [Fraction(int(e)) for e in v.tolist()]
    g_src = (np.zeros(source_np.shape, dtype=object) + Fraction(0))
    ties = []

    def setv(arr, pos, val):
        if isinstance(arr, np.ndarray):
            arr[pos] = val
        elif len(pos) == 1:
            arr[pos[0]] = val
        else:
            arr[pos[0]][pos[1]] = val

    def getv(arr, pos):
        return arr[pos[0]] if len(pos) == 1 else arr[pos[0]][pos[1]]

    for pos, contrib in groups.items():
        up = Fraction(int(v[pos]))
        base = Fraction(getv(self_t, pos))
        parts = ([base] if (inc or op == "index_add") else []) + [Fraction(c) for c in contrib]
        sps = srcpos[pos]
        if op == "index_add":
            alpha = Fraction(cond["alpha"])
            d_self, d_src = Fraction(1), [alpha] * len(sps)
        elif red == "sum":
            d_self, d_src = (Fraction(1) if inc else Fraction(0)), [Fraction(1)] * len(sps)
        elif red == "mean":
            n = len(parts)
            d_self, d_src = (Fraction(1, n) if inc else Fraction(0)), [Fraction(1, n)] * len(sps)
        elif red == "prod":
            ds = six.prod_grad_factors(parts)
            d_self, d_src = (ds[0], ds[1:]) if inc else (Fraction(0), ds)
        else:                                            # amax / amin
            vals = parts if red == "amax" else [-p for p in parts]
            S = six.maximisers(vals)
            if len(S) > 1:
                ties.append({"target": pos, "values": [float(p) for p in parts], "self_included": bool(inc),
                             "upstream": int(v[pos]), "source_positions": sps})
                d_self, d_src = None, None
            else:
                (w,) = S
                coeff = [Fraction(1) if i == w else Fraction(0) for i in range(len(parts))]
                d_self, d_src = (coeff[0], coeff[1:]) if inc else (Fraction(0), coeff)
        if d_self is None:
            setv(g_self, pos, None)
            for sp in sps:
                g_src[sp] = None
            continue
        setv(g_self, pos, up * d_self)
        for sp, d in zip(sps, d_src):
            g_src[sp] = g_src[sp] + up * d if g_src[sp] is not None else None
    def enc(arr, shape):
        flat = np.array(arr, dtype=object).reshape(-1)
        lo = np.array([np.nan if e is None else common.enclose(e)[0] for e in flat]).reshape(shape)
        hi = np.array([np.nan if e is None else common.enclose(e)[1] for e in flat]).reshape(shape)
        return lo, hi
    out["grad"] = {"self": enc(g_self, self_np.shape), "source": enc(g_src, source_np.shape)}
    out["ties"] = ties
    return out
