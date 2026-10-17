"""Adapter for contracts/SORT.yaml: torch.sort / torch.argsort / torch.topk, exact values and allowed index sets.

ATen-style input {op: aten.sort | aten.argsort | aten.topk, args, kwargs} (kernel_spec.aten_adapter conventions). Along
the chosen dimension the values are the sorted (or k largest / smallest, sorted) elements, exactly. Indices are exact when
the order is determined (distinct values, or stable=True); otherwise each position carries the allowed set of original
indices whose value equals the value at that position (contract decision SO.INDEX_SETS). MUTANTS are independent code.
"""
from __future__ import annotations

import builtins

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "SORT.yaml").read_text())["covers"]


def _key(x):
    """Total order key of an exact extended real."""
    v = x.v
    if v == "-inf":
        return (-1, 0)
    if v == "+inf":
        return (1, 0)
    if not x.exact:
        raise Invalid("sort of an enclosure")
    return (0, v)


def _lines(a, dim):
    a = A.as_ten(a)
    if a.a.ndim == 0:
        return a, None
    d = dim % a.a.ndim
    return a, d


def _apply(a, dim, fn):
    """fn(list of values) -> (values, indices) along dim; rebuilt into two tensors."""
    a, d = _lines(a, dim)
    if d is None:
        v, i = fn([a.a[()]])
        return A.Ten(np.array(v[0], dtype=object), a.dtype), A.Ten(np.array(i[0], dtype=object), "i64")
    moved = np.moveaxis(a.a, d, -1)
    lead = moved.shape[:-1]
    outs_v, outs_i = None, None
    for idx in np.ndindex(*lead):
        v, i = fn(list(moved[idx]))
        if outs_v is None:
            outs_v = np.empty(lead + (len(v),), dtype=object)
            outs_i = np.empty(lead + (len(v),), dtype=object)
        for j in range(len(v)):
            outs_v[idx + (j,)] = v[j]
            outs_i[idx + (j,)] = i[j]
    if outs_v is None:
        outs_v = np.empty(lead + (0,), dtype=object)
        outs_i = np.empty(lead + (0,), dtype=object)
    return A.Ten(np.moveaxis(outs_v, -1, d), a.dtype), A.Ten(np.moveaxis(outs_i, -1, d), "i64")


def _index_sets(xs, order, exact):
    out = []
    for j, i in enumerate(order):
        if exact:
            out.append(i)
            continue
        same = [m for m in range(len(xs)) if _key(xs[m]) == _key(xs[i])]
        out.append(i if len(same) == 1 else A.AllowedSet(sorted(same)))
    return out


def sort(a, dim=-1, descending=False, stable=False):
    def line(xs):
        if descending:
            order = sorted(range(len(xs)), key=lambda i: (tuple(-c for c in _key(xs[i])), i))
        else:
            order = sorted(range(len(xs)), key=lambda i: (_key(xs[i]), i))
        return [xs[i] for i in order], _index_sets(xs, order, stable)
    return list(_apply(a, dim, line))


def argsort(a, dim=-1, descending=False, stable=False):
    return sort(a, dim, descending, stable)[1]


def topk(a, k, dim=-1, largest=True, sorted=True):
    if not sorted:
        raise Unsupported("topk with sorted=False: the order of the k elements is unspecified (decision SO.UNSORTED)")
    a = A.as_ten(a)
    n = a.a.shape[dim % a.a.ndim] if a.a.ndim else 1
    if not isinstance(k, int) or k < 0 or k > n:
        raise Invalid(f"topk: k = {k} out of range for a dimension of size {n}")

    def line(xs):
        if largest:
            order = builtins.sorted(range(len(xs)), key=lambda i: (tuple(-c for c in _key(xs[i])), i))[:k]
        else:
            order = builtins.sorted(range(len(xs)), key=lambda i: (_key(xs[i]), i))[:k]
        return [xs[i] for i in order], _index_sets(xs, order, False)
    return list(_apply(a, dim, line))


EXTRA = {"aten.sort": sort, "aten.argsort": argsort, "aten.topk": topk}
reference = make_reference(COVERS, extra=EXTRA)


# ------------------------------------------------------------------ mutants (independent implementations)
def _num(x):
    v = x.v
    return float("-inf") if v == "-inf" else float("inf") if v == "+inf" else v


def _m_sort(kind):
    def f(a, dim=-1, descending=False, stable=False):
        a = A.as_ten(a)
        d = (-1 if kind == "dim_ignored" else dim) % a.a.ndim
        moved = np.moveaxis(a.a, d, -1)
        vals = np.empty(moved.shape, dtype=object)
        inds = np.empty(moved.shape, dtype=object)
        for idx in np.ndindex(*moved.shape[:-1]):
            xs = list(moved[idx])
            desc = descending and kind != "descending_ignored"
            order = sorted(range(len(xs)), key=lambda i: (_num(xs[i]), i))
            if desc:
                order = order[::-1] if kind == "stable_desc_reversed" else sorted(
                    range(len(xs)), key=lambda i: (-_num(xs[i]), i))
            for j, i in enumerate(order):
                vals[idx + (j,)] = xs[i]
                if stable or kind == "first_occurrence":
                    inds[idx + (j,)] = i
                else:
                    same = [m for m in range(len(xs)) if _num(xs[m]) == _num(xs[i])]
                    inds[idx + (j,)] = i if len(same) == 1 else A.AllowedSet(same)
        return [A.Ten(np.moveaxis(vals, -1, d), a.dtype), A.Ten(np.moveaxis(inds, -1, d), "i64")]
    return f


def _m_topk(kind):
    def f(a, k, dim=-1, largest=True, sorted=True):
        a = A.as_ten(a)
        d = dim % a.a.ndim
        moved = np.moveaxis(a.a, d, -1)
        vals = np.empty(moved.shape[:-1] + (k,), dtype=object)
        inds = np.empty(moved.shape[:-1] + (k,), dtype=object)
        for idx in np.ndindex(*moved.shape[:-1]):
            xs = list(moved[idx])
            big = True if kind == "largest_ignored" else largest
            order = builtins.sorted(range(len(xs)), key=lambda i: _num(xs[i]), reverse=big)[:k]
            if kind == "ascending_output":
                order = order[::-1]
            for j, i in enumerate(order):
                vals[idx + (j,)] = xs[i]
                same = [m for m in range(len(xs)) if _num(xs[m]) == _num(xs[i])]
                inds[idx + (j,)] = i if len(same) == 1 else A.AllowedSet(same)
        return [A.Ten(np.moveaxis(vals, -1, d), a.dtype), A.Ten(np.moveaxis(inds, -1, d), "i64")]
    return f


def _m(**ops):
    return make_reference(COVERS, extra=dict(EXTRA, **ops))


MUTANTS = {
    "SO.M1": _m(**{"aten.sort": _m_sort("descending_ignored")}),
    "SO.M2": _m(**{"aten.sort": _m_sort("stable_desc_reversed")}),
    "SO.M3": _m(**{"aten.sort": _m_sort("first_occurrence")}),
    "SO.M4": _m(**{"aten.sort": _m_sort("dim_ignored")}),
    "SO.M5": _m(**{"aten.topk": _m_topk("ascending_output")}),
    "SO.M6": _m(**{"aten.topk": _m_topk("largest_ignored")}),
}
READINGS = {}
