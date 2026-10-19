"""sort / argsort / topk of extended reals (contracts/SORT.yaml), shared by the SORT adapter and the GRAPH composition.

Exact elements are ordered by their value; ties are equal exact values (index allowed sets unless stable). An element
may also be an enclosure (a value computed inside a graph, e.g. a softmax probability): two elements are ordered when
their order is decided by their enclosures, and the result is NotEstablished otherwise. A comparison sort compares every
pair that ends up adjacent, so an undecided adjacent pair always surfaces; an order decided for adjacent pairs holds for
all pairs by transitivity of the order of the reals. Only exact equal values can be decided equal.
"""
from __future__ import annotations

import builtins
import functools
import itertools

import numpy as np

from . import aten as A
from .num import Invalid, NotEstablished, Undefined, Unsupported
from .real import R


def _key(x):
    """Total order key of an exact extended real."""
    v = x.v
    if v == "-inf":
        return (-1, 0)
    if v == "+inf":
        return (1, 0)
    return (0, v)


def _order(xs, descending, k=None):
    """Positions of xs in sorted order (ties by position); NotEstablished when an enclosure leaves an order undecided."""
    idx = range(len(xs))
    if all(x.exact for x in xs):
        if descending:
            order = builtins.sorted(idx, key=lambda i: (tuple(-c for c in _key(xs[i])), i))
        else:
            order = builtins.sorted(idx, key=lambda i: (_key(xs[i]), i))
    else:
        sgn = -1 if descending else 1

        def cmp(i, j):
            c = xs[i]._cmp(xs[j]) * sgn          # raises NotEstablished when the enclosures do not decide the order
            return c if c else (i > j) - (i < j)
        order = builtins.sorted(idx, key=functools.cmp_to_key(cmp))
    return order if k is None else order[:k]


def _dimension(a, dim):
    rank = max(1, a.a.ndim)
    if isinstance(dim, bool) or not isinstance(dim, int) or not -rank <= dim < rank:
        raise Invalid(f"sorting: dim {dim!r} is not an integer in [{-rank}, {rank - 1}]")
    return dim % rank


def _apply(a, dim, fn, output_size=None):
    """fn(list of values) -> (values, indices) along dim; rebuilt into two tensors."""
    a = A.as_ten(a)
    d = _dimension(a, dim)
    if a.a.ndim == 0:
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
        n = moved.shape[-1] if output_size is None else output_size
        outs_v = np.empty(lead + (n,), dtype=object)
        outs_i = np.empty(lead + (n,), dtype=object)
    return A.Ten(np.moveaxis(outs_v, -1, d), a.dtype), A.Ten(np.moveaxis(outs_i, -1, d), "i64")


def _index_sets(xs, order, exact):
    out = []
    for i in order:
        if exact or not xs[i].exact:
            out.append(i)
            continue
        same = [m for m in range(len(xs)) if xs[m].exact and _key(xs[m]) == _key(xs[i])]
        out.append(i if len(same) == 1 else A.AllowedSet(sorted(same)))
    return out


def sort(a, dim=-1, descending=False, stable=False):
    def line(xs):
        order = _order(xs, descending)
        return [xs[i] for i in order], _index_sets(xs, order, stable)
    return list(_apply(a, dim, line))


def argsort(a, dim=-1, descending=False, stable=False):
    return sort(a, dim, descending, stable)[1]


def topk(a, k, dim=-1, largest=True, sorted=True):
    if not sorted:
        raise Unsupported("topk with sorted=False: the order of the k elements is unspecified (decision SO.UNSORTED)")
    a = A.as_ten(a)
    d = _dimension(a, dim)
    n = a.a.shape[d] if a.a.ndim else 1
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > n:
        raise Invalid(f"topk: k = {k} out of range for a dimension of size {n}")

    def line(xs):
        order = _order(xs, largest, k if a.a.ndim else 1)
        return [xs[i] for i in order], _index_sets(xs, order, False)
    return list(_apply(a, dim, line, output_size=k))


OPS = {"aten.sort": sort, "aten.argsort": argsort, "aten.topk": topk}


def concrete_outputs(result, op, dim, budget):
    """Complete alternatives for a sorting node, retaining row-wise uniqueness.

    The bounded enumeration is for small ambiguous graph nodes. Exceeding the
    budget refuses the query instead of returning an incomplete allowed set.
    """
    indices = result if op == "aten.argsort" else result[1]
    positions = list(np.ndindex(indices.a.shape))
    choices = [indices.a[p].members if isinstance(indices.a[p], A.AllowedSet) else [indices.a[p]]
               for p in positions]
    if all(len(c) == 1 for c in choices):
        return [result]
    alternatives = []
    d = _dimension(indices, dim)
    for attempts, assignment in enumerate(itertools.product(*choices), 1):
        if attempts > budget:
            raise NotEstablished("sorting joint-choice enumeration exceeded its budget")
        array = np.empty(indices.a.shape, dtype=object)
        for p, value in zip(positions, assignment):
            array[p] = value
        rows = np.moveaxis(array, d, -1) if array.ndim else array.reshape(1)
        if any(len(set(rows[p])) != len(rows[p]) for p in np.ndindex(rows.shape[:-1])):
            continue
        concrete = A.Ten(array, indices.dtype)
        alternatives.append(concrete if op == "aten.argsort" else [result[0], concrete])
    return alternatives


def check_output(a, indices, values=None, *, op="sort", dim=-1, k=None, descending=None, stable=False):
    """Membership of one complete output in the declared sorting task.

    Checks joint index uniqueness, value/index correspondence, order and selection.
    This is a task-level check; the caller must still bind the complete kernel
    outputs to this task. Unresolved comparisons raise NotEstablished.
    Candidates are raw arrays/lists, so boolean or fractional indices are not
    silently converted to integer indices by a tensor constructor.
    """
    if op not in ("sort", "argsort", "topk"):
        raise Unsupported(f"unknown sorting task {op!r}")
    if descending is None:
        descending = op == "topk"
    if not isinstance(descending, bool) or not isinstance(stable, bool):
        raise Invalid("descending and stable must be booleans")
    a = A.as_ten(a)
    d = _dimension(a, dim)
    n = a.a.shape[d] if a.a.ndim else 1
    size = n
    if op == "topk":
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= n:
            raise Invalid("topk requires an integer k in [0, n]")
        size = k if a.a.ndim else 1
    expected_shape = list(a.a.shape)
    if expected_shape:
        expected_shape[d] = size
    try:
        ix = np.asarray(indices, dtype=object)
        val = np.asarray(values, dtype=object) if values is not None else None
    except (ValueError, TypeError):
        return False
    if ix.shape != tuple(expected_shape):
        return False
    if op != "argsort" and (val is None or val.shape != ix.shape):
        return False
    if op == "argsort" and values is not None:
        return False
    xs = np.moveaxis(a.a, d, -1) if a.a.ndim else a.a.reshape(1)
    ix = np.moveaxis(ix, d, -1) if a.a.ndim else ix.reshape(1)
    if val is not None:
        val = np.moveaxis(val, d, -1) if a.a.ndim else val.reshape(1)
    for row in np.ndindex(xs.shape[:-1]):
        selected = list(ix[row])
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               or not 0 <= i < n for i in selected):
            return False
        if len(set(selected)) != len(selected):
            return False
        source = [x if isinstance(x, R) else R(x) for x in xs[row]]
        for j, index in enumerate(selected):
            if val is not None:
                try:
                    observed = R(val[row + (j,)])
                except (Invalid, Undefined, TypeError, ValueError):
                    return False
                if source[index]._cmp(observed) != 0:
                    return False
            if j:
                previous = selected[j - 1]
                c = source[previous]._cmp(source[index])
                if (c < 0 if descending else c > 0):
                    return False
                if stable and op != "topk" and c == 0 and previous > index:
                    return False
        if op == "topk" and selected:
            last = source[selected[-1]]
            for index in set(range(n)) - set(selected):
                c = last._cmp(source[index])
                if (c < 0 if descending else c > 0):
                    return False
    return True
