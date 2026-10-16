"""Shared adapter for the contracts whose clauses are ATen operations (ARITH, MATHFN, LOGIC, CAST, LAYOUT, INDEX,
REDUCE, SOFTMAX, MATMUL, CONV, ACT, RANDOM).

Example inputs: {"op": "aten.add", "args": [...], "kwargs": {...}} where a tensor argument is written
{"t": "f32", "v": nested list} (values: integers, "p/q" strings, "inf" / "-inf", booleans; an optional "shape" reshapes
the values, needed for empty tensors such as shape [0, 3]), a list of tensors is a
list of such objects, and other arguments are plain YAML values ("p/q" strings become exact rationals).
The output field `out` is the operation's result as nested lists; operations with several results give a list.
"""
from __future__ import annotations

from fractions import Fraction

from . import aten as A
from .num import Invalid, Result, Unsupported, rational
from .real import NEG, POS, R


def decode(x):
    if isinstance(x, dict) and "t" in x and "v" in x:
        t = A.tensor(_vals(x["v"], x["t"]), x["t"])
        return A.view(t, x["shape"]) if "shape" in x else t
    if isinstance(x, dict) and "dtype" in x and len(x) == 1:
        return A.dt(x["dtype"])
    if isinstance(x, list):
        return [decode(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s in ("inf", "+inf"):
            return R(POS)
        if s == "-inf":
            return R(NEG)
        if s.startswith("torch.") or s in A.FLOAT | A.INT | A.BOOL:
            return s
        try:
            return rational(s)
        except Invalid:
            return s
    return x


def _vals(v, d):
    if isinstance(v, list):
        return [_vals(x, d) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s in ("inf", "+inf"):
            return float("inf")
        if s == "-inf":
            return float("-inf")
        return rational(s)
    return v


def encode(r):
    if isinstance(r, A.Ten):
        return _enc(r.a)
    if isinstance(r, (tuple, list)):
        return [encode(x) for x in r]
    return _enc(r)


def _enc(a):
    import numpy as np
    if isinstance(a, np.ndarray):
        return _enc(a[()]) if a.ndim == 0 else [_enc(x) for x in a]
    if isinstance(a, A.AllowedSet):
        return frozenset(_enc(m) for m in a.members)
    if isinstance(a, R):
        return a.v
    if isinstance(a, Fraction):
        return a
    return a


def make_reference(covers, extra=None):
    """reference(profile, inp) for a contract that covers the given ATen operation names."""
    table = dict(A.OPS)
    table.update(extra or {})

    def reference(profile, inp):
        op = inp.get("op")
        if op not in covers:
            raise Unsupported(f"{op} is not covered by this contract")
        fn = table.get(op)
        if fn is None:
            raise Unsupported(f"{op} has no executable semantics")
        args = [decode(a) for a in inp.get("args", [])]
        kwargs = {k: decode(v) for k, v in (inp.get("kwargs") or {}).items()}
        return Result("value", {"out": encode(fn(*args, **kwargs))})   # a TypeError stays an implementation error
    return reference
