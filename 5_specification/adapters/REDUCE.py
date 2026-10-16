"""Adapter for contracts/REDUCE.yaml. Reference: kernel_spec.aten (reductions and vjp.*); each mutant replaces one
operation by an independent implementation of the misreading."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import real as rl
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Undefined
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "REDUCE.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _flat(t):
    return list(A.as_ten(t).a.reshape(-1))


def _var_by_n(a, dim=None, correction=1, keepdim=False):
    xs = _flat(a)
    n = len(xs)
    m = sum(xs[1:], xs[0]) / n
    return A.Ten(np.array(sum(((x - m) * (x - m) for x in xs[1:]), (xs[0] - m) * (xs[0] - m)) / n, dtype=object), "f32")


def _argmax_last(a, dim=None, keepdim=False):
    t = A.as_ten(a)
    rows = t.a.reshape(-1, t.a.shape[-1])
    out = []
    for r in rows:
        best, k = r[0], 0
        for i, x in enumerate(r):
            if x >= best:
                best, k = x, i
        out.append(k)
    return A.tensor(out, "i64")


def _amax_grad_first(g, x, dim=None, keepdim=False):
    x = A.as_ten(x)
    out = np.full(x.shape, R(0), dtype=object)
    gv = _flat(g)
    for r, row in enumerate(x.a.reshape(-1, x.a.shape[-1])):
        k = max(range(len(row)), key=lambda i: (row[i].v, -i))
        out.reshape(-1, x.a.shape[-1])[r, k] = gv[r]
    return A.Ten(out, x.dtype)


def _max_grad_even(g, x, dim, keepdim=False):
    return A.vjp_amax(g, x, [dim], keepdim)


def _prod_grad_div(g, x, dim=None, keepdim=False):
    x = A.as_ten(x)
    out = np.empty(x.shape, dtype=object)
    gv = _flat(g)
    for r, row in enumerate(x.a.reshape(-1, x.a.shape[-1])):
        p = R(1)
        for v in row:
            p = p * v
        for i, v in enumerate(row):
            out.reshape(-1, x.a.shape[-1])[r, i] = gv[r] * p / v          # raises Undefined at zeros
    return A.Ten(out, x.dtype)


def _mean_empty_zero(a, dim=None, keepdim=False, dtype=None):
    t = A.as_ten(a)
    if t.a.size == 0:
        shape = [s for i, s in enumerate(t.shape) if i not in A._axes(t, dim)]
        return A.full(shape, 0, "f32")
    return A.mean(a, dim, keepdim)


def _cumsum_grad_forward(g, x, dim):
    return A.cumsum(A._floatify(g), dim)


def _lse_naive(a, dim, keepdim=False):
    def row(xs):
        m = xs[0]
        for v in xs[1:]:
            m = rl.max_(m, v)
        return m + rl.log(sum((rl.exp(v - m) for v in xs[1:]), rl.exp(xs[0] - m)))
    return A._reduce(A._floatify(a), dim, keepdim, row)


def _m(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "RD.M1": _m(**{"aten.var": _var_by_n}),
    "RD.M2": _m(**{"aten.argmax": _argmax_last}),
    "RD.M3": _m(**{"vjp.aten.amax": _amax_grad_first}),
    "RD.M4": _m(**{"vjp.aten.max": _max_grad_even}),
    "RD.M5": _m(**{"vjp.aten.prod": _prod_grad_div}),
    "RD.M6": _m(**{"aten.mean": _mean_empty_zero}),
    "RD.M7": _m(**{"vjp.aten.cumsum": _cumsum_grad_forward}),
    "RD.M8": _m(**{"aten.logsumexp": _lse_naive}),
}
READINGS = {}
