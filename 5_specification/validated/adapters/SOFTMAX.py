"""Adapter for contracts/SOFTMAX.yaml. Reference: kernel_spec.aten; mutants are independent implementations of the
misreadings."""
from __future__ import annotations

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import real as rl
from kernel_spec.aten_adapter import make_reference, make_output_checker
from kernel_spec.real import NEG, R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "SOFTMAX.yaml").read_text())["covers"]
reference = make_reference(COVERS, profiles=("real",))
check_output = make_output_checker(COVERS, profiles=("real",))


def _rows_last(a, fn):
    t = A._floatify(a)
    out = np.empty(t.shape, dtype=object)
    flat_in, flat_out = t.a.reshape(-1, t.shape[-1]), out.reshape(-1, t.shape[-1])
    for r in range(flat_in.shape[0]):
        flat_out[r] = fn(list(flat_in[r]))
    return A.Ten(out, t.dtype)


def _sm(xs):
    es = [rl.exp(x) for x in xs]
    s = es[0]
    for e in es[1:]:
        s = s + e
    return [e / s for e in es]


def _softmax_last(a, dim, half_to_float=False):
    return _rows_last(a, _sm)


def _softmax_uniform_masked(a, dim, half_to_float=False):
    def row(xs):
        if all(x.v == NEG for x in xs):
            return [R(1) / len(xs)] * len(xs)
        return A._softmax_row(xs)
    return A._rowwise(a, dim, row)


def _softmax_inf_as_zero(a, dim, half_to_float=False):
    return A._rowwise(a, dim, lambda xs: _sm([x if x.finite else R(0) for x in xs]))


def _sm_bwd_no_sum(grad, out, dim, input_dtype=None):
    return A.mul(A._floatify(grad), A._floatify(out))


def _lsm_bwd_y(grad, out, dim, input_dtype=None):
    g, y = A._floatify(grad), A._floatify(out)
    return A.sub(g, A.mul(y, A.sum_(g, dim, keepdim=True)))


def _pso_no_max(a, dim):
    return A.amax(a, dim, keepdim=True), A.sum_(A.exp(a), dim, keepdim=True)


def _safe_undefined(a, dim):
    return A.softmax(a, dim)


def _public_vjp_logits_as_probabilities(grad, a, dim):
    return A.softmax_backward_data(grad, a, dim, A.as_ten(a).dtype)


def _public_vjp_no_sum(grad, a, dim):
    return A.mul(grad, A.softmax(a, dim))


def _public_vjp_last_axis(grad, a, dim):
    return A.vjp_softmax(grad, a, -1)


def _m(**ops):
    return make_reference(COVERS, extra=ops, profiles=("real",))


MUTANTS = {
    "SM.M8": _m(**{"vjp.aten.softmax": _public_vjp_logits_as_probabilities}),
    "SM.M9": _m(**{"vjp.aten.softmax": _public_vjp_no_sum}),
    "SM.M10": _m(**{"vjp.aten.softmax": _public_vjp_last_axis}),
    "SM.M1": _m(**{"aten._softmax": _softmax_last}),
    "SM.M2": _m(**{"aten._softmax": _softmax_uniform_masked}),
    "SM.M3": _m(**{"aten._softmax": _softmax_inf_as_zero}),
    "SM.M4": _m(**{"aten._softmax_backward_data": _sm_bwd_no_sum}),
    "SM.M5": _m(**{"aten._log_softmax_backward_data": _lsm_bwd_y}),
    "SM.M6": _m(**{"prims.prepare_softmax_online": _pso_no_max}),
    "SM.M7": _m(**{"aten._safe_softmax": _safe_undefined}),
}
READINGS = {}
