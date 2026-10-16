"""Adapter for contracts/MATMUL.yaml. Reference: kernel_spec.aten; mutants are independent implementations of the
misreadings."""
from __future__ import annotations

import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "MATMUL.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _linear_no_t(x, w, b=None):
    y = A.matmul(x, w)
    return y if b is None else A.add(y, b)


def _addmm_beta0_adds(c, a, b, beta=1, alpha=1):
    return A.add(A.mul(c, 1 if beta == 0 else beta), A.mul(A.mm(a, b), alpha))


def _vjp_matmul_no_reduce(g, a, b):
    return [A.matmul(g, A.transpose(b, -1, -2)), A.matmul(A.transpose(a, -1, -2), g)]


def _vjp_mm_wrong(g, a, b):
    return [A.mm(g, A.t(b)), A.mm(g, a)]


def _vjp_linear_bias_rows(g, x, w, b=None):
    dx, dw, db = A.vjp_linear(g, x, w, b)
    return [dx, dw, A.as_ten(g)]


def _addmm_no_alpha(c, a, b, beta=1, alpha=1):
    return A.addmm(c, a, b, beta, 1)


def _m(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "MM.M1": _m(**{"aten.linear": _linear_no_t}),
    "MM.M2": _m(**{"aten.addmm": _addmm_beta0_adds}),
    "MM.M3": _m(**{"vjp.aten.matmul": _vjp_matmul_no_reduce}),
    "MM.M4": _m(**{"vjp.aten.mm": _vjp_mm_wrong}),
    "MM.M5": _m(**{"vjp.aten.linear": _vjp_linear_bias_rows}),
    "MM.M6": _m(**{"aten.addmm": _addmm_no_alpha}),
}
READINGS = {}
