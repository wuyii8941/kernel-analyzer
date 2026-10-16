"""Adapter for contracts/RANDOM.yaml. Reference: kernel_spec.aten (conditioned random operations); mutants are
independent implementations of the misreadings."""
from __future__ import annotations

from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Undefined
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "RANDOM.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _vals(t):
    return list(A.as_ten(t).a.reshape(-1))


def _dropout_mul(x, p, train=True, keep=None):
    p = Fraction(p) if not isinstance(p, R) else p.v
    if not train or p == 0:
        return [A._floatify(x), A.full(A.as_ten(x).shape, True, "b8")]
    xs, ks = _vals(x), _vals(keep)
    return [A.tensor([v.v * (1 - p) if k else 0 for v, k in zip(xs, ks)], "f32"), A.as_ten(keep)]


def _dropout_eval_scaled(x, p, train=True, keep=None):
    if not train:
        p = Fraction(p) if not isinstance(p, R) else p.v
        return [A.tensor([v.v / (1 - p) for v in _vals(x)], "f32"), A.full(A.as_ten(x).shape, True, "b8")]
    return A.native_dropout(x, p, train, keep)


def _bwd_no_scale(grad, mask, scale):
    return A.tensor([g.v if m else 0 for g, m in zip(_vals(grad), _vals(mask))], "f32")


def _keep_ge(u, p):
    return A.tensor([v.v >= Fraction(p) for v in _vals(u)], "b8")


def _p1_divides(x, p, train=True, keep=None):
    if train and Fraction(p) == 1:
        raise Undefined("1 / (1 - p) with p = 1")
    return A.native_dropout(x, p, train, keep)


def _m(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "RN.M1": _m(**{"aten.native_dropout": _dropout_mul}),
    "RN.M2": _m(**{"aten.native_dropout": _dropout_eval_scaled}),
    "RN.M3": _m(**{"aten.native_dropout_backward": _bwd_no_scale}),
    "RN.M4": _m(**{"keep_from_uniform": _keep_ge}),
    "RN.M5": _m(**{"aten.native_dropout": _p1_divides}),
}
READINGS = {}
