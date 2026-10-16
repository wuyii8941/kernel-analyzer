"""Adapter for contracts/MATHFN.yaml. Reference: kernel_spec.aten; each mutant replaces one function by an independent
implementation of the misreading."""
from __future__ import annotations

import yaml
from mpmath import iv

from kernel_spec import aten as A
from kernel_spec import real as rl
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Undefined, to_iv
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "MATHFN.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _map(fn):
    return lambda a: A.unary(fn, A._floatify(a))


def _finite_only(fn):
    def g(x):
        if not x.finite:
            raise Undefined("infinite input rejected (misreading)")
        return fn(x)
    return g


MUTANTS = {
    "MF.M1": make_reference(COVERS, {"aten.log1p": _map(rl.log)}),
    "MF.M2": make_reference(COVERS, {"aten.expm1": _map(rl.exp)}),
    "MF.M3": make_reference(COVERS, {"aten.rsqrt": _map(rl.sqrt)}),
    "MF.M4": make_reference(COVERS, {"aten.sigmoid": _map(lambda x: (R(1) + rl.tanh(x)) / 2)}),
    "MF.M5": make_reference(COVERS, {"aten.erf": _map(lambda x: rl.erf(x / rl.sqrt(R(2))))}),
    "MF.M6": make_reference(COVERS, {"aten.log": _map(lambda x: rl.log(x) / R(iv.log(2)))}),
    "MF.M7": make_reference(COVERS, {"aten.sin": _map(lambda x: rl.sin(x * R(iv.pi * 1) / 180))}),
    "MF.M8": make_reference(COVERS, {"aten.exp": _map(_finite_only(rl.exp))}),
}
READINGS = {}
