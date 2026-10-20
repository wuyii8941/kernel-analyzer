"""Adapter for the finite floating scope in contracts/MATHFN.yaml.

exp, expm1, log, log1p, sin, cos, tanh and sigmoid are enclosed with python-flint Arb
(kernel_spec.arith_mathfn_index_arb, decision MF.EVALUATION); sqrt, rsqrt and erf keep the kernel_spec.real evaluator.
sin, cos and tanh accept every finite value of the reviewed dtypes (MF.MAGNITUDE); the other functions keep |x| <= 100.
"""
from __future__ import annotations

import math
from fractions import Fraction

from pathlib import Path

import yaml

from kernel_spec import aten as A
from kernel_spec import arith_mathfn_index_arb as arb
from kernel_spec import real as rl
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R

# Read this adapter's own contract: next to the draft file (contracts/draft/MATHFN_path_a_A2_20261020.yaml) or, when
# staged as <stage>/adapters/MATHFN.py, <stage>/contracts/MATHFN.yaml. Never the top-level validated contract.
_HERE = Path(__file__).resolve()
_OWN_CONTRACT = next(p for p in (_HERE.parent / "MATHFN_path_a_A2_20261020.yaml",
                                 _HERE.parents[1] / "contracts" / "MATHFN.yaml") if p.is_file())
COVERS = yaml.safe_load(_OWN_CONTRACT.read_text())["covers"]
PROFILE = ("real",)
DTYPES = {"f16", "bf16", "f32", "f64"}
LIMIT = R(100)
UNBOUNDED = {"aten.sin", "aten.cos", "aten.tanh"}     # MF.MAGNITUDE: bounded functions, no magnitude limit


def _finite_tensor(value, name, *, bounded=True):
    if not isinstance(value, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if value.dtype not in DTYPES:
        raise Unsupported(f"{name} dtype {value.dtype!r} is outside the finite MATHFN profile")
    for item in value.a.flat:
        if not isinstance(item, R):
            raise Invalid(f"{name} is not a real tensor")
        if not item.finite:
            raise Unsupported(f"{name} contains an infinity outside the finite MATHFN profile")
        if not item.exact:
            raise Unsupported(f"{name} contains a non-point value outside the MATHFN input representation")
        if bounded and (item < -LIMIT or item > LIMIT):
            raise Unsupported(f"{name} contains a value with magnitude above 100")
    return value


def _decode_argument(op, name, value, raw):
    if name != "a":
        return decode(value)
    if isinstance(value, dict) and "t" in value:
        try:
            dtype = A.dt(value["t"])
        except Invalid:
            if "complex" in str(value["t"]).lower():
                raise Unsupported("complex tensors are outside the real MATHFN profile")
            raise
        if dtype not in DTYPES:
            raise Unsupported(f"{name} dtype {dtype!r} is outside the finite MATHFN profile")
    return _finite_tensor(decode(value), name, bounded=op not in UNBOUNDED)


def _domain_map(predicate, message, fn):
    def op(a):
        def scalar(x):
            if not predicate(x):
                raise Undefined(message)
            return fn(x)
        return A.unary(scalar, A._floatify(a))
    return op


def _arb(name):
    return lambda x: arb.evaluate(name, x)


def _total(fn):
    return lambda a: A.unary(fn, A._floatify(a))


STRICT_OPS = {
    "aten.exp": _total(_arb("exp")),
    "aten.expm1": _total(_arb("expm1")),
    "aten.sin": _total(_arb("sin")),
    "aten.cos": _total(_arb("cos")),
    "aten.tanh": _total(_arb("tanh")),
    "aten.sigmoid": _total(_arb("sigmoid")),
    "aten.log": _domain_map(lambda x: x > 0, "log requires x > 0", _arb("log")),
    "aten.log1p": _domain_map(lambda x: x > -1, "log1p requires x > -1", _arb("log1p")),
    "aten.rsqrt": _domain_map(lambda x: x > 0, "rsqrt requires x > 0", rl.rsqrt),
}

reference = make_reference(COVERS, extra=STRICT_OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=STRICT_OPS, profiles=PROFILE,
                                   argument_decoder=_decode_argument)


def _map(fn):
    return lambda a: A.unary(fn, A._floatify(a))


def _reduced_with_binary64_two_pi(x, fn):
    """Misreading: large arguments are reduced modulo the binary64 value of 2*pi instead of the real 2*pi."""
    two_pi = Fraction(2 * math.pi)
    k = math.floor(x.v / two_pi)
    return fn(R(x.v - k * two_pi))


def _tanh_saturating(x):
    """Misreading: tanh is exactly +-1 once |x| >= 20."""
    if abs(x.v) >= 20:
        return R(1 if x.v > 0 else -1)
    return arb.evaluate("tanh", x)


def _mut(**ops):
    return make_reference(COVERS, extra={**STRICT_OPS, **ops}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


MUTANTS = {
    "MF.M1": _mut(**{"aten.log1p": _map(rl.log)}),
    "MF.M2": _mut(**{"aten.expm1": _map(rl.exp)}),
    "MF.M3": _mut(**{"aten.rsqrt": _map(rl.sqrt)}),
    "MF.M4": _mut(**{"aten.sigmoid": _map(lambda x: (R(1) + rl.tanh(x)) / 2)}),
    "MF.M5": _mut(**{"aten.erf": _map(lambda x: rl.erf(x / rl.sqrt(R(2))))}),
    "MF.M6": _mut(**{"aten.log": _map(lambda x: rl.log(x) / rl.log(R(2)))}),
    "MF.M7": _mut(**{"aten.sin": _map(lambda x: rl.sin(x * R(A._pi()) / 180))}),
    "MF.M8": _mut(**{"aten.exp": _map(lambda x: rl.exp(-x))}),
    "MF.M9": _mut(**{"aten.sin": _map(lambda x: _reduced_with_binary64_two_pi(x, rl.sin)),
                     "aten.cos": _map(lambda x: _reduced_with_binary64_two_pi(x, rl.cos))}),
    "MF.M10": _mut(**{"aten.tanh": _map(_tanh_saturating)}),
}
READINGS = {}
