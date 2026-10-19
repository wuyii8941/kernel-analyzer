"""Adapter for the reviewed finite floating scope in contracts/MATHFN.yaml."""
from __future__ import annotations

import yaml

from kernel_spec import aten as A
from kernel_spec import real as rl
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "MATHFN.yaml").read_text())["covers"]
PROFILE = ("real",)
DTYPES = {"f16", "bf16", "f32", "f64"}
LIMIT = R(100)


def _finite_tensor(value, name):
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
        if item < -LIMIT or item > LIMIT:
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
    return _finite_tensor(decode(value), name)


def _domain_map(predicate, message, fn):
    def op(a):
        def scalar(x):
            if not predicate(x):
                raise Undefined(message)
            return fn(x)
        return A.unary(scalar, A._floatify(a))
    return op


STRICT_OPS = {
    "aten.log": _domain_map(lambda x: x > 0, "log requires x > 0", rl.log),
    "aten.log1p": _domain_map(lambda x: x > -1, "log1p requires x > -1", rl.log1p),
    "aten.rsqrt": _domain_map(lambda x: x > 0, "rsqrt requires x > 0", rl.rsqrt),
}

reference = make_reference(COVERS, extra=STRICT_OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=STRICT_OPS, profiles=PROFILE,
                                   argument_decoder=_decode_argument)


def _map(fn):
    return lambda a: A.unary(fn, A._floatify(a))


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
}
READINGS = {}
