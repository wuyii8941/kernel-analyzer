"""Adapter for the reviewed physical comparison and boolean scope in contracts/LOGIC.yaml."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "LOGIC.yaml").read_text())["covers"]
REAL_PROFILE = ("real",)
IEEE_PROFILE = ("ieee_values",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
REAL_OPS = {
    "aten.eq", "aten.ne", "aten.lt", "aten.le", "aten.gt", "aten.ge",
    "aten.logical_and", "aten.logical_or", "aten.logical_not",
    "aten.bitwise_and", "aten.bitwise_or", "aten.bitwise_not",
    "aten.any", "aten.all",
}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed LOGIC profiles")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _require_raw_booleans(value, name):
    raw = value.get("v") if isinstance(value, dict) else None
    for item in _raw_leaves(raw):
        if not isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name} values must be literal booleans")


def _physical_finite_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype == "b8":
        _require_raw_booleans(value, name)
    elif dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the reviewed LOGIC real profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype == "b8":
        return tensor
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {tensor.dtype!r} is outside the reviewed LOGIC real profile")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains a special value outside the real profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _physical_ieee_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the isnan profile")
    tensor = decode(value, allow_ignored_nan=True)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {tensor.dtype!r} is outside the isnan profile")
    for item in tensor.a.flat:
        if isinstance(item, (float, np.floating)) and math.isnan(float(item)):
            continue
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be physical floating values")
        if item.finite:
            stored = ieee.round_to(item, tensor.dtype)
            if not stored.exact or stored.v != item.v:
                raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _decode_argument(op, name, value, arguments):
    if isinstance(value, dict) and ("t" in value or "v" in value):
        return (_physical_ieee_tensor(value, name) if op == "aten.isnan"
                else _physical_finite_tensor(value, name))
    return decode(value)


def _tensor(value, name="input"):
    if not isinstance(value, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if value.dtype not in FLOAT_DTYPES | {"b8"}:
        raise Unsupported(f"{name} dtype {value.dtype!r} is outside the reviewed LOGIC real profile")
    return value


def _same_dtype(a, b):
    a, b = _tensor(a, "first input"), _tensor(b, "second input")
    if a.dtype != b.dtype:
        raise Unsupported("mixed tensor dtypes are outside the reviewed LOGIC real profile")
    return a, b


def _scalar(value, dtype):
    if isinstance(value, A.Ten):
        raise Invalid("comparison scalar must not be a tensor")
    if dtype == "b8":
        if not isinstance(value, (bool, np.bool_)):
            raise Invalid("a boolean tensor comparison requires a literal boolean scalar")
        return bool(value)
    if isinstance(value, (bool, np.bool_)):
        raise Invalid("a floating tensor comparison requires a real scalar")
    try:
        item = value if isinstance(value, R) else R(Fraction(value))
    except (TypeError, ValueError, ZeroDivisionError):
        raise Invalid("comparison scalar must be a finite exact real number") from None
    if not item.exact:
        raise Invalid("comparison scalar must be exact")
    if not item.finite:
        raise Unsupported("special comparison scalars are outside the real profile")
    stored = ieee.round_to(item, dtype)
    if not stored.exact or stored.v != item.v:
        raise Invalid(f"comparison scalar {item!r} is not exactly representable as {dtype}")
    return item.v


def _comparison(a, b, fn):
    a = _tensor(a, "first input")
    if isinstance(b, A.Ten):
        a, b = _same_dtype(a, b)
    else:
        b = _scalar(b, a.dtype)
    return fn(a, b)


def _eq(a, b):
    return _comparison(a, b, A.eq)


def _ne(a, b):
    return _comparison(a, b, A.ne)


def _lt(a, b):
    return _comparison(a, b, A.lt)


def _le(a, b):
    return _comparison(a, b, A.le)


def _gt(a, b):
    return _comparison(a, b, A.gt)


def _ge(a, b):
    return _comparison(a, b, A.ge)


def _logical_and(a, b):
    a, b = _same_dtype(a, b)
    return A.logical_and(a, b)


def _logical_or(a, b):
    a, b = _same_dtype(a, b)
    return A.logical_or(a, b)


def _logical_not(a):
    return A.logical_not(_tensor(a))


def _boolean_pair(a, b):
    a, b = _same_dtype(a, b)
    if a.dtype != "b8":
        raise Unsupported("integer bitwise tasks are outside the reviewed boolean profile")
    return a, b


def _bitwise_and(a, b):
    a, b = _boolean_pair(a, b)
    return A.bitwise_and(a, b)


def _bitwise_or(a, b):
    a, b = _boolean_pair(a, b)
    return A.bitwise_or(a, b)


def _bitwise_not(a):
    a = _tensor(a)
    if a.dtype != "b8":
        raise Unsupported("integer bitwise tasks are outside the reviewed boolean profile")
    return A.bitwise_not(a)


def _truth_reduction(a, dim=None, keepdim=False, *, all_values=False):
    a = _tensor(a)
    if dim is not None and (isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))):
        if isinstance(dim, (list, tuple)):
            raise Unsupported("dimension-list truth reductions are outside the reviewed profile")
        raise Invalid("truth-reduction dim must be one integer or None")
    if not isinstance(keepdim, (bool, np.bool_)):
        raise Invalid("truth-reduction keepdim must be boolean")
    return A.all_(a, None if dim is None else int(dim), bool(keepdim)) if all_values \
        else A.any_(a, None if dim is None else int(dim), bool(keepdim))


def _any(a, dim=None, keepdim=False):
    return _truth_reduction(a, dim, keepdim)


def _all(a, dim=None, keepdim=False):
    return _truth_reduction(a, dim, keepdim, all_values=True)


def _isnan(a, *, infinity_is_nan=False):
    if not isinstance(a, A.Ten) or a.dtype not in FLOAT_DTYPES:
        raise Unsupported("isnan requires a reviewed floating tensor")
    out = np.empty(a.shape, dtype=object)
    for index in np.ndindex(a.shape):
        value = a.a[index]
        nan = isinstance(value, (float, np.floating)) and math.isnan(float(value))
        out[index] = bool(nan or (infinity_is_nan and isinstance(value, R) and not value.finite))
    return A.Ten(out, "b8")


OPS = {
    "aten.eq": _eq, "aten.ne": _ne, "aten.lt": _lt, "aten.le": _le,
    "aten.gt": _gt, "aten.ge": _ge,
    "aten.logical_and": _logical_and, "aten.logical_or": _logical_or,
    "aten.logical_not": _logical_not,
    "aten.bitwise_and": _bitwise_and, "aten.bitwise_or": _bitwise_or,
    "aten.bitwise_not": _bitwise_not,
    "aten.any": _any, "aten.all": _all, "aten.isnan": _isnan,
}


def _profiled_reference(ops=None):
    table = dict(OPS if ops is None else ops)
    real = make_reference(COVERS, extra=table, profiles=REAL_PROFILE, argument_decoder=_decode_argument)
    ieee = make_reference(COVERS, extra=table, profiles=IEEE_PROFILE, argument_decoder=_decode_argument)

    def reference(profile, inp):
        op = inp.get("op") if isinstance(inp, dict) else None
        if profile == "real":
            if op not in REAL_OPS:
                raise Unsupported("the real profile does not cover isnan")
            return real(profile, inp)
        if profile == "ieee_values":
            if op != "aten.isnan":
                raise Unsupported("the ieee_values profile covers isnan only")
            return ieee(profile, inp)
        raise Unsupported(f"profile {profile!r} is not covered by LOGIC")
    return reference


reference = _profiled_reference()
_real_checker = make_output_checker(COVERS, extra=OPS, profiles=REAL_PROFILE,
                                    argument_decoder=_decode_argument)
_ieee_checker = make_output_checker(COVERS, extra=OPS, profiles=IEEE_PROFILE,
                                    argument_decoder=_decode_argument)


def check_output(profile, inp, observed):
    op = inp.get("op") if isinstance(inp, dict) else None
    if not isinstance(observed, dict) or set(observed) != {"out"}:
        return False
    record = observed["out"]
    if not isinstance(record, dict) or "v" not in record:
        return False
    try:
        _require_raw_booleans(record, "observed output")
    except Invalid:
        return False
    if profile == "real" and op in REAL_OPS:
        return _real_checker(profile, inp, observed)
    if profile == "ieee_values" and op == "aten.isnan":
        return _ieee_checker(profile, inp, observed)
    raise Unsupported("profile and operation do not select one reviewed LOGIC task")


def _lt_as_le(a, b):
    return _comparison(a, b, A.le)


def _ne_as_eq(a, b):
    return _comparison(a, b, A.eq)


def _positive_truth(value):
    if isinstance(value, bool):
        return value
    return bool(value > 0)


def _logical_and_positive(a, b):
    a, b = _same_dtype(a, b)
    return A.binary(lambda x, y: _positive_truth(x) and _positive_truth(y), a, b, out_dtype="b8")


def _bitwise_not_identity(a):
    a = _tensor(a)
    if a.dtype != "b8":
        raise Unsupported("integer bitwise tasks are outside the reviewed boolean profile")
    return A.clone(a)


def _any_empty_true(a, dim=None, keepdim=False):
    a = _tensor(a)
    if a.a.size == 0 and dim is None:
        return A.Ten(np.array(True, dtype=object), "b8")
    return _any(a, dim, keepdim)


def _all_empty_false(a, dim=None, keepdim=False):
    a = _tensor(a)
    if a.a.size == 0 and dim is None:
        return A.Ten(np.array(False, dtype=object), "b8")
    return _all(a, dim, keepdim)


def _any_all_dimensions(a, dim=None, keepdim=False):
    _truth_reduction(a, dim, keepdim)
    return A.any_(a, None, False)


def _isnan_with_infinity(a, *, infinity_is_nan=False):
    return _isnan(a, infinity_is_nan=True)


def _ge_as_gt(a, b):
    return _comparison(a, b, A.gt)


def _mut(**replacements):
    return _profiled_reference({**OPS, **replacements})


MUTANTS = {
    "LG.M1": _mut(**{"aten.lt": _lt_as_le}),
    "LG.M2": _mut(**{"aten.ne": _ne_as_eq}),
    "LG.M3": _mut(**{"aten.logical_and": _logical_and_positive}),
    "LG.M4": _mut(**{"aten.bitwise_not": _bitwise_not_identity}),
    "LG.M5": _mut(**{"aten.any": _any_empty_true}),
    "LG.M6": _mut(**{"aten.all": _all_empty_false}),
    "LG.M7": _mut(**{"aten.any": _any_all_dimensions}),
    "LG.M8": _mut(**{"aten.isnan": _isnan_with_infinity}),
    "LG.M9": _mut(**{"aten.ge": _ge_as_gt}),
}
READINGS = {}
