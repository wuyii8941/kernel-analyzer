"""Adapter for the reviewed finite same-dtype floating scope in contracts/ARITH.yaml."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, encode, make_output_checker, make_reference
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, require_fields
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "ARITH.yaml").read_text())["covers"]
REAL_PROFILE = "real"
MASKED_FILL_PROFILE = "torch210_masked_fill_scalar_autograd"
WHERE_PROFILE = "torch210_where_tensor_autograd"
CLAMP_PROFILE = "torch210_clamp_scalar_autograd"
EXTREMA_PROFILE = "torch210_extrema_tensor_autograd"
PROFILE = (REAL_PROFILE,)
DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the finite ARITH profile")
        raise


def _require_raw_booleans(value, name):
    raw = value.get("v") if isinstance(value, dict) else None
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif not isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name} values must be booleans")


def _point_tensor(value, name, *, boolean=False):
    dtype = _raw_dtype(value)
    if boolean:
        if dtype is not None and dtype != "b8":
            raise Invalid(f"{name} must be a boolean tensor")
        _require_raw_booleans(value, name)
    elif dtype is not None and dtype not in DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite ARITH profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if boolean:
        if tensor.dtype != "b8":
            raise Invalid(f"{name} must be a boolean tensor")
        return tensor
    if tensor.dtype not in DTYPES:
        raise Unsupported(f"{name} dtype {tensor.dtype!r} is outside the finite ARITH profile")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains infinity outside the finite ARITH profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name in {"condition", "mask"}:
        return _point_tensor(value, name, boolean=True)
    if isinstance(value, dict) and ("t" in value or "v" in value):
        return _point_tensor(value, name)
    return decode(value)


def _data_dtype(*values):
    tensors = [value for value in values if isinstance(value, A.Ten)]
    if not tensors:
        raise Invalid("the selected functional operation requires at least one data tensor")
    if any(t.dtype not in DTYPES for t in tensors):
        raise Unsupported("data tensors must be finite f16, bf16, f32 or f64 tensors")
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) != 1:
        raise Unsupported("mixed data tensor dtypes are outside the finite ARITH profile")
    return tensors[0].dtype


def _scalar(value, dtype, name, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, A.Ten):
        raise Unsupported(f"tensor-valued {name} is outside the selected scalar overload")
    if isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be a real scalar")
    try:
        item = value if isinstance(value, R) else R(Fraction(value))
    except (TypeError, ValueError, ZeroDivisionError):
        raise Invalid(f"{name} must be a finite real scalar") from None
    if not item.exact:
        raise Invalid(f"{name} must be exact")
    if not item.finite:
        raise Unsupported(f"infinite {name} is outside the finite ARITH profile")
    stored = ieee.round_to(item, dtype)
    if not stored.exact or stored.v != item.v:
        raise Invalid(f"{name}={item!r} is not exactly representable as {dtype}")
    return item.v


def _binary_operands(a, b, *, require_left_tensor=True):
    if require_left_tensor and not isinstance(a, A.Ten):
        raise Invalid("the selected tensor/scalar overload requires its first input to be a tensor")
    dtype = _data_dtype(a, b)
    left = a if isinstance(a, A.Ten) else _scalar(a, dtype, "left operand")
    right = b if isinstance(b, A.Ten) else _scalar(b, dtype, "right operand")
    return left, right, dtype


def _add(a, b, *, alpha=1):
    a, b, dtype = _binary_operands(a, b)
    return A.add(a, b, _scalar(alpha, dtype, "alpha"))


def _sub(a, b, *, alpha=1):
    a, b, dtype = _binary_operands(a, b)
    return A.sub(a, b, _scalar(alpha, dtype, "alpha"))


def _rsub(a, b, *, alpha=1):
    a, b, dtype = _binary_operands(a, b)
    return A.rsub(a, b, _scalar(alpha, dtype, "alpha"))


def _mul(a, b):
    a, b, _ = _binary_operands(a, b)
    return A.mul(a, b)


def _div(a, b, *, rounding_mode=None):
    a, b, _ = _binary_operands(a, b)
    if rounding_mode not in (None, "trunc", "floor"):
        raise Invalid("rounding_mode must be None, trunc or floor")
    return A.div(a, b, rounding_mode)


def _unary(a, fn):
    _data_dtype(a)
    return fn(a)


def _neg(a):
    return _unary(a, A.neg)


def _reciprocal(a):
    return _unary(a, A.reciprocal)


def _abs(a):
    return _unary(a, A.abs_)


def _sign(a):
    return _unary(a, A.sign)


def _floor(a):
    return _unary(a, A.floor)


def _ceil(a):
    return _unary(a, A.ceil)


def _fma(a, b, c):
    dtype = _data_dtype(a, b, c)
    if not all(isinstance(value, A.Ten) for value in (a, b, c)):
        raise Unsupported("prims.fma requires three tensors in the reviewed profile")
    if a.shape != b.shape or a.shape != c.shape:
        raise Unsupported("prims.fma broadcasting is outside the reviewed profile")
    if any(value.dtype != dtype for value in (a, b, c)):
        raise Unsupported("prims.fma mixed dtypes are outside the reviewed profile")
    return A.fma(a, b, c)


def _clamp(a, min=None, max=None):
    dtype = _data_dtype(a)
    if min is None and max is None:
        raise Invalid("clamp requires min, max or both")
    return A.clamp(a, _scalar(min, dtype, "min", optional=True),
                   _scalar(max, dtype, "max", optional=True))


def _clamp_min(a, min):
    dtype = _data_dtype(a)
    return A.clamp_min(a, _scalar(min, dtype, "min"))


def _clamp_max(a, max):
    dtype = _data_dtype(a)
    return A.clamp_max(a, _scalar(max, dtype, "max"))


def _maximum(a, b):
    _binary_operands(a, b)
    if not isinstance(a, A.Ten) or not isinstance(b, A.Ten):
        raise Unsupported("maximum requires two tensor operands in the selected overload")
    return A.maximum(a, b)


def _minimum(a, b):
    _binary_operands(a, b)
    if not isinstance(a, A.Ten) or not isinstance(b, A.Ten):
        raise Unsupported("minimum requires two tensor operands in the selected overload")
    return A.minimum(a, b)


def _where(condition, a, b):
    if not isinstance(condition, A.Ten) or condition.dtype != "b8":
        raise Invalid("where condition must be a boolean tensor")
    a, b, _ = _binary_operands(a, b, require_left_tensor=False)
    return A.where(condition, a, b)


def _masked_fill(a, mask, value):
    dtype = _data_dtype(a)
    if not isinstance(mask, A.Ten) or mask.dtype != "b8":
        raise Invalid("masked_fill mask must be a boolean tensor")
    return A.masked_fill(a, mask, _scalar(value, dtype, "fill value"))


def _floor_divide(a, b):
    a, b, _ = _binary_operands(a, b)
    return A.floor_divide(a, b)


OPS = {
    "aten.add": _add, "aten.sub": _sub, "aten.rsub": _rsub, "aten.mul": _mul,
    "aten.div": _div, "aten.neg": _neg, "aten.reciprocal": _reciprocal,
    "aten.abs": _abs, "aten.sign": _sign, "prims.fma": _fma,
    "aten.floor": _floor, "aten.ceil": _ceil, "aten.clamp": _clamp,
    "aten.clamp_min": _clamp_min, "aten.clamp_max": _clamp_max,
    "aten.maximum": _maximum, "aten.minimum": _minimum, "aten.where": _where,
    "aten.masked_fill": _masked_fill, "aten.floor_divide": _floor_divide,
}

_real_reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
_real_check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE,
                                         argument_decoder=_decode_argument)


def _complete_tensor(value, name, *, boolean=False):
    if not isinstance(value, dict) or set(value) != {"t", "v", "shape"}:
        raise Invalid(f"{name} requires a complete physical tensor record")
    return _point_tensor(value, name, boolean=boolean)


def _masked_fill_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "input", "mask", "value", "upstream"}, "masked_fill task")
    if set(inp) != {"op", "input", "mask", "value", "upstream"}:
        raise Unsupported("masked_fill task contains undeclared fields")
    if inp["op"] != "torch.Tensor.masked_fill":
        raise Unsupported("this profile selects the public out-of-place Tensor.masked_fill task")

    x = _complete_tensor(inp["input"], "input")
    mask = _complete_tensor(inp["mask"], "mask", boolean=True)
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if upstream.dtype != x.dtype:
        raise Invalid("upstream must have the input dtype")
    fill = _scalar(decode(inp["value"]), x.dtype, "fill value")
    output_shape = A._broadcast_shape(x.shape, mask.shape)
    if upstream.shape != output_shape:
        raise Invalid(f"upstream shape must equal the broadcast output shape {output_shape}")

    used_mask = mask
    if mode == "inverted":
        used_mask = A.unary(lambda item: not item, mask, out_dtype="b8")
    out = A.masked_fill(x, used_mask, fill)

    if mode == "unmasked_gradient":
        grad = A.sum_to_shape(upstream, x.shape)
    else:
        selected = A.masked_fill(upstream, mask, Fraction(0))
        grad = selected if mode == "unreduced_gradient" else A.sum_to_shape(selected, x.shape)
    fields = {"out": encode(out), "grad": encode(grad)}
    if mode == "omit_gradient":
        fields.pop("grad")
    return Result("value", fields)


def _where_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "condition", "input", "other", "upstream"}, "where task")
    if set(inp) != {"op", "condition", "input", "other", "upstream"}:
        raise Unsupported("where task contains undeclared fields")
    if inp["op"] != "torch.where":
        raise Unsupported("this profile selects the public Tensor/Tensor torch.where task")
    if not isinstance(inp["input"], dict) or not isinstance(inp["other"], dict):
        raise Unsupported("this profile selects Tensor input and Tensor other")

    condition = _complete_tensor(inp["condition"], "condition", boolean=True)
    x = _complete_tensor(inp["input"], "input")
    other = _complete_tensor(inp["other"], "other")
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if other.dtype != x.dtype:
        raise Unsupported("mixed data dtypes are outside the selected where task")
    if upstream.dtype != x.dtype:
        raise Invalid("upstream must have the data dtype")
    output_shape = A._broadcast_shape(condition.shape, x.shape, other.shape)
    if upstream.shape != output_shape:
        raise Invalid(f"upstream shape must equal the broadcast output shape {output_shape}")

    out = A.where(condition, other, x) if mode == "inverted" else A.where(condition, x, other)
    true_upstream = A.where(condition, upstream, Fraction(0))
    false_upstream = A.where(condition, Fraction(0), upstream)
    if mode == "inverted":
        grad_input = A.sum_to_shape(false_upstream, x.shape)
        grad_other = A.sum_to_shape(true_upstream, other.shape)
    else:
        grad_input = true_upstream if mode == "unreduced_input" else A.sum_to_shape(true_upstream, x.shape)
        selected_other = true_upstream if mode == "same_branch" else false_upstream
        grad_other = A.sum_to_shape(selected_other, other.shape)
    fields = {"out": encode(out), "grad_input": encode(grad_input),
              "grad_other": encode(grad_other)}
    if mode == "omit_other":
        fields.pop("grad_other")
    return Result("value", fields)


def _clamp_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "input", "min", "max", "upstream"}, "clamp task")
    if set(inp) != {"op", "input", "min", "max", "upstream"}:
        raise Unsupported("clamp task contains undeclared fields")
    if inp["op"] != "torch.clamp":
        raise Unsupported("this profile selects public functional torch.clamp with scalar bounds")

    x = _complete_tensor(inp["input"], "input")
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if upstream.dtype != x.dtype:
        raise Invalid("upstream must have the input dtype")
    if upstream.shape != x.shape:
        raise Invalid(f"upstream shape must equal the input shape {x.shape}")
    lo = _scalar(decode(inp["min"]), x.dtype, "min", optional=True)
    hi = _scalar(decode(inp["max"]), x.dtype, "max", optional=True)
    if lo is None and hi is None:
        raise Invalid("torch.clamp requires min, max or both")

    if mode == "min_wins" and lo is not None and hi is not None:
        out = A.clamp_min(A.clamp_max(x, hi), lo)
    else:
        out = A.clamp(x, lo, hi)
    grad_values = np.empty(x.shape, dtype=object)
    for index in np.ndindex(x.shape):
        value = x.a[index].v
        if mode == "strict_boundaries":
            active = (lo is None or value > lo) and (hi is None or value < hi)
        elif mode == "ignore_upper":
            active = lo is None or value >= lo
        else:
            active = (lo is None or value >= lo) and (hi is None or value <= hi)
        grad_values[index] = upstream.a[index] if active else R(0)
    fields = {"out": encode(out), "grad": encode(A.Ten(grad_values, x.dtype))}
    if mode == "omit_gradient":
        fields.pop("grad")
    return Result("value", fields)


def _extrema_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "input", "other", "upstream"}, "extrema task")
    if set(inp) != {"op", "input", "other", "upstream"}:
        raise Unsupported("extrema task contains undeclared fields")
    if inp["op"] not in {"torch.maximum", "torch.minimum"}:
        raise Unsupported("this profile selects public Tensor/Tensor torch.maximum or torch.minimum")
    if not isinstance(inp["input"], dict) or not isinstance(inp["other"], dict):
        raise Unsupported("this profile requires two Tensor operands")

    x = _complete_tensor(inp["input"], "input")
    other = _complete_tensor(inp["other"], "other")
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if other.dtype != x.dtype:
        raise Unsupported("mixed data dtypes are outside the selected extrema task")
    if upstream.dtype != x.dtype:
        raise Invalid("upstream must have the data dtype")
    output_shape = A._broadcast_shape(x.shape, other.shape)
    if upstream.shape != output_shape:
        raise Invalid(f"upstream shape must equal the broadcast output shape {output_shape}")

    selected_op = inp["op"]
    if mode == "swap_ops":
        selected_op = "torch.minimum" if selected_op == "torch.maximum" else "torch.maximum"
    out = A.maximum(x, other) if selected_op == "torch.maximum" else A.minimum(x, other)
    xb, yb, ub = np.broadcast_arrays(x.a, other.a, upstream.a)
    grad_x_values = np.empty(output_shape, dtype=object)
    grad_other_values = np.empty(output_shape, dtype=object)
    for index in np.ndindex(output_shape):
        xv, yv, uv = xb[index], yb[index], ub[index]
        if xv.v == yv.v:
            if mode == "tie_to_input":
                grad_x_values[index], grad_other_values[index] = uv, R(0)
            else:
                grad_x_values[index] = uv * R(Fraction(1, 2))
                grad_other_values[index] = uv * R(Fraction(1, 2))
        else:
            input_selected = xv.v > yv.v if selected_op == "torch.maximum" else xv.v < yv.v
            grad_x_values[index] = uv if input_selected else R(0)
            grad_other_values[index] = R(0) if input_selected else uv
    grad_input_full = A.Ten(grad_x_values, x.dtype)
    grad_other_full = A.Ten(grad_other_values, x.dtype)
    grad_input = grad_input_full if mode == "unreduced_input" else A.sum_to_shape(grad_input_full, x.shape)
    grad_other = A.sum_to_shape(grad_other_full, other.shape)
    fields = {"out": encode(out), "grad_input": encode(grad_input),
              "grad_other": encode(grad_other)}
    if mode == "omit_other":
        fields.pop("grad_other")
    return Result("value", fields)


def reference(profile, inp):
    if profile == REAL_PROFILE:
        return _real_reference(profile, inp)
    if profile == MASKED_FILL_PROFILE:
        return _masked_fill_autograd(inp)
    if profile == WHERE_PROFILE:
        return _where_autograd(inp)
    if profile == CLAMP_PROFILE:
        return _clamp_autograd(inp)
    if profile == EXTREMA_PROFILE:
        return _extrema_autograd(inp)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def _check_masked_fill_output(inp, observed):
    expected = _masked_fill_autograd(inp).fields
    if not isinstance(observed, dict) or set(observed) != {"out", "grad"}:
        return False
    x = _complete_tensor(inp["input"], "input")
    shapes = {"out": A._broadcast_shape(x.shape, _complete_tensor(inp["mask"], "mask", boolean=True).shape),
              "grad": x.shape}
    candidate = {}
    for name in ("out", "grad"):
        raw = observed[name]
        if not isinstance(raw, dict) or set(raw) != {"t", "v", "shape"}:
            return False
        try:
            tensor = _complete_tensor(raw, f"observed {name}")
        except (Invalid, Unsupported):
            return False
        if tensor.dtype != x.dtype or tensor.shape != shapes[name]:
            return False
        candidate[name] = encode(tensor)
    return check_enclosed_output(expected, candidate)


def _check_where_output(inp, observed):
    expected = _where_autograd(inp).fields
    required = {"out", "grad_input", "grad_other"}
    if not isinstance(observed, dict) or set(observed) != required:
        return False
    try:
        condition = _complete_tensor(inp["condition"], "condition", boolean=True)
        x = _complete_tensor(inp["input"], "input")
        other = _complete_tensor(inp["other"], "other")
        shapes = {"out": A._broadcast_shape(condition.shape, x.shape, other.shape),
                  "grad_input": x.shape, "grad_other": other.shape}
        candidate = {}
        for name in ("out", "grad_input", "grad_other"):
            tensor = _complete_tensor(observed[name], f"observed {name}")
            if tensor.dtype != x.dtype or tensor.shape != shapes[name]:
                return False
            candidate[name] = encode(tensor)
    except (Invalid, Unsupported):
        return False
    return check_enclosed_output(expected, candidate)


def _check_clamp_output(inp, observed):
    if not isinstance(observed, dict) or set(observed) != {"out", "grad"}:
        return False
    try:
        expected = _clamp_autograd(inp).fields
        x = _complete_tensor(inp["input"], "input")
        candidate = {}
        for name in ("out", "grad"):
            tensor = _complete_tensor(observed[name], f"observed {name}")
            if tensor.dtype != x.dtype or tensor.shape != x.shape:
                return False
            candidate[name] = encode(tensor)
    except (Invalid, Unsupported, KeyError, TypeError):
        return False
    return check_enclosed_output(expected, candidate)


def _check_extrema_output(inp, observed):
    required = {"out", "grad_input", "grad_other"}
    if not isinstance(observed, dict) or set(observed) != required:
        return False
    try:
        expected = _extrema_autograd(inp).fields
        x = _complete_tensor(inp["input"], "input")
        other = _complete_tensor(inp["other"], "other")
        shapes = {"out": A._broadcast_shape(x.shape, other.shape),
                  "grad_input": x.shape, "grad_other": other.shape}
        candidate = {}
        for name in ("out", "grad_input", "grad_other"):
            tensor = _complete_tensor(observed[name], f"observed {name}")
            if tensor.dtype != x.dtype or tensor.shape != shapes[name]:
                return False
            candidate[name] = encode(tensor)
    except (Invalid, Unsupported, KeyError, TypeError):
        return False
    return check_enclosed_output(expected, candidate)


def check_output(profile, inp, observed):
    if profile == REAL_PROFILE:
        return _real_check_output(profile, inp, observed)
    if profile == MASKED_FILL_PROFILE:
        return _check_masked_fill_output(inp, observed)
    if profile == WHERE_PROFILE:
        return _check_where_output(inp, observed)
    if profile == CLAMP_PROFILE:
        return _check_clamp_output(inp, observed)
    if profile == EXTREMA_PROFILE:
        return _check_extrema_output(inp, observed)
    return False


def _mut(**replacements):
    return make_reference(COVERS, extra={**OPS, **replacements}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


def _add_no_alpha(a, b, *, alpha=1):
    a, b, _ = _binary_operands(a, b)
    return A.add(a, b)


def _sub_no_alpha(a, b, *, alpha=1):
    a, b, _ = _binary_operands(a, b)
    return A.sub(a, b)


def _rsub_unswapped(a, b, *, alpha=1):
    return _sub(a, b, alpha=alpha)


def _floor_as_trunc(a, b, *, rounding_mode=None):
    return _div(a, b, rounding_mode="trunc" if rounding_mode == "floor" else rounding_mode)


def _reciprocal_identity(a):
    _data_dtype(a)
    return a


def _sign_zero_one(a):
    _data_dtype(a)
    return A.unary(lambda x: R(1) if x >= 0 else R(-1), a)


def _fma_without_c(a, b, c):
    _fma(a, b, c)
    return A.mul(a, b)


def _clamp_min_wins(a, min=None, max=None):
    dtype = _data_dtype(a)
    lo = _scalar(min, dtype, "min", optional=True)
    hi = _scalar(max, dtype, "max", optional=True)
    out = A.clamp_max(a, hi) if hi is not None else a
    return A.clamp_min(out, lo) if lo is not None else out


def _where_swapped(condition, a, b):
    return _where(condition, b, a)


def _masked_fill_inverted(a, mask, value):
    dtype = _data_dtype(a)
    return A.where(mask, a, _scalar(value, dtype, "fill value"))


MUTANTS = {
    "AR.M1": _mut(**{"aten.add": _add_no_alpha, "aten.sub": _sub_no_alpha}),
    "AR.M2": _mut(**{"aten.rsub": _rsub_unswapped}),
    "AR.M3": _mut(**{"aten.div": _floor_as_trunc}),
    "AR.M4": _mut(**{"aten.reciprocal": _reciprocal_identity}),
    "AR.M5": _mut(**{"aten.sign": _sign_zero_one}),
    "AR.M6": _mut(**{"prims.fma": _fma_without_c}),
    "AR.M7": _mut(**{"aten.floor": _ceil}),
    "AR.M8": _mut(**{"aten.clamp": _clamp_min_wins}),
    "AR.M9": _mut(**{"aten.maximum": _minimum, "aten.minimum": _maximum}),
    "AR.M10": _mut(**{"aten.where": _where_swapped}),
    "AR.M11": _mut(**{"aten.masked_fill": _masked_fill_inverted}),
    "AR.M12": make_reference(COVERS, extra=OPS, profiles=PROFILE),
    "AR.M13": lambda profile, inp: _masked_fill_autograd(inp, mode="inverted")
    if profile == MASKED_FILL_PROFILE else reference(profile, inp),
    "AR.M14": lambda profile, inp: _masked_fill_autograd(inp, mode="unmasked_gradient")
    if profile == MASKED_FILL_PROFILE else reference(profile, inp),
    "AR.M15": lambda profile, inp: _masked_fill_autograd(inp, mode="unreduced_gradient")
    if profile == MASKED_FILL_PROFILE else reference(profile, inp),
    "AR.M16": lambda profile, inp: _masked_fill_autograd(inp, mode="omit_gradient")
    if profile == MASKED_FILL_PROFILE else reference(profile, inp),
    "AR.M17": lambda profile, inp: _where_autograd(inp, mode="inverted")
    if profile == WHERE_PROFILE else reference(profile, inp),
    "AR.M18": lambda profile, inp: _where_autograd(inp, mode="unreduced_input")
    if profile == WHERE_PROFILE else reference(profile, inp),
    "AR.M19": lambda profile, inp: _where_autograd(inp, mode="same_branch")
    if profile == WHERE_PROFILE else reference(profile, inp),
    "AR.M20": lambda profile, inp: _where_autograd(inp, mode="omit_other")
    if profile == WHERE_PROFILE else reference(profile, inp),
    "AR.M21": lambda profile, inp: _clamp_autograd(inp, mode="strict_boundaries")
    if profile == CLAMP_PROFILE else reference(profile, inp),
    "AR.M22": lambda profile, inp: _clamp_autograd(inp, mode="ignore_upper")
    if profile == CLAMP_PROFILE else reference(profile, inp),
    "AR.M23": lambda profile, inp: _clamp_autograd(inp, mode="min_wins")
    if profile == CLAMP_PROFILE else reference(profile, inp),
    "AR.M24": lambda profile, inp: _clamp_autograd(inp, mode="omit_gradient")
    if profile == CLAMP_PROFILE else reference(profile, inp),
    "AR.M25": lambda profile, inp: _extrema_autograd(inp, mode="tie_to_input")
    if profile == EXTREMA_PROFILE else reference(profile, inp),
    "AR.M26": lambda profile, inp: _extrema_autograd(inp, mode="swap_ops")
    if profile == EXTREMA_PROFILE else reference(profile, inp),
    "AR.M27": lambda profile, inp: _extrema_autograd(inp, mode="unreduced_input")
    if profile == EXTREMA_PROFILE else reference(profile, inp),
    "AR.M28": lambda profile, inp: _extrema_autograd(inp, mode="omit_other")
    if profile == EXTREMA_PROFILE else reference(profile, inp),
}
READINGS = {}
