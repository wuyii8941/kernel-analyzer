"""Adapter for contracts/LAYOUT.yaml: exact logical values of selected layout and data-movement operations.

The real profile admits finite floating and integer tensors; the boolean profile admits b8 tensors and returns
its complete result in the boolean output bool_out (LY.EXACT_DATA)."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from pathlib import Path

import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, encode, make_output_checker, make_reference
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, require_fields
from kernel_spec.real import R

_HERE = Path(__file__).resolve()
# The contract next to this adapter: <root>/contracts/LAYOUT.yaml when staged as <root>/adapters/LAYOUT.py, or the
# paired draft contract when read from contracts/draft/. Never the top-level validated contract by default.
_CONTRACT = (_HERE.with_name("LAYOUT_path_a_A2_20261020.yaml") if _HERE.parent.name == "draft"
             else _HERE.parents[1] / "contracts" / "LAYOUT.yaml")
COVERS = yaml.safe_load(_CONTRACT.read_text())["covers"]
REAL_PROFILE = "real"
PAD_PROFILE = "torch210_functional_constant_pad_autograd"
CAT_PROFILE = "torch210_cat_autograd"
BOOL_PROFILE = "boolean"
PROFILE = (REAL_PROFILE,)
DTYPES = {"f16", "bf16", "f32", "f64"}
INT_DTYPES = {"i64", "i32", "i16", "i8", "u8"}
BOOL_DTYPES = {"b8"}
REAL_DTYPES = DTYPES | INT_DTYPES
ANY_DTYPES = REAL_DTYPES | BOOL_DTYPES


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the finite LAYOUT profile")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _exact_data_tensor(value, name, dtype):
    """Integer or boolean tensor record: integer literals within the dtype range, or boolean literals (LY.EXACT_DATA)."""
    if isinstance(value, dict) and "v" in value:
        low, high = A.INT_RANGE.get(dtype, (0, 1))
        for raw in _raw_leaves(value["v"]):
            if dtype in BOOL_DTYPES:
                if not isinstance(raw, (bool, np.bool_)):
                    raise Invalid(f"{name} b8 values must be boolean literals")
            elif isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                raise Invalid(f"{name} {dtype} values must be integer literals")
            elif not low <= int(raw) <= high:
                raise Invalid(f"{name} value {int(raw)} is not representable as {dtype}")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten) or tensor.dtype != dtype:
        raise Invalid(f"{name} must be a {dtype} tensor")
    return tensor


def _point_tensor(value, name, allowed=DTYPES):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in allowed:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the selected LAYOUT profile")
    if dtype in INT_DTYPES | BOOL_DTYPES:
        return _exact_data_tensor(value, name, dtype)
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (float, np.floating)) and raw == 0 and math.copysign(1.0, float(raw)) < 0:
                raise Invalid(f"{name} contains negative zero outside the logical-value profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in DTYPES:
        raise Unsupported(f"{name} dtype {tensor.dtype!r} is outside the finite LAYOUT profile")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains infinity outside the finite LAYOUT profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


TENSOR_PARAMETERS = {"a", "src", "grad_output"}   # plus the tensor list "tensors"


def _contains_tensor_record(value):
    if isinstance(value, dict):
        return "t" in value or "v" in value
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor_record(item) for item in value)
    return False


def _argument_decoder(allowed):
    def decode_argument(op, name, value, arguments):
        if name == "tensors":
            if not isinstance(value, (list, tuple)) or not value:
                raise Invalid(f"{op}: tensors must be a nonempty list")
            return [_point_tensor(item, f"tensors[{index}]", allowed) for index, item in enumerate(value)]
        if name not in TENSOR_PARAMETERS:
            # A tensor for a size, dimension, index, range, repeat count or scalar parameter selects another
            # overload (e.g. narrow.Tensor, repeat_interleave.self_Tensor): a legal call outside the profiles.
            if _contains_tensor_record(value):
                raise Unsupported(f"{op}: tensor-valued {name} belongs to another overload outside the selected profiles")
            return decode(value)
        if isinstance(value, dict) and ("t" in value or "v" in value):
            return _point_tensor(value, name, allowed)
        return decode(value)
    return decode_argument


_decode_argument = _argument_decoder(REAL_DTYPES)
_decode_bool_argument = _argument_decoder(BOOL_DTYPES)


def _tensor(value, name="input"):
    # The profile's argument decoder has already restricted the dtype.
    if not isinstance(value, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if value.dtype not in ANY_DTYPES:
        raise Unsupported(f"{name} dtype {value.dtype!r} is outside the selected LAYOUT profiles")
    return value


def _same_dtype(tensors):
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise Invalid("a nonempty tensor list is required")
    tensors = [_tensor(value, f"tensor {index}") for index, value in enumerate(tensors)]
    if len({value.dtype for value in tensors}) != 1:
        raise Unsupported("mixed tensor dtypes are outside the finite LAYOUT profile")
    return tensors, tensors[0].dtype


def _scalar(value, dtype, name):
    if dtype in BOOL_DTYPES:
        return _bool_fill(value, name)
    if isinstance(value, A.Ten) or isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be a real scalar")
    if dtype in INT_DTYPES:
        return _integer_fill(value, dtype, name)
    try:
        item = value if isinstance(value, R) else R(Fraction(value))
    except OverflowError:
        # Fraction(+-inf): an infinite fill, outside the finite profiles (not an implementation error).
        raise Unsupported(f"infinite {name} is outside the selected LAYOUT profiles") from None
    except (TypeError, ValueError, ZeroDivisionError):
        raise Invalid(f"{name} must be a finite real scalar") from None
    if not item.exact:
        raise Invalid(f"{name} must be exact")
    if not item.finite:
        raise Unsupported(f"infinite {name} is outside the finite LAYOUT profile")
    stored = ieee.round_to(item, dtype)
    if not stored.exact or stored.v != item.v:
        raise Invalid(f"{name}={item!r} is not exactly representable as {dtype}")
    return item.v


def _exact_scalar(value, name):
    try:
        item = value if isinstance(value, R) else R(Fraction(value))
    except OverflowError:
        # Fraction(+-inf): an infinite fill, outside the finite profiles (not an implementation error).
        raise Unsupported(f"infinite {name} is outside the selected LAYOUT profiles") from None
    except (TypeError, ValueError, ZeroDivisionError):
        raise Invalid(f"{name} must be a finite real scalar") from None
    if not item.exact:
        raise Invalid(f"{name} must be exact")
    if not item.finite:
        raise Unsupported(f"infinite {name} is outside the selected LAYOUT profiles")
    return item.v


def _integer_fill(value, dtype, name):
    """LY.EXACT_DATA: an integer tensor stores the fill only when the received value is an integer in range."""
    exact = _exact_scalar(value, name)
    if exact.denominator != 1:
        raise Unsupported(f"{name}={exact} is not an integer; its conversion to {dtype} is not documented")
    low, high = A.INT_RANGE[dtype]
    if not low <= exact.numerator <= high:
        raise Unsupported(f"{name}={exact} is outside the {dtype} range; its conversion is not documented")
    return int(exact.numerator)


def _bool_fill(value, name):
    """LY.EXACT_DATA: a b8 tensor stores False/True, or the numeric values 0 and 1 that denote them."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, A.Ten):
        raise Invalid(f"{name} must be a scalar")
    exact = _exact_scalar(value, name)
    if exact not in (0, 1):
        raise Unsupported(f"{name}={exact} is neither 0 nor 1; its conversion to b8 is not documented")
    return exact == 1


def _zero(dtype):
    return False if dtype in BOOL_DTYPES else 0 if dtype in INT_DTYPES else R(0)


def _view(a, shape):
    return A.view(_tensor(a), shape)


def _reshape(a, shape):
    return A.reshape(_tensor(a), shape)


def _unsafe_view(a, size):
    return A.view(_tensor(a), size)


def _permute(a, dims):
    return A.permute(_tensor(a), dims)


def _transpose(a, dim0, dim1):
    return A.transpose(_tensor(a), dim0, dim1)


def _t(a):
    return A.t(_tensor(a))


def _expand(a, size, *, implicit=False):
    if not isinstance(implicit, (bool, np.bool_)):
        raise Invalid("expand implicit must be boolean")
    if implicit:
        raise Unsupported("implicit expand is outside the reviewed profile")
    return A.expand(_tensor(a), size)


def _unsqueeze(a, dim):
    return A.unsqueeze(_tensor(a), dim)


def _squeeze(a, dim=None):
    return A.squeeze(_tensor(a), dim)


def _select(a, dim, index):
    return A.select(_tensor(a), dim, index)


def _slice(a, dim=0, start=None, end=None, step=1):
    return A.slice_(_tensor(a), dim, start, end, step)


def _narrow(a, dim, start, length):
    a = _tensor(a)
    dim = A._layout_axis(dim, len(a.shape))
    start, length = A._layout_sizes([start, length], "narrow range")
    return A.narrow(a, dim, start, length)


def _cat(tensors, dim=0):
    tensors, _ = _same_dtype(tensors)
    return A.cat(tensors, dim)


def _stack(tensors, dim=0):
    tensors, _ = _same_dtype(tensors)
    return A.stack(tensors, dim)


def _split(a, split_size, dim=0):
    return A.split(_tensor(a), split_size, dim)


def _split_with_sizes(a, split_sizes, dim=0):
    return A.split_with_sizes(_tensor(a), split_sizes, dim)


def _constant_pad_nd(a, pad, value=0):
    a = _tensor(a)
    return A.constant_pad_nd(a, pad, _scalar(value, a.dtype, "padding value"))


PAD_MODES = ("constant", "reflect", "replicate", "circular")


def _pad(a, pad, mode="constant", value=None):
    """LY.ATEN_PAD: aten.pad in constant mode is constant padding with the documented default value zero."""
    a = _tensor(a)
    if not isinstance(mode, str) or mode not in PAD_MODES:
        raise Invalid(f"pad mode {mode!r} is not one of the documented modes")
    if mode != "constant":
        raise Unsupported(f"pad mode {mode!r} is outside the selected profiles")
    fill = _scalar(0 if value is None else value, a.dtype, "padding value")
    return A.constant_pad_nd(a, pad, fill)


def _slice_backward(grad_output, input_sizes, dim, start, end, step):
    """LY.SLICE_BACKWARD: zeros of input_sizes with grad_output placed at the slice dim[start:end:step]."""
    grad_output = _tensor(grad_output, "grad_output")
    sizes = A._layout_sizes(input_sizes, "slice_backward input_sizes")
    if any(size < 0 for size in sizes):
        raise Invalid("slice_backward input_sizes must be nonnegative")
    if len(sizes) != len(grad_output.shape):
        raise Invalid("slice_backward input_sizes must have the rank of grad_output")
    base = np.empty(tuple(sizes), dtype=object)
    base.fill(_zero(grad_output.dtype))
    return A.slice_scatter(A.Ten(base, grad_output.dtype), grad_output, dim, start, end, step)


def _repeat_interleave(a, repeats=None, dim=None, *, output_size=None):
    """LY.REPEAT_INTERLEAVE: self_int overload; each element along dim is repeated `repeats` consecutive times."""
    a = _tensor(a)
    if repeats is None:
        raise Unsupported("the repeats-only repeat_interleave.Tensor overload is outside the selected profiles")
    if isinstance(repeats, A.Ten):
        raise Unsupported("tensor-valued repeats (repeat_interleave.self_Tensor) are outside the selected profiles")
    repeats = A._layout_sizes([repeats], "repeat_interleave repeats")[0]
    if repeats < 0:
        raise Invalid("repeat_interleave repeats is a number of repetitions and cannot be negative")
    array = a.a
    if dim is None:
        array, axis = array.reshape(-1), 0
    else:
        axis = A._layout_axis(dim, array.ndim)
    total = array.shape[axis] * repeats
    if output_size is not None:
        if A._layout_sizes([output_size], "repeat_interleave output_size")[0] != total:
            raise Invalid(f"output_size must equal the total output size {total} along the selected axis")
    shape = list(array.shape)
    shape[axis] = total
    out = np.empty(tuple(shape), dtype=object)
    for index in np.ndindex(out.shape):
        source = list(index)
        source[axis] = index[axis] // repeats
        out[index] = array[tuple(source)]
    return A.Ten(out, a.dtype)


def _select_scatter(a, src, dim, index):
    tensors, _ = _same_dtype([a, src])
    return A.select_scatter(tensors[0], tensors[1], dim, index)


def _slice_scatter(a, src, dim=0, start=None, end=None, step=1):
    tensors, _ = _same_dtype([a, src])
    return A.slice_scatter(tensors[0], tensors[1], dim, start, end, step)


def _clone(a, *, memory_format=None):
    if memory_format is not None:
        raise Unsupported("nondefault clone memory_format is outside the reviewed profile")
    return A.clone(_tensor(a))


def _identity_copy(a):
    return A.clone(_tensor(a))


def _triu(a, diagonal=0):
    a = _tensor(a)
    if len(a.shape) < 2:
        raise Invalid("triu requires a tensor with at least two dimensions")
    diagonal = A._layout_sizes([diagonal], "triu diagonal")[0]
    return A.triu(a, diagonal)


def _tril(a, diagonal=0):
    a = _tensor(a)
    if len(a.shape) < 2:
        raise Invalid("tril requires a tensor with at least two dimensions")
    diagonal = A._layout_sizes([diagonal], "tril diagonal")[0]
    return A.tril(a, diagonal)


OPS = {
    "aten.view": _view, "aten.reshape": _reshape, "aten._unsafe_view": _unsafe_view,
    "aten.permute": _permute, "aten.transpose": _transpose, "aten.t": _t,
    "aten.expand": _expand, "aten.unsqueeze": _unsqueeze, "aten.squeeze": _squeeze,
    "aten.select": _select, "aten.slice": _slice, "aten.narrow": _narrow,
    "aten.cat": _cat, "aten.stack": _stack, "aten.split": _split,
    "aten.split_with_sizes": _split_with_sizes, "aten.constant_pad_nd": _constant_pad_nd,
    "aten.select_scatter": _select_scatter, "aten.slice_scatter": _slice_scatter,
    "aten.clone": _clone, "aten.lift_fresh_copy": _identity_copy,
    "aten.lift_fresh": _identity_copy, "aten.alias": _identity_copy,
    "aten.triu": _triu, "aten.tril": _tril,
    "aten.pad": _pad, "aten.slice_backward": _slice_backward, "aten.repeat_interleave": _repeat_interleave,
}
ATEN_COVERS = [name for name in COVERS if name.startswith("aten.")]
missing = set(ATEN_COVERS) - set(OPS)
if missing:
    raise ImportError(f"LAYOUT covers ATen operations without semantics: {sorted(missing)}")


def _bool_rename(reference):
    def renamed(profile, inp):
        result = reference(profile, inp)
        if result.outcome == "value":
            return Result("value", {"bool_out": result.fields["out"]})
        return result
    return renamed


def _references(ops):
    real = make_reference(ATEN_COVERS, extra=ops, profiles=(REAL_PROFILE,), argument_decoder=_decode_argument)
    boolean = _bool_rename(make_reference(ATEN_COVERS, extra=ops, profiles=(BOOL_PROFILE,),
                                          argument_decoder=_decode_bool_argument))
    return real, boolean


_real_reference, _bool_reference = _references(OPS)
_real_check_output = make_output_checker(ATEN_COVERS, extra=OPS, profiles=(REAL_PROFILE,),
                                         argument_decoder=_decode_argument)
_bool_check_output = make_output_checker(ATEN_COVERS, extra=OPS, profiles=(BOOL_PROFILE,),
                                         argument_decoder=_decode_bool_argument)


def _complete_tensor(value, name):
    if not isinstance(value, dict) or set(value) != {"t", "v", "shape"}:
        raise Invalid(f"{name} requires a complete physical tensor record")
    return _point_tensor(value, name)


def _pad_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "input", "pad", "mode", "value", "upstream"}, "constant-pad task")
    if set(inp) != {"op", "input", "pad", "mode", "value", "upstream"}:
        raise Unsupported("constant-pad task contains undeclared fields")
    if inp["op"] != "torch.nn.functional.pad":
        raise Unsupported("this profile selects the public functional pad task")
    if inp["mode"] != "constant":
        raise Unsupported("this profile selects constant padding")

    x = _complete_tensor(inp["input"], "input")
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if not x.shape or any(length <= 0 for length in x.shape):
        raise Unsupported("the selected task requires a positive-rank tensor with positive input dimensions")
    if upstream.dtype != x.dtype:
        raise Invalid("upstream must have the input dtype")
    pad = A._layout_sizes(inp["pad"], "padding")
    lo, hi, output_shape = A._padding_layout(x.shape, pad)
    if upstream.shape != tuple(output_shape):
        raise Invalid(f"upstream shape must equal the padded output shape {tuple(output_shape)}")
    raw_value = Fraction(0) if inp["value"] is None else decode(inp["value"])
    fill = _scalar(raw_value, x.dtype, "padding value")

    forward_pad = [max(item, 0) for item in pad] if mode == "negative_as_zero" else pad
    out = A.constant_pad_nd(x, forward_pad, fill)
    grad = np.empty(x.shape, dtype=object)
    grad.fill(R(0))
    for index in np.ndindex(x.shape):
        output_index, retained = [], True
        for axis, coordinate in enumerate(index):
            crop_left, crop_right = max(-lo[axis], 0), max(-hi[axis], 0)
            if coordinate < crop_left or coordinate >= x.shape[axis] - crop_right:
                retained = False
                break
            shift = max(lo[axis], 0) - crop_left
            output_index.append(coordinate + shift)
        if retained:
            chosen = tuple(index) if mode == "unshifted_gradient" else tuple(output_index)
            if all(chosen[axis] < upstream.shape[axis] for axis in range(len(chosen))):
                grad[index] = upstream.a[chosen]
        elif mode == "cropped_passthrough":
            grad[index] = R(1)
    fields = {"out": encode(out), "grad": encode(A.Ten(grad, x.dtype))}
    if mode == "omit_gradient":
        fields.pop("grad")
    return Result("value", fields)


def _cat_autograd(inp, *, mode="correct"):
    require_fields(inp, {"op", "tensors", "dim", "upstream"}, "cat task")
    if set(inp) != {"op", "tensors", "dim", "upstream"}:
        raise Unsupported("cat task contains undeclared fields")
    if inp["op"] != "torch.cat":
        raise Unsupported("this profile selects the public functional torch.cat task")
    if not isinstance(inp["tensors"], (list, tuple)) or not inp["tensors"]:
        raise Invalid("torch.cat requires a nonempty tensor sequence")

    tensors = [_complete_tensor(value, f"tensors[{index}]")
               for index, value in enumerate(inp["tensors"])]
    tensors, dtype = _same_dtype(tensors)
    if any(len(tensor.shape) == 0 for tensor in tensors):
        raise Invalid("torch.cat does not accept scalar tensors")
    upstream = _complete_tensor(inp["upstream"], "upstream")
    if upstream.dtype != dtype:
        raise Invalid("upstream must have the input dtype")

    forward_inputs = list(reversed(tensors)) if mode == "reversed" else tensors
    out = A.cat(forward_inputs, inp["dim"])
    if upstream.shape != out.shape:
        raise Invalid(f"upstream shape must equal the concatenated output shape {out.shape}")

    effective = [tensor for tensor in tensors if tensor.shape != (0,)]
    grads = []
    if not effective:
        grads = [A.zeros_like(tensor) for tensor in tensors]
    else:
        dim = A._layout_axis(A._layout_sizes([inp["dim"]], "cat axis")[0],
                             len(effective[0].shape))
        offset = 0
        for tensor in tensors:
            if tensor.shape == (0,):
                grads.append(A.zeros_like(tensor))
                continue
            start = 0 if mode == "reset_offsets" else offset
            grads.append(A.slice_(upstream, dim, start, start + tensor.shape[dim]))
            offset += tensor.shape[dim]
    if mode == "omit_last_gradient":
        grads = grads[:-1]
    return Result("value", {"out": encode(out), "grads": [encode(grad) for grad in grads]})


def _aten_call(inp):
    return isinstance(inp, dict) and isinstance(inp.get("op"), str) and inp["op"].startswith("aten.")


def reference(profile, inp):
    if profile == REAL_PROFILE:
        return _real_reference(profile, inp)
    if profile == BOOL_PROFILE:
        return _bool_reference(profile, inp)
    if profile in (PAD_PROFILE, CAT_PROFILE) and _aten_call(inp):
        raise Unsupported(f"profile {profile} selects a public functional task, not an ATen call")
    if profile == PAD_PROFILE:
        return _pad_autograd(inp)
    if profile == CAT_PROFILE:
        return _cat_autograd(inp)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def _check_pad_output(inp, observed):
    expected = _pad_autograd(inp).fields
    if not isinstance(observed, dict) or set(observed) != {"out", "grad"}:
        return False
    try:
        x = _complete_tensor(inp["input"], "input")
        _, _, out_shape = A._padding_layout(x.shape, inp["pad"])
    except (Invalid, Unsupported):
        return False
    shapes = {"out": tuple(out_shape), "grad": x.shape}
    candidate = {}
    for name in ("out", "grad"):
        try:
            tensor = _complete_tensor(observed[name], f"observed {name}")
        except (Invalid, Unsupported):
            return False
        if tensor.dtype != x.dtype or tensor.shape != shapes[name]:
            return False
        candidate[name] = encode(tensor)
    return check_enclosed_output(expected, candidate)


def _check_cat_output(inp, observed):
    try:
        expected = _cat_autograd(inp).fields
        tensors = [_complete_tensor(value, f"tensors[{index}]")
                   for index, value in enumerate(inp["tensors"])]
        out = A.cat(tensors, inp["dim"])
        if not isinstance(observed, dict) or set(observed) != {"out", "grads"}:
            return False
        if not isinstance(observed["grads"], (list, tuple)) \
                or len(observed["grads"]) != len(tensors):
            return False
        observed_out = _complete_tensor(observed["out"], "observed out")
        if observed_out.dtype != out.dtype or observed_out.shape != out.shape:
            return False
        observed_grads = []
        for index, (raw, tensor) in enumerate(zip(observed["grads"], tensors)):
            grad = _complete_tensor(raw, f"observed grads[{index}]")
            if grad.dtype != tensor.dtype or grad.shape != tensor.shape:
                return False
            observed_grads.append(encode(grad))
        candidate = {"out": encode(observed_out), "grads": observed_grads}
    except (Invalid, Unsupported, KeyError, TypeError):
        return False
    return check_enclosed_output(expected, candidate)


def check_output(profile, inp, observed):
    if profile == REAL_PROFILE:
        return _real_check_output(profile, inp, observed)
    if profile == BOOL_PROFILE:
        if not isinstance(observed, dict) or set(observed) != {"bool_out"}:
            return False
        return _bool_check_output(profile, inp, {"out": observed["bool_out"]})
    if profile == PAD_PROFILE:
        return _check_pad_output(inp, observed)
    if profile == CAT_PROFILE:
        return _check_cat_output(inp, observed)
    return False


def _mut(**replacements):
    real, boolean = _references({**OPS, **replacements})

    def mutant(profile, inp):
        if profile == BOOL_PROFILE:
            return boolean(profile, inp)
        return real(profile, inp)
    return mutant


def _view_colmajor(a, shape):
    a = _tensor(a)
    shape = A._view_shape(a.shape, shape)
    flat = np.array(a.a.reshape(-1, order="F"), dtype=object)
    return A.Ten(flat.reshape(shape, order="F"), a.dtype)


def _permute_inverse(a, dims):
    a = _tensor(a)
    axes = A._permutation(len(a.shape), dims)
    inverse = [0] * len(axes)
    for index, axis in enumerate(axes):
        inverse[axis] = index
    return A.Ten(np.transpose(a.a, inverse), a.dtype)


def _transpose_identity(a, dim0, dim1):
    _transpose(a, dim0, dim1)
    return A.clone(a)


def _expand_identity(a, size, *, implicit=False):
    _expand(a, size, implicit=implicit)
    return A.clone(a)


def _squeeze_all(a, dim=None):
    _squeeze(a, dim)
    return A.squeeze(a)


def _slice_no_step(a, dim=0, start=None, end=None, step=1):
    _slice(a, dim, start, end, step)
    return A.slice_(a, dim, start, end, 1)


def _cat_reversed(tensors, dim=0):
    tensors, _ = _same_dtype(tensors)
    return A.cat(list(reversed(tensors)), dim)


def _split_drop_last(a, split_size, dim=0):
    pieces = _split(a, split_size, dim)
    return pieces[:-1]


def _pad_forward(a, pad, value=0):
    a = _tensor(a)
    fill = _scalar(value, a.dtype, "padding value")
    A._padding_layout(a.shape, pad)
    if len(pad) > 2 * len(a.shape):
        raise Invalid("padding contains too many dimensions")
    axes = list(range(len(a.shape)))[::-1]
    transposed = A.permute(a, axes)
    padded = A.constant_pad_nd(transposed, pad, fill)
    return A.permute(padded, axes)


def _select_scatter_identity(a, src, dim, index):
    _select_scatter(a, src, dim, index)
    return A.clone(a)


def _slice_scatter_identity(a, src, dim=0, start=None, end=None, step=1):
    _slice_scatter(a, src, dim, start, end, step)
    return A.clone(a)


def _triu_flipped(a, diagonal=0):
    _triu(a, diagonal)
    return A.triu(a, -diagonal)


def _pad_zero_fill(a, pad, mode="constant", value=None):
    _pad(a, pad, mode, value)
    return A.constant_pad_nd(a, pad, _zero(a.dtype))


def _pad_truncating_fill(a, pad, mode="constant", value=None):
    # Misreading: an integer tensor truncates a non-integral fill toward zero (undocumented conversion).
    a = _tensor(a)
    if mode != "constant":
        raise Unsupported("constant mode only")
    raw = Fraction(0 if value is None else value)
    if a.dtype in INT_DTYPES:
        fill = int(raw)  # int() truncates toward zero
    elif a.dtype in BOOL_DTYPES:
        fill = raw != 0
    else:
        fill = raw
    return A.constant_pad_nd(a, pad, fill)


def _slice_backward_at_zero(grad_output, input_sizes, dim, start, end, step):
    _slice_backward(grad_output, input_sizes, dim, start, end, step)
    return _slice_backward(grad_output, input_sizes, dim, 0, grad_output.shape[dim], 1)


def _slice_backward_no_step(grad_output, input_sizes, dim, start, end, step):
    _slice_backward(grad_output, input_sizes, dim, start, end, step)
    axis = A._layout_axis(dim, len(grad_output.shape))
    begin = A._slice_bounds(input_sizes[axis], start, end, step)[0]
    return _slice_backward(grad_output, input_sizes, dim, begin, begin + grad_output.shape[axis], 1)


def _repeat_tiled(a, repeats=None, dim=None, *, output_size=None):
    # Misreading: Tensor.repeat semantics, the whole axis is repeated instead of each element.
    expected = _repeat_interleave(a, repeats, dim, output_size=output_size)
    array = a.a.reshape(-1) if dim is None else a.a
    axis = 0 if dim is None else A._layout_axis(dim, a.a.ndim)
    out = np.empty(expected.shape, dtype=object)
    for index in np.ndindex(out.shape):
        source = list(index)
        source[axis] = index[axis] % array.shape[axis]
        out[index] = array[tuple(source)]
    return A.Ten(out, a.dtype)


def _repeat_flattened(a, repeats=None, dim=None, *, output_size=None):
    # Misreading: dim is ignored and the flattened input is always repeated.
    _repeat_interleave(a, repeats, dim, output_size=output_size)
    return _repeat_interleave(a, repeats, None)


def _tril_upper(a, diagonal=0):
    _tril(a, diagonal)
    return A.triu(a, diagonal)


MUTANTS = {
    "LY.M1": _mut(**{"aten.view": _view_colmajor}),
    "LY.M2": _mut(**{"aten.permute": _permute_inverse}),
    "LY.M3": _mut(**{"aten.transpose": _transpose_identity}),
    "LY.M4": _mut(**{"aten.expand": _expand_identity}),
    "LY.M5": _mut(**{"aten.squeeze": _squeeze_all}),
    "LY.M6": _mut(**{"aten.slice": _slice_no_step}),
    "LY.M7": _mut(**{"aten.cat": _cat_reversed}),
    "LY.M8": _mut(**{"aten.split": _split_drop_last}),
    "LY.M9": _mut(**{"aten.constant_pad_nd": _pad_forward}),
    "LY.M10": _mut(**{"aten.select_scatter": _select_scatter_identity,
                        "aten.slice_scatter": _slice_scatter_identity}),
    "LY.M11": _mut(**{"aten.triu": _triu_flipped}),
    "LY.M12": lambda profile, inp: _pad_autograd(inp, mode="negative_as_zero")
    if profile == PAD_PROFILE else reference(profile, inp),
    "LY.M13": lambda profile, inp: _pad_autograd(inp, mode="unshifted_gradient")
    if profile == PAD_PROFILE else reference(profile, inp),
    "LY.M14": lambda profile, inp: _pad_autograd(inp, mode="cropped_passthrough")
    if profile == PAD_PROFILE else reference(profile, inp),
    "LY.M15": lambda profile, inp: _pad_autograd(inp, mode="omit_gradient")
    if profile == PAD_PROFILE else reference(profile, inp),
    "LY.M16": lambda profile, inp: _cat_autograd(inp, mode="reversed")
    if profile == CAT_PROFILE else reference(profile, inp),
    "LY.M17": lambda profile, inp: _cat_autograd(inp, mode="reset_offsets")
    if profile == CAT_PROFILE else reference(profile, inp),
    "LY.M18": lambda profile, inp: _cat_autograd(inp, mode="omit_last_gradient")
    if profile == CAT_PROFILE else reference(profile, inp),
    "LY.M19": _mut(**{"aten.pad": _pad_zero_fill}),
    "LY.M20": _mut(**{"aten.pad": _pad_truncating_fill}),
    "LY.M21": _mut(**{"aten.slice_backward": _slice_backward_at_zero}),
    "LY.M22": _mut(**{"aten.slice_backward": _slice_backward_no_step}),
    "LY.M23": _mut(**{"aten.repeat_interleave": _repeat_tiled}),
    "LY.M24": _mut(**{"aten.repeat_interleave": _repeat_flattened}),
    "LY.M25": _mut(**{"aten.tril": _tril_upper}),
}
READINGS = {}
