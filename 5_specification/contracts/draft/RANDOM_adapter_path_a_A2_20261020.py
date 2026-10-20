"""Dropout scopes from contracts/RANDOM.yaml: deterministic dropout forward boundaries, native dropout backward with
an explicit mask, and native dropout forward given the mask the call returned (profile native_forward_given_mask)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, encode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Result, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "RANDOM.yaml").read_text())["covers"]
FORWARD_COVERS = ["aten.dropout", "torch.nn.functional.dropout"]
BACKWARD_COVERS = ["aten.native_dropout_backward"]
NATIVE_COVERS = ["aten.native_dropout"]
NATIVE_PROFILE = "native_forward_given_mask"
OBSERVED_MASK = "observed_mask"

# Harness conventions (path A rule v2). native_dropout returns (output, mask). Position 1 is defined by the task but
# is not compared: in the conditioned case it is the reference input itself, and the RANDOM output `out` is a numeric
# field that cannot carry Boolean values (decision RN3.MASK_POSITION).
ATEN_RETURNS = {"aten.native_dropout": {
    "positions": 2,
    "not_compared": {1: "RN3.MASK_POSITION: the returned mask is the conditioning input observed_mask; comparing it "
                        "with itself is vacuous, and the numeric output field cannot hold Boolean values"}}}
# The returned mask (return position 1) is mapped into the reference input as the keyword observed_mask; the measured
# call itself is replayed exactly as recorded, without that keyword.
ATEN_OBSERVED_INPUTS = {"aten.native_dropout": {"argument": OBSERVED_MASK, "return_position": 1}}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed RANDOM profiles")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _floating_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating RANDOM profiles")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)):
                raise Invalid(f"{name} floating values cannot be Boolean literals")
            if isinstance(raw, str):
                spelling = raw.strip().lower()
                if spelling in {"nan", "inf", "+inf", "-inf"}:
                    raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
                try:
                    if spelling.startswith("-") and Fraction(spelling) == 0:
                        raise Unsupported("signed-zero distinctions are outside the reviewed RANDOM profiles")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed RANDOM profiles")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
    if tensor.a.ndim not in (1, 2, 3, 4) or any(size == 0 for size in tensor.shape):
        raise Unsupported(f"{name} must be a positive rank-1 through rank-4 tensor")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _boolean_mask(value, name):
    if not isinstance(value, dict) or value.get("t") not in {"b8", "bool", "torch.bool"}:
        raise Invalid(f"{name} must be a Boolean tensor")
    if "v" in value and any(not isinstance(raw, (bool, np.bool_)) for raw in _raw_leaves(value["v"])):
        raise Invalid(f"{name} must contain only Boolean values")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten) or tensor.dtype != "b8":
        raise Invalid(f"{name} must be a Boolean tensor")
    if tensor.a.ndim not in (1, 2, 3, 4) or any(size == 0 for size in tensor.shape):
        raise Unsupported(f"{name} must be a positive rank-1 through rank-4 tensor")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name in {"input", "grad_output"}:
        return _floating_tensor(value, name)
    if name == "mask":
        return _boolean_mask(value, name)
    return decode(value)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be a real scalar")
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    try:
        result = R(value)
    except Undefined:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    except (TypeError, ValueError):
        raise Invalid(f"{name} must be a real scalar")
    if not result.exact:
        raise Invalid(f"{name} must be an exact scalar")
    if not result.finite:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    return result


def _dropout_values(input, p=0.5, training=True, inplace=False, *, mode="correct"):
    probability = _finite_scalar(p, "p")
    if probability.sign() < 0 or (probability - R(1)).sign() > 0:
        raise Invalid("dropout probability must lie in [0, 1]")
    training = _flag(training, "training")
    if _flag(inplace, "inplace"):
        raise Unsupported("in-place dropout is outside the reviewed RANDOM profile")
    if not training:
        if mode == "eval_scaled":
            factor = R(1) / (R(1) - probability)
            return A.Ten(np.vectorize(lambda value: value * factor, otypes=[object])(input.a), input.dtype)
        return input
    if probability.same(0):
        return input
    if probability.same(1):
        return input if mode == "p1_identity" else A.Ten(np.full(input.shape, R(0), dtype=object), input.dtype)
    raise Unsupported("training dropout with 0 < p < 1 needs a random-law test, not a unique value reference")


def _aten_dropout(input, p, train, *, mode="correct"):
    return _dropout_values(input, p, train, False, mode=mode)


def _functional_dropout(input, p=0.5, training=True, inplace=False, *, mode="correct"):
    return _dropout_values(input, p, training, inplace, mode=mode)


def _dropout_backward(grad_output, mask, scale, *, mode="correct"):
    if grad_output.shape != mask.shape:
        raise Invalid("mask must have the same shape as grad_output")
    scale = _finite_scalar(scale, "scale")
    out = np.empty(grad_output.shape, dtype=object)
    for index in np.ndindex(grad_output.shape):
        grad = grad_output.a[index]
        keep = bool(mask.a[index])
        if mode == "no_scale":
            value = grad if keep else R(0)
        elif mode == "invert_mask":
            value = grad * scale if not keep else R(0)
        elif mode == "add_scale":
            value = grad + scale if keep else R(0)
        else:
            value = grad * scale if keep else R(0)
        out[index] = value
    return A.Ten(out, grad_output.dtype)


FORWARD_OPS = {
    "aten.dropout": _aten_dropout,
    "torch.nn.functional.dropout": _functional_dropout,
}
BACKWARD_OPS = {"aten.native_dropout_backward": _dropout_backward}
_FORWARD_REFERENCE = make_reference(FORWARD_COVERS, extra=FORWARD_OPS,
                                    profiles=("deterministic_forward",), argument_decoder=_decode_argument)
_BACKWARD_REFERENCE = make_reference(BACKWARD_COVERS, extra=BACKWARD_OPS,
                                     profiles=("finite_backward",), argument_decoder=_decode_argument)
_FORWARD_CHECK = make_output_checker(FORWARD_COVERS, extra=FORWARD_OPS,
                                     profiles=("deterministic_forward",), argument_decoder=_decode_argument)
_BACKWARD_CHECK = make_output_checker(BACKWARD_COVERS, extra=BACKWARD_OPS,
                                      profiles=("finite_backward",), argument_decoder=_decode_argument)


# ---------------------------------------------------------------- native_dropout given the returned mask
def _native_input(value):
    """Physical floating input of any positive-size rank; a negative zero is the real value 0 (RN3.CONDITIONED)."""
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"input dtype {dtype!r} is outside the native_forward_given_mask profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)):
                raise Invalid("input floating values cannot be Boolean literals")
            if isinstance(raw, str) and raw.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
                raise Unsupported("input contains a special value outside the native_forward_given_mask profile")
            if isinstance(raw, (float, np.floating)) and not math.isfinite(float(raw)):
                raise Unsupported("input contains a special value outside the native_forward_given_mask profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid("input must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported("input must use f16, bf16, f32 or f64")
    if any(size == 0 for size in tensor.shape):
        raise Unsupported("empty inputs are outside the native_forward_given_mask profile")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid("input values must be exact real numbers")
        if not item.finite:
            raise Unsupported("input contains a special value outside the native_forward_given_mask profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _native_mask(value):
    if not isinstance(value, dict) or value.get("t") not in {"b8", "bool", "torch.bool"}:
        raise Invalid("observed_mask must be a Boolean tensor")
    if "v" in value and any(not isinstance(raw, (bool, np.bool_)) for raw in _raw_leaves(value["v"])):
        raise Invalid("observed_mask must contain only Boolean values")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten) or tensor.dtype != "b8":
        raise Invalid("observed_mask must be a Boolean tensor")
    return tensor


def _decode_native(op, name, value, arguments):
    if name == "input":
        return _native_input(value)
    if name == OBSERVED_MASK:
        return _native_mask(value)
    return decode(value)


def _native_probability(p):
    if isinstance(p, (bool, np.bool_)):
        raise Invalid("p must be a real number")
    if isinstance(p, (float, np.floating)) and math.isnan(float(p)):
        raise Invalid("p must be a real number in [0, 1]")
    try:
        value = p if isinstance(p, R) else R(p)
    except (TypeError, ValueError, Undefined):
        raise Invalid("p must be a real number in [0, 1]") from None
    if not value.exact or not value.finite or value.v < 0 or value.v > 1:
        raise Invalid("p must be a real number in [0, 1]")
    return value.v


def _native_train(train):
    if train is None:
        raise Unsupported("train=None: the decomposition treats it as evaluation, the CPU and CUDA kernels as "
                          "training (RN3.TRAIN_NONE)")
    if not isinstance(train, (bool, np.bool_)):
        raise Invalid("train must be a Boolean or None")
    return bool(train)


def _native_dropout(input, p, train, *, observed_mask=None):
    """[output, None]: output = input * mask / (1 - p) given the returned mask; position 1 is not compared."""
    q = _native_probability(p)
    training = _native_train(train)
    if observed_mask is not None and observed_mask.shape != input.shape:
        raise Invalid("observed_mask must have the shape of input")
    if not training or q == 0:
        return [input, None]
    if q == 1:
        return [A.Ten(np.full(input.shape, R(0), dtype=object), input.dtype), None]
    if observed_mask is None:
        raise Unsupported("training native_dropout with 0 < p < 1 needs the returned mask as observed_mask")
    scale = R(Fraction(1) / (1 - q))
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):
        out[index] = input.a[index] * scale if bool(observed_mask.a[index]) else R(0)
    return [A.Ten(out, input.dtype), None]


_NATIVE_REFERENCE = make_reference(NATIVE_COVERS, extra={"aten.native_dropout": _native_dropout},
                                   profiles=(NATIVE_PROFILE,), argument_decoder=_decode_native)


def reference(profile, inp):
    if profile == NATIVE_PROFILE:
        return _NATIVE_REFERENCE(profile, inp)
    if profile == "deterministic_forward":
        return _FORWARD_REFERENCE(profile, inp)
    if profile == "finite_backward":
        return _BACKWARD_REFERENCE(profile, inp)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def _check_native(profile, inp, observed):
    """observed = {"out": [output record {t, v, shape}, anything]}; position 1 is not compared (RN3.MASK_POSITION)."""
    from kernel_spec.compare import check_enclosed_output
    raw = make_reference(NATIVE_COVERS, extra={"aten.native_dropout": _native_dropout}, profiles=(NATIVE_PROFILE,),
                         encode_output=False, argument_decoder=_decode_native)
    expected = raw(profile, inp).fields["out"][0]
    if not isinstance(observed, dict) or set(observed) != {"out"}:
        return False
    pair = observed["out"]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    record = pair[0]
    if not isinstance(record, dict) or set(record) != {"t", "v", "shape"}:
        return False
    try:
        if A.dt(record["t"]) != expected.dtype or tuple(record["shape"]) != expected.shape:
            return False
        got = decode(record)
    except (Invalid, Unsupported, Undefined):
        return False
    return got.shape == expected.shape and check_enclosed_output(encode(expected), encode(got))


def check_output(profile, inp, observed):
    if profile == NATIVE_PROFILE:
        return _check_native(profile, inp, observed)
    if profile == "deterministic_forward":
        return _FORWARD_CHECK(profile, inp, observed)
    if profile == "finite_backward":
        return _BACKWARD_CHECK(profile, inp, observed)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def _forward_mutant(mode):
    def aten(input, p, train):
        return _aten_dropout(input, p, train, mode=mode)

    def functional(input, p=0.5, training=True, inplace=False):
        return _functional_dropout(input, p, training, inplace, mode=mode)

    return make_reference(FORWARD_COVERS,
                          extra={"aten.dropout": aten, "torch.nn.functional.dropout": functional},
                          profiles=("deterministic_forward",), argument_decoder=_decode_argument)


def _backward_mutant(mode):
    def operation(grad_output, mask, scale):
        return _dropout_backward(grad_output, mask, scale, mode=mode)
    return make_reference(BACKWARD_COVERS, extra={"aten.native_dropout_backward": operation},
                          profiles=("finite_backward",), argument_decoder=_decode_argument)


# Independent wrong implementations of native_dropout given the mask (each a separate function).
def _nm_scale_by_keep(input, p, train, *, observed_mask=None):
    keep = 1 - Fraction(p if not isinstance(p, R) else p.v)
    if not train:
        return [input, None]
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):   # misreading: multiplies by (1 - p) instead of dividing
        out[index] = input.a[index] * R(keep) if observed_mask.a[index] else R(0)
    return [A.Ten(out, input.dtype), None]


def _nm_no_scale(input, p, train, *, observed_mask=None):
    if not train:
        return [input, None]
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):   # misreading: the mask is applied but the 1/(1-p) scaling is dropped
        out[index] = input.a[index] if observed_mask.a[index] else R(0)
    return [A.Ten(out, input.dtype), None]


def _nm_ignore_mask(input, p, train, *, observed_mask=None):
    if not train:
        return [input, None]
    factor = R(1 / (1 - Fraction(p if not isinstance(p, R) else p.v)))
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):   # misreading: scaling applied, mask dropped
        out[index] = input.a[index] * factor
    return [A.Ten(out, input.dtype), None]


def _nm_inverted_mask(input, p, train, *, observed_mask=None):
    if not train:
        return [input, None]
    factor = R(1 / (1 - Fraction(p if not isinstance(p, R) else p.v)))
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):   # misreading: True marks dropped elements
        out[index] = R(0) if observed_mask.a[index] else input.a[index] * factor
    return [A.Ten(out, input.dtype), None]


def _nm_eval_scaled(input, p, train, *, observed_mask=None):
    q = Fraction(p if not isinstance(p, R) else p.v)
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):   # misreading: evaluation mode keeps the 1/(1-p) scaling
        keep = True if (not train or observed_mask is None) else bool(observed_mask.a[index])
        out[index] = input.a[index] * R(1 / (1 - q)) if keep else R(0)
    return [A.Ten(out, input.dtype), None]


def _nm_p1_identity(input, p, train, *, observed_mask=None):
    q = Fraction(p if not isinstance(p, R) else p.v)
    if not train or q in (0, 1):            # misreading: p = 1 returns the input like p = 0
        return [input, None]
    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):
        out[index] = input.a[index] * R(1 / (1 - q)) if observed_mask.a[index] else R(0)
    return [A.Ten(out, input.dtype), None]


def _nm_none_is_eval(input, p, train, *, observed_mask=None):
    if train is None:                       # misreading: train=None is evaluation (decomposition only)
        return [input, None]
    return _native_dropout(input, p, train, observed_mask=observed_mask)


def _native_mutant(fn):
    return make_reference(NATIVE_COVERS, extra={"aten.native_dropout": fn}, profiles=(NATIVE_PROFILE,),
                          argument_decoder=_decode_native)


def _keep_fraction(profile, inp):
    """Diagnostic reading: [fraction of true entries of observed_mask, documented keep probability 1 - p]."""
    if profile != NATIVE_PROFILE or not isinstance(inp, dict) or inp.get("op") != "aten.native_dropout":
        raise Unsupported("keep_fraction reads aten.native_dropout under native_forward_given_mask")
    kwargs = inp.get("kwargs") or {}
    args = inp.get("args") or []
    if OBSERVED_MASK not in kwargs or len(args) < 2:
        raise Unsupported("keep_fraction needs p and observed_mask")
    mask = _native_mask(kwargs[OBSERVED_MASK])
    q = _native_probability(decode(args[1]))
    kept = sum(1 for item in mask.a.flat if item)
    return Result("value", {"out": [Fraction(kept, mask.a.size), 1 - q]})


MUTANTS = {
    "RN3.M1": _native_mutant(_nm_scale_by_keep),
    "RN3.M2": _native_mutant(_nm_no_scale),
    "RN3.M3": _native_mutant(_nm_ignore_mask),
    "RN3.M4": _native_mutant(_nm_inverted_mask),
    "RN3.M5": _native_mutant(_nm_eval_scaled),
    "RN3.M6": _native_mutant(_nm_p1_identity),
    "RN3.M7": _native_mutant(_nm_none_is_eval),
    "RN2.M1": _forward_mutant("eval_scaled"),
    "RN2.M2": _forward_mutant("p1_identity"),
    "RN2.M3": _backward_mutant("no_scale"),
    "RN2.M4": _backward_mutant("invert_mask"),
    "RN2.M5": _backward_mutant("add_scale"),
}
READINGS = {"keep_fraction": _keep_fraction}
