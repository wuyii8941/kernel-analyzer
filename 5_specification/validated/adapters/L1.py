"""Reviewed PyTorch 2.10 unweighted same-shape float32 L1 and complete VJP."""
from __future__ import annotations

from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS


COVERS = yaml.safe_load((CONTRACTS / "L1.yaml").read_text())["covers"]
PROFILE = ("torch210_mean_f32_same_shape",)
FULL_PROFILE = "torch210_unweighted_f32_same_shape"


def _physical_f32_tensor(value, name):
    if not isinstance(value, dict) or "t" not in value or "v" not in value:
        raise Invalid(f"{name} must be a physical tensor record")
    try:
        dtype = A.dt(value["t"])
    except Invalid:
        raise Unsupported(f"{name} dtype is outside the reviewed L1 profile")
    if dtype != "f32":
        raise Unsupported(f"{name} dtype {dtype!r} is outside the reviewed float32 L1 profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    for item in tensor.a.flat:
        if not isinstance(item, R):
            raise Invalid(f"{name} is not a real tensor")
        if not item.finite:
            raise Unsupported(f"{name} contains infinity outside the finite L1 profile")
        if not item.exact:
            raise Unsupported(f"{name} contains a non-point input")
        rounded = ieee.round_to(item, "f32")
        if not rounded.exact or rounded.v != item.v:
            raise Invalid(f"{name} contains a value not exactly representable as float32")
    return tensor


def _decode_argument(op, name, value, raw):
    if name == "reduction" and value != "mean":
        raise Unsupported("the original mean profile requires reduction=mean")
    if name in {"a", "target", "grad"}:
        tensor = _physical_f32_tensor(value, name)
        if name == "grad" and tensor.shape != ():
            raise Invalid(f"mean L1 upstream must be scalar, got shape {tensor.shape}")
        return tensor
    return decode(value)


OPS = {"aten.l1_loss": A.l1_loss, "vjp.aten.l1_loss": A.vjp_l1_loss}
def _decode_full_argument(op, name, value, raw):
    if name in {"a", "target", "grad"}:
        return _physical_f32_tensor(value, name)
    return decode(value)


_mean_reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
_full_reference = make_reference(COVERS, extra=OPS, profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument)
_mean_check = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
_full_check = make_output_checker(COVERS, extra=OPS, profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument)


def reference(profile, inp):
    return (_full_reference if profile == FULL_PROFILE else _mean_reference)(profile, inp)


def check_output(profile, inp, observed):
    return (_full_check if profile == FULL_PROFILE else _mean_check)(profile, inp, observed)


def _inputs(a, target, reduction="mean"):
    a, target = A.as_ten(a), A.as_ten(target)
    if a.shape != target.shape:
        raise Invalid(f"l1_loss requires equal shapes, got {a.shape} and {target.shape}")
    if a.a.size == 0:
        raise Invalid("the reviewed l1_loss profile requires at least one element")
    if reduction != "mean":
        raise Unsupported("the reviewed l1_loss profile fixes reduction='mean'")
    return a, target


def m_mean_by_first_dimension(a, target, reduction="mean"):
    a, target = _inputs(a, target, reduction)
    divisor = a.shape[0] if a.shape else 1
    return A.mul(A.sum_(A.abs_(A.sub(a, target))), Fraction(1, divisor))


def m_signed_difference(a, target, reduction="mean"):
    a, target = _inputs(a, target, reduction)
    return A.mean(A.sub(a, target))


def _vjp_parts(grad, a, target, reduction="mean", *, zero_positive=False, use_upstream=True):
    a, target = _inputs(a, target, reduction)
    grad = A.as_ten(grad)
    if grad.shape != ():
        raise Invalid("mean L1 upstream must be scalar")
    difference = A.sub(a, target)
    direction = A.sign(difference)
    if zero_positive:
        direction = A.where(A.eq(difference, 0), 1, direction)
    dx = A.mul(direction, Fraction(1, a.a.size))
    if use_upstream:
        dx = A.mul(dx, grad)
    return dx, A.neg(dx)


def m_vjp_zero_positive(grad, a, target, reduction="mean"):
    return list(_vjp_parts(grad, a, target, reduction, zero_positive=True))


def m_vjp_target_same_sign(grad, a, target, reduction="mean"):
    dx, _ = _vjp_parts(grad, a, target, reduction)
    return [dx, dx]


def m_vjp_ignore_upstream(grad, a, target, reduction="mean"):
    return list(_vjp_parts(grad, a, target, reduction, use_upstream=False))


def m_vjp_drop_target(grad, a, target, reduction="mean"):
    dx, _ = _vjp_parts(grad, a, target, reduction)
    return [dx]


def _mut(**ops):
    return make_reference(COVERS, extra={**OPS, **ops}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


MUTANTS = {
    "L1.M1": _mut(**{"aten.l1_loss": m_mean_by_first_dimension}),
    "L1.M2": _mut(**{"aten.l1_loss": m_signed_difference}),
    "L1.M3": _mut(**{"vjp.aten.l1_loss": m_vjp_zero_positive}),
    "L1.M4": _mut(**{"vjp.aten.l1_loss": m_vjp_target_same_sign}),
    "L1.M5": _mut(**{"vjp.aten.l1_loss": m_vjp_ignore_upstream}),
    "L1.M6": _mut(**{"vjp.aten.l1_loss": m_vjp_drop_target}),
}
READINGS = {}


# Reduction errors have witnesses in the new profile, independently of mean-only mutants.
def _wrong_none(a, target, reduction="mean"):
    return A.l1_loss(a, target, "sum")


def _wrong_sum(a, target, reduction="mean"):
    return A.l1_loss(a, target, "mean")


MUTANTS.update({
    "L1.M7": make_reference(COVERS, extra={**OPS, "aten.l1_loss": _wrong_none},
        profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument),
    "L1.M8": make_reference(COVERS, extra={**OPS, "aten.l1_loss": _wrong_sum},
        profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument),
})
