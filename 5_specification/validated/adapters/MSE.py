"""Reviewed PyTorch 2.10 unweighted same-shape float32 MSE and complete VJP."""
from __future__ import annotations

from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS


COVERS = yaml.safe_load((CONTRACTS / "MSE.yaml").read_text())["covers"]
PROFILE = ("torch210_mean_f32_same_shape",)
FULL_PROFILE = "torch210_unweighted_f32_same_shape"


def _physical_f32_tensor(value, name):
    if not isinstance(value, dict) or "t" not in value or "v" not in value:
        raise Invalid(f"{name} must be a physical tensor record")
    try:
        dtype = A.dt(value["t"])
    except Invalid:
        raise Unsupported(f"{name} dtype is outside the reviewed MSE profile")
    if dtype != "f32":
        raise Unsupported(f"{name} dtype {dtype!r} is outside the reviewed float32 MSE profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    for item in tensor.a.flat:
        if not isinstance(item, R):
            raise Invalid(f"{name} is not a real tensor")
        if not item.finite:
            raise Unsupported(f"{name} contains infinity outside the finite MSE profile")
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
            raise Invalid(f"mean MSE upstream must be scalar, got shape {tensor.shape}")
        return tensor
    return decode(value)


OPS = {"aten.mse_loss": A.mse_loss, "vjp.aten.mse_loss": A.vjp_mse_loss}
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
        raise Invalid(f"mse_loss requires equal shapes, got {a.shape} and {target.shape}")
    if a.a.size == 0:
        raise Invalid("the reviewed mse_loss profile requires at least one element")
    if reduction != "mean":
        raise Unsupported("the reviewed mse_loss profile fixes reduction='mean'")
    return a, target


def _base_difference(a, target, reduction="mean"):
    a, target = _inputs(a, target, reduction)
    return a, target, A.sub(a, target)


def m_mean_by_first_dimension(a, target, reduction="mean"):
    a, target, difference = _base_difference(a, target, reduction)
    divisor = a.shape[0] if a.shape else 1
    return A.mul(A.sum_(A.mul(difference, difference)), Fraction(1, divisor))


def m_sum_instead_of_mean(a, target, reduction="mean"):
    _, _, difference = _base_difference(a, target, reduction)
    return A.sum_(A.mul(difference, difference))


def _vjp_parts(grad, a, target, reduction="mean", factor=2, use_upstream=True):
    a, target, difference = _base_difference(a, target, reduction)
    grad = A.as_ten(grad)
    if grad.shape != ():
        raise Invalid("mean MSE upstream must be scalar")
    scale = Fraction(factor, a.a.size)
    dx = A.mul(difference, scale)
    if use_upstream:
        dx = A.mul(dx, grad)
    return dx, A.neg(dx)


def m_vjp_missing_factor_two(grad, a, target, reduction="mean"):
    return list(_vjp_parts(grad, a, target, reduction, factor=1))


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
    "MSE.M1": _mut(**{"aten.mse_loss": m_mean_by_first_dimension}),
    "MSE.M2": _mut(**{"aten.mse_loss": m_sum_instead_of_mean}),
    "MSE.M3": _mut(**{"vjp.aten.mse_loss": m_vjp_missing_factor_two}),
    "MSE.M4": _mut(**{"vjp.aten.mse_loss": m_vjp_target_same_sign}),
    "MSE.M5": _mut(**{"vjp.aten.mse_loss": m_vjp_ignore_upstream}),
    "MSE.M6": _mut(**{"vjp.aten.mse_loss": m_vjp_drop_target}),
}
READINGS = {}


# Reduction errors have witnesses in the new profile, independently of mean-only mutants.
def _wrong_none(a, target, reduction="mean"):
    return A.mse_loss(a, target, "sum")


def _wrong_sum(a, target, reduction="mean"):
    return A.mse_loss(a, target, "mean")


MUTANTS.update({
    "MSE.M7": make_reference(COVERS, extra={**OPS, "aten.mse_loss": _wrong_none},
        profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument),
    "MSE.M8": make_reference(COVERS, extra={**OPS, "aten.mse_loss": _wrong_sum},
        profiles=(FULL_PROFILE,), argument_decoder=_decode_full_argument),
})
