"""Adapter for contracts/MATMUL.yaml. Reference: kernel_spec.aten; mutants are independent implementations of the
misreadings."""
from __future__ import annotations

import yaml
import math

from kernel_spec import aten as A
from kernel_spec.aten_adapter import decode, make_reference, make_output_checker
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "MATMUL.yaml").read_text())["covers"]
FINITE_FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _finite_tensor(value, name):
    if value.dtype not in FINITE_FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {value.dtype!r} is outside the finite floating MATMUL profile")
    for item in value.a.flat:
        if isinstance(item, R):
            finite = item.finite
        elif isinstance(item, float):
            finite = math.isfinite(item)
        else:
            finite = True
        if not finite:
            if isinstance(item, float) and math.isnan(item):
                raise Invalid(f"{name} contains NaN")
            raise Unsupported(f"{name} contains an infinite value outside the finite MATMUL profile")
    return value


def _decode_argument(op, name, value, arguments):
    ignored = False
    if op in ("aten.addmm", "aten.baddbmm", "vjp.aten.addmm") and name == "c":
        ignored = A._product_scalar(decode(arguments.get("beta", 1)), "beta") == 0
    if isinstance(value, dict) and ("t" in value or "v" in value) and name not in ("alpha", "beta"):
        raw_dtype = value.get("t")
        try:
            dtype = A.dt(raw_dtype)
        except Invalid:
            if "complex" in str(raw_dtype).lower():
                raise Unsupported("complex tensor operands are outside the finite floating MATMUL profile")
            raise
        if dtype not in FINITE_FLOAT_DTYPES:
            raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating MATMUL profile")
    decoded = decode(value, allow_ignored_nan=ignored)
    if isinstance(decoded, A.Ten) and name not in ("alpha", "beta"):
        if ignored:
            if decoded.dtype not in FINITE_FLOAT_DTYPES:
                raise Unsupported(f"{name} dtype {decoded.dtype!r} is outside the finite floating MATMUL profile")
        else:
            _finite_tensor(decoded, name)
    return decoded


# Explicit default ATen schemas from retained PyTorch 2.10 native_functions.yaml.
# VJP names are our mathematical interface, not ATen schemas, and are unchanged.
ARGUMENT_ALIASES = {
    "aten.mm": {"self": "a", "mat2": "b"},
    "aten.bmm": {"self": "a", "mat2": "b"},
    "aten.matmul": {"self": "a", "other": "b"},
    "aten.addmm": {"self": "c", "mat1": "a", "mat2": "b"},
    "aten.baddbmm": {"self": "c", "batch1": "a", "batch2": "b"},
    "aten.linear": {"input": "x", "weight": "w", "bias": "b"},
}

reference = make_reference(COVERS, profiles=("real",), argument_decoder=_decode_argument,
                           argument_aliases=ARGUMENT_ALIASES)
check_output = make_output_checker(COVERS, profiles=("real",), argument_decoder=_decode_argument,
                                  argument_aliases=ARGUMENT_ALIASES)


def _linear_no_t(x, w, b=None):
    y = A.matmul(x, w)
    return y if b is None else A.add(y, b)


def _addmm_beta0_adds(c, a, b, beta=1, alpha=1):
    return A.add(A.mul(c, 1 if beta == 0 else beta), A.mul(A.mm(a, b), alpha))


def _vjp_matmul_no_reduce(g, a, b):
    return [A.matmul(g, A.transpose(b, -1, -2)), A.matmul(A.transpose(a, -1, -2), g)]


def _vjp_mm_wrong(g, a, b):
    return [A.mm(g, A.t(b)), A.mm(g, a)]


def _vjp_linear_bias_rows(g, x, w, b=None):
    dx, dw, db = A.vjp_linear(g, x, w, b)
    return [dx, dw, A.as_ten(g)]


def _addmm_no_alpha(c, a, b, beta=1, alpha=1):
    return A.addmm(c, a, b, beta=beta, alpha=1)


def _unrestricted_decode(op, name, value, arguments):
    ignored = op in ("aten.addmm", "aten.baddbmm", "vjp.aten.addmm") and name == "c" and \
        A._product_scalar(decode(arguments.get("beta", 1)), "beta") == 0
    return decode(value, allow_ignored_nan=ignored)


def _addmm_positional_beta(c, a, b, beta=1, alpha=1):
    return A.addmm(c, a, b, beta=beta, alpha=alpha)


def _addmm_ignores_out_dtype(c, a, b, out_dtype=None, *, beta=1, alpha=1):
    return A.addmm(c, a, b, beta=beta, alpha=alpha)


def _m(**ops):
    return make_reference(COVERS, extra=ops, profiles=("real",), argument_decoder=_decode_argument,
                          argument_aliases=ARGUMENT_ALIASES)


MUTANTS = {
    "MM.M1": _m(**{"aten.linear": _linear_no_t}),
    "MM.M2": _m(**{"aten.addmm": _addmm_beta0_adds}),
    "MM.M3": _m(**{"vjp.aten.matmul": _vjp_matmul_no_reduce}),
    "MM.M4": _m(**{"vjp.aten.mm": _vjp_mm_wrong}),
    "MM.M5": _m(**{"vjp.aten.linear": _vjp_linear_bias_rows}),
    "MM.M6": _m(**{"aten.addmm": _addmm_no_alpha}),
    "MM.M7": make_reference(COVERS, profiles=("real",), argument_decoder=_unrestricted_decode,
                            argument_aliases=ARGUMENT_ALIASES),
    "MM.M8": _m(**{"aten.addmm": _addmm_positional_beta}),
    "MM.M9": _m(**{"aten.addmm": _addmm_ignores_out_dtype}),
}
READINGS = {}
