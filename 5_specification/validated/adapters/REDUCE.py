"""Adapter for the reviewed finite-floating REDUCE profile."""
from __future__ import annotations

import math
import numbers

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "REDUCE.yaml").read_text())["covers"]
FINITE_FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _sum(a, dim=None, keepdim=False, *, dtype=None):
    return A.sum_(a, dim, _bool(keepdim, "keepdim"), dtype=dtype)


def _mean(a, dim=None, keepdim=False, *, dtype=None):
    return A.mean(a, dim, _bool(keepdim, "keepdim"), dtype=dtype)


def _prod(a, dim=None, keepdim=False, *, dtype=None):
    if dim is not None and (isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))):
        raise Invalid("prod dim must be one integer or None")
    return A.prod(a, dim, _bool(keepdim, "keepdim"), dtype=dtype)


def _amax(a, dim=None, keepdim=False):
    return A.amax(a, dim, _bool(keepdim, "keepdim"))


def _amin(a, dim=None, keepdim=False):
    return A.amin(a, dim, _bool(keepdim, "keepdim"))


def _argmax(a, dim=None, keepdim=False):
    return A.argmax(a, dim, _bool(keepdim, "keepdim"))


def _argmin(a, dim=None, keepdim=False):
    return A.argmin(a, dim, _bool(keepdim, "keepdim"))


def _max(a, dim, keepdim=False):
    return A.max_dim(a, dim, _bool(keepdim, "keepdim"))


def _min(a, dim, keepdim=False):
    return A.min_dim(a, dim, _bool(keepdim, "keepdim"))


def _cumsum(a, dim, *, dtype=None):
    return A.cumsum(a, dim, dtype=dtype)


def _logsumexp(a, dim, keepdim=False):
    if dim is None or (isinstance(dim, (list, tuple)) and not dim):
        raise Invalid("logsumexp requires one or more reduction dimensions")
    return A.logsumexp(a, dim, _bool(keepdim, "keepdim"))


def _variance_dim(dim):
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise Unsupported("empty-dimension variance spelling is outside the selected profile")
        values = dim
    elif dim is None:
        return None
    else:
        values = [dim]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in values):
        raise Invalid("variance dimensions must be integers")
    return dim


def _variance_correction(value):
    if value is None:
        return 1
    if isinstance(value, (bool, np.bool_)):
        raise Invalid("variance correction must be numeric, not boolean")
    if not isinstance(value, (int, np.integer)):
        if isinstance(value, numbers.Number):
            raise Unsupported("only variance correction 0 or 1 is selected")
        raise Invalid("variance correction must be numeric")
    value = int(value)
    if value not in (0, 1):
        raise Unsupported("only variance correction 0 or 1 is selected")
    return value


def _var(a, dim=None, *, correction=None, keepdim=False):
    return A.var(a, _variance_dim(dim), _variance_correction(correction),
                 _bool(keepdim, "keepdim"))


def _std(a, dim=None, *, correction=None, keepdim=False):
    return A.std(a, _variance_dim(dim), _variance_correction(correction),
                 _bool(keepdim, "keepdim"))


def _var_mean(a, dim=None, *, correction=None, keepdim=False):
    return A.var_mean(a, _variance_dim(dim), _variance_correction(correction),
                      _bool(keepdim, "keepdim"))


def _vjp_sum(g, x, dim=None, keepdim=False):
    return A.vjp_sum(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_mean(g, x, dim=None, keepdim=False):
    return A.vjp_mean(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_prod(g, x, dim=None, keepdim=False):
    if dim is not None and (isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))):
        raise Invalid("prod VJP dim must be one integer or None")
    return A.vjp_prod(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_amax(g, x, dim=None, keepdim=False):
    return A.vjp_amax(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_amin(g, x, dim=None, keepdim=False):
    return A.vjp_amin(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_max(g, x, dim, keepdim=False):
    return A.vjp_max_dim(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_min(g, x, dim, keepdim=False):
    return A.vjp_min_dim(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_logsumexp(g, x, dim, keepdim=False):
    if dim is None or (isinstance(dim, (list, tuple)) and not dim):
        raise Invalid("logsumexp VJP requires one or more reduction dimensions")
    return A.vjp_logsumexp(g, x, dim, _bool(keepdim, "keepdim"))


def _vjp_cumsum(g, x, dim):
    return A.vjp_cumsum(g, x, dim)


def _vjp_var(g, x, dim=None, *, correction=None, keepdim=False):
    return A.vjp_var(g, x, _variance_dim(dim), _variance_correction(correction),
                     _bool(keepdim, "keepdim"))


def _vjp_std(g, x, dim=None, *, correction=None, keepdim=False):
    x = A.as_ten(x)
    dim = _variance_dim(dim)
    correction = _variance_correction(correction)
    keepdim = _bool(keepdim, "keepdim")
    variance = A.var(x, dim, correction, True)
    reduced_count = 1
    for axis in A._axes(x, dim):
        reduced_count *= x.shape[axis]
    if reduced_count > 1 and any(value.v == 0 for value in variance.a.flat):
        raise Undefined("standard-deviation VJP is not uniquely defined at a zero-variance slice")
    return A.vjp_std(g, x, dim, correction, keepdim)


def _vjp_var_mean(g_var, g_mean, x, dim=None, *, correction=None, keepdim=False):
    dim = _variance_dim(dim)
    correction = _variance_correction(correction)
    keepdim = _bool(keepdim, "keepdim")
    return A.vjp_var_mean(g_var, g_mean, x, dim, correction, keepdim)


def _vector_norm_dim(dim):
    if dim is None:
        return None
    values = dim if isinstance(dim, (list, tuple)) else [dim]
    if not values:
        raise Unsupported("empty vector-norm dimension sequence is outside the selected profile")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) for value in values):
        raise Invalid("vector-norm dimensions must be integers")
    return dim


def _vector_norm_order(value):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid("vector-norm order must be numeric, not boolean")
    if not isinstance(value, numbers.Number):
        raise Invalid("vector-norm order must be numeric")
    if value != 2:
        raise Unsupported("only the vector 2-norm is selected")
    return 2


def _vector_norm_dtype(value):
    if value is not None:
        raise Unsupported("explicit vector-norm dtype conversion is outside the selected profile")


def _vector_norm_input(value):
    value = A.as_ten(value)
    if value.dtype not in {"f32", "f64"}:
        raise Unsupported("the selected public vector-norm profile supports real float or double input")
    return value


def _linalg_vector_norm(a, ord=2, dim=None, keepdim=False, *, dtype=None):
    _vector_norm_dtype(dtype)
    return A.linalg_vector_norm(
        _vector_norm_input(a), _vector_norm_order(ord), _vector_norm_dim(dim),
        _bool(keepdim, "keepdim"),
    )


def _vjp_linalg_vector_norm(g, x, ord=2, dim=None, keepdim=False, *, dtype=None):
    _vector_norm_dtype(dtype)
    dim = _vector_norm_dim(dim)
    ord = _vector_norm_order(ord)
    keepdim = _bool(keepdim, "keepdim")
    x = _vector_norm_input(x)
    norms = A.linalg_vector_norm(x, ord, dim, True)
    if any(value.v == 0 for value in norms.a.flat):
        raise Undefined("the real vector 2-norm VJP is not unique at a zero-norm slice")
    return A.vjp_linalg_vector_norm(g, x, ord, dim, keepdim)


REVIEWED_OPS = {
    "aten.sum": _sum, "aten.mean": _mean, "aten.prod": _prod,
    "aten.amax": _amax, "aten.amin": _amin, "aten.argmax": _argmax,
    "aten.argmin": _argmin, "aten.max": _max, "aten.min": _min,
    "aten.cumsum": _cumsum, "aten.logsumexp": _logsumexp, "aten.var": _var,
    "aten.std": _std, "aten.var_mean": _var_mean,
    "aten.linalg_vector_norm": _linalg_vector_norm,
    "vjp.aten.sum": _vjp_sum, "vjp.aten.mean": _vjp_mean,
    "vjp.aten.prod": _vjp_prod, "vjp.aten.amax": _vjp_amax,
    "vjp.aten.amin": _vjp_amin, "vjp.aten.max": _vjp_max,
    "vjp.aten.min": _vjp_min, "vjp.aten.logsumexp": _vjp_logsumexp,
    "vjp.aten.cumsum": _vjp_cumsum, "vjp.aten.var": _vjp_var,
    "vjp.aten.std": _vjp_std, "vjp.aten.var_mean": _vjp_var_mean,
    "vjp.aten.linalg_vector_norm": _vjp_linalg_vector_norm,
}


def _raw_dtype(value):
    if isinstance(value, dict) and "t" in value:
        try:
            return A.dt(value["t"])
        except Invalid:
            return None
    return None


def _finite_decode(op, name, value, arguments, *, match_vjp=True):
    if name == "dtype":
        decoded = decode(value)
        if decoded is None:
            return None
        try:
            dtype = A.dt(decoded)
        except Invalid:
            raise Unsupported(f"dtype {decoded!r} is outside the finite floating REDUCE profile")
        if dtype not in FINITE_FLOAT_DTYPES:
            raise Unsupported(f"dtype {dtype!r} is outside the finite floating REDUCE profile")
        return dtype
    if isinstance(value, dict) and ("t" in value or "v" in value):
        raw_dtype = value.get("t")
        try:
            dtype = A.dt(raw_dtype)
        except Invalid:
            if "complex" in str(raw_dtype).lower():
                raise Unsupported("complex tensors are outside the finite floating REDUCE profile")
            raise
        if dtype not in FINITE_FLOAT_DTYPES:
            raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating REDUCE profile")
    decoded = decode(value)
    if isinstance(decoded, A.Ten):
        for item in decoded.a.flat:
            finite = item.finite if isinstance(item, R) else (math.isfinite(item) if isinstance(item, float) else True)
            if not finite:
                if isinstance(item, float) and math.isnan(item):
                    raise Invalid(f"{name} contains NaN")
                raise Unsupported(f"{name} contains infinity outside the finite REDUCE profile")
        if match_vjp and op.startswith("vjp.") and name in {"g", "g_var", "g_mean", "x"}:
            xd = _raw_dtype(arguments.get("x"))
            upstream_dtypes = [
                _raw_dtype(arguments.get(key))
                for key in ("g", "g_var", "g_mean")
                if key in arguments
            ]
            if xd is not None and any(dtype is not None and dtype != xd
                                      for dtype in upstream_dtypes):
                raise Invalid("reduction VJP upstream and input dtypes must match")
    return decoded


def _decode_argument(op, name, value, arguments):
    return _finite_decode(op, name, value, arguments)


reference = make_reference(COVERS, extra=REVIEWED_OPS, profiles=("real",), argument_decoder=_decode_argument)
check_output = make_output_checker(
    COVERS, extra=REVIEWED_OPS, profiles=("real",), argument_decoder=_decode_argument)


def _argmax_last(a, dim=None, keepdim=False):
    t = A.as_ten(a)
    rows = t.a.reshape(-1, t.a.shape[-1])
    out = []
    for row in rows:
        best, k = row[0], 0
        for i, value in enumerate(row):
            if value >= best:
                best, k = value, i
        out.append(k)
    return A.tensor(out, "i64")


def _amax_grad_first(g, x, dim=None, keepdim=False):
    x = A.as_ten(x)
    out = np.full(x.shape, R(0), dtype=object)
    gv = list(A.as_ten(g).a.reshape(-1))
    for r, row in enumerate(x.a.reshape(-1, x.a.shape[-1])):
        k = max(range(len(row)), key=lambda i: (row[i].v, -i))
        out.reshape(-1, x.a.shape[-1])[r, k] = gv[r]
    return A.Ten(out, x.dtype)


def _max_grad_even(g, x, dim, keepdim=False):
    return A.vjp_amax(g, x, [dim], keepdim)


def _prod_grad_div(g, x, dim=None, keepdim=False):
    x = A.as_ten(x)
    out = np.empty(x.shape, dtype=object)
    gv = list(A.as_ten(g).a.reshape(-1))
    for r, row in enumerate(x.a.reshape(-1, x.a.shape[-1])):
        product = R(1)
        for value in row:
            product = product * value
        for i, value in enumerate(row):
            out.reshape(-1, x.a.shape[-1])[r, i] = gv[r] * product / value
    return A.Ten(out, x.dtype)


def _mean_empty_zero(a, dim=None, keepdim=False, *, dtype=None):
    t = A.as_ten(a)
    if t.a.size == 0:
        shape = [s for i, s in enumerate(t.shape) if i not in A._axes(t, dim)]
        return A.full(shape, 0, t.dtype)
    return A.mean(a, dim, keepdim, dtype=dtype)


def _cumsum_grad_forward(g, x, dim):
    return A.cumsum(A._floatify(g), dim)


def _lse_without_log(a, dim, keepdim=False):
    return A.sum_(A.exp(a), dim, keepdim)


def _var_ignore_correction(a, dim=None, *, correction=None, keepdim=False):
    return A.var(a, _variance_dim(dim), 0, _bool(keepdim, "keepdim"))


def _vjp_var_missing_two(g, x, dim=None, *, correction=None, keepdim=False):
    result = A.vjp_var(g, x, _variance_dim(dim), _variance_correction(correction),
                       _bool(keepdim, "keepdim"))
    return A.div(result, 2)


def _std_without_sqrt(a, dim=None, *, correction=None, keepdim=False):
    return A.var(a, _variance_dim(dim), _variance_correction(correction),
                 _bool(keepdim, "keepdim"))


def _std_ignore_correction(a, dim=None, *, correction=None, keepdim=False):
    return A.std(a, _variance_dim(dim), 1, _bool(keepdim, "keepdim"))


def _vjp_std_without_sd(g, x, dim=None, *, correction=None, keepdim=False):
    result = A.vjp_var(g, x, _variance_dim(dim), _variance_correction(correction),
                       _bool(keepdim, "keepdim"))
    return A.div(result, 2)


def _var_mean_reversed(a, dim=None, *, correction=None, keepdim=False):
    variance, mean_value = _var_mean(
        a, dim, correction=correction, keepdim=keepdim
    )
    return mean_value, variance


def _vjp_var_mean_without_mean(g_var, g_mean, x, dim=None, *, correction=None,
                               keepdim=False):
    return A.vjp_var(g_var, x, _variance_dim(dim), _variance_correction(correction),
                     _bool(keepdim, "keepdim"))


def _vjp_var_mean_without_variance(g_var, g_mean, x, dim=None, *, correction=None,
                                   keepdim=False):
    return A.vjp_mean(g_mean, x, _variance_dim(dim),
                      _bool(keepdim, "keepdim"))


def _vector_norm_squared(a, ord=2, dim=None, keepdim=False, *, dtype=None):
    _vector_norm_dtype(dtype)
    a = A._floatify(a)
    return A._reduce(
        a, _vector_norm_dim(dim), _bool(keepdim, "keepdim"),
        lambda values: A._rsum([value * value for value in values], R(0)),
    )


def _vjp_vector_norm_without_norm(g, x, ord=2, dim=None, keepdim=False, *, dtype=None):
    _vector_norm_dtype(dtype)
    gb = A._bcast_back(A._floatify(g), A._floatify(x), _vector_norm_dim(dim),
                       _bool(keepdim, "keepdim"))
    return A.mul(gb, A._floatify(x))


def _m(**ops):
    table = dict(REVIEWED_OPS)
    table.update(ops)
    return make_reference(COVERS, extra=table, profiles=("real",), argument_decoder=_decode_argument)


MUTANTS = {
    "RD.M1": _m(**{"aten.argmax": _argmax_last}),
    "RD.M2": _m(**{"vjp.aten.amax": _amax_grad_first}),
    "RD.M3": _m(**{"vjp.aten.max": _max_grad_even}),
    "RD.M4": _m(**{"vjp.aten.prod": _prod_grad_div}),
    "RD.M5": _m(**{"aten.mean": _mean_empty_zero}),
    "RD.M6": _m(**{"vjp.aten.cumsum": _cumsum_grad_forward}),
    "RD.M7": _m(**{"aten.logsumexp": _lse_without_log}),
    "RD.M8": make_reference(COVERS, extra=REVIEWED_OPS, profiles=("real",)),
    "RD.M9": make_reference(
        COVERS, extra={**REVIEWED_OPS, "aten.prod": A.prod}, profiles=("real",),
        argument_decoder=_decode_argument),
    "RD.M10": make_reference(
        COVERS, extra=REVIEWED_OPS, profiles=("real",),
        argument_decoder=lambda op, name, value, arguments: _finite_decode(
            op, name, value, arguments, match_vjp=False)),
    "RD.M11": _m(**{"aten.var": _var_ignore_correction}),
    "RD.M12": _m(**{"vjp.aten.var": _vjp_var_missing_two}),
    "RD.M13": _m(**{"aten.std": _std_without_sqrt}),
    "RD.M14": _m(**{"vjp.aten.std": _vjp_std_without_sd}),
    "RD.M15": _m(**{"aten.std": _std_ignore_correction}),
    "RD.M16": _m(**{"aten.var_mean": _var_mean_reversed}),
    "RD.M17": _m(**{"vjp.aten.var_mean": _vjp_var_mean_without_mean}),
    "RD.M18": _m(**{"vjp.aten.var_mean": _vjp_var_mean_without_variance}),
    "RD.M19": _m(**{"aten.linalg_vector_norm": _vector_norm_squared}),
    "RD.M20": _m(**{"vjp.aten.linalg_vector_norm": _vjp_vector_norm_without_norm}),
}
READINGS = {}
