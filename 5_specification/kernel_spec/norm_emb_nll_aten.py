"""ATen-call semantics for the NORM, EMB and NLL contracts (package norm_emb_nll, 2026-10-20).

Functions take the arguments of the PyTorch 2.10 ATen schemas (decoded by kernel_spec.aten_adapter.decode, tensors as
kernel_spec.aten.Ten) and return Ten results (tuples as lists in ATen return order, None for absent positions).

- aten.native_layer_norm / aten.native_layer_norm_backward / aten.rms_norm: contract NORM, profile aten_native
  (decisions NM.ATEN_STATS, NM.ATEN_BWD_SUPPLIED, NM.ATEN_RMS_EPS_NONE, NM.ATEN_DTYPES).
- aten.embedding_backward / aten.embedding_dense_backward: contract EMB, profile aten_dense_backward
  (decisions EM.ATEN_PADDING, EM.ATEN_BACKWARD).
- aten.nll_loss_forward / aten.nll_loss_backward: contract NLL, profile aten_real (decisions NL.*).

Values are exact rationals; 1/sqrt(.) is exact when rational and a rigorous enclosure otherwise (kernel_spec.real.R).
Nothing here rounds. Non-finite tensor values are outside the profiles (Unsupported); NaN inputs are Invalid
(rejected by the decoder) except where the profile states the values are not read.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import aten as A
from . import real as rl
from .num import Invalid, Undefined, Unsupported, integer, rational
from .real import R

FLOAT_DTYPES = ("f16", "bf16", "f32", "f64")
# torch.finfo(dtype).eps for the dtypes where the documented default (torch.finfo(x.dtype).eps) and the PyTorch 2.10
# native default (epsilon of the computation type) coincide (decision NM.ATEN_RMS_EPS_NONE).
RMS_DEFAULT_EPS = {"f32": Fraction(1, 2 ** 23), "f64": Fraction(1, 2 ** 52)}


# ------------------------------------------------------------------ argument checks
def float_values(t, name, *, allow_ignored=False):
    """(flat list of Fractions, shape, dtype) of a finite floating tensor argument."""
    if not isinstance(t, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if t.dtype not in FLOAT_DTYPES:
        if t.dtype in A.FLOAT:
            raise Unsupported(f"{name} dtype {t.dtype} is outside the profile (f16, bf16, f32, f64)")
        raise Invalid(f"{name} must be a floating tensor, got {t.dtype}")
    if allow_ignored:
        return None, tuple(t.shape), t.dtype
    out = []
    for v in t.a.flat:
        if isinstance(v, float):                     # retained NaN storage cannot reach here without allow_ignored
            raise Invalid(f"{name} contains NaN")
        if not v.finite:
            raise Unsupported(f"{name} contains an infinite value, outside the finite profile")
        out.append(v.v)
    return out, tuple(t.shape), t.dtype


def int_values(t, name, dtypes=("i64",)):
    if not isinstance(t, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if t.dtype not in A.INT:
        raise Invalid(f"{name} must be an integer tensor, got {t.dtype}")
    if t.dtype not in dtypes:
        raise Unsupported(f"{name} dtype {t.dtype} is outside the profile ({', '.join(dtypes)})")
    return [int(v) for v in t.a.flat], tuple(t.shape)


def int_arg(x, name, lo=None):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise Invalid(f"{name} must be an integer")
    return integer(int(x), name, lo=lo)


def bool_arg(x, name):
    if not isinstance(x, (bool, np.bool_)):
        raise Invalid(f"{name} must be a boolean")
    return bool(x)


def real_scalar(x, name):
    if isinstance(x, R):
        if not x.finite or not x.exact:
            raise Invalid(f"{name} must be a finite exact number")
        return x.v
    if isinstance(x, float) and not math.isfinite(x):
        raise Invalid(f"{name} must be finite")
    return rational(x, name)


def shape_list(x, name):
    if not isinstance(x, (list, tuple)):
        raise Invalid(f"{name} must be a list of integers")
    return [int_arg(v, f"{name} element", lo=0) for v in x]


def prod(xs):
    p = 1
    for v in xs:
        p *= v
    return p


def ten(values, shape, dtype):
    """Ten of R values (values: flat list of Fraction / R) in the given shape."""
    arr = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        arr[i] = v if isinstance(v, R) else R(v)
    return A.Ten(arr.reshape(tuple(shape)), dtype)


# ------------------------------------------------------------------ NORM
def _layer_norm_geometry(shape, normalized_shape, weight, bias):
    ns = shape_list(normalized_shape, "normalized_shape")
    D = len(ns)
    if D < 1:
        raise Invalid("normalized_shape must contain at least one element")
    if len(shape) < D or tuple(shape[len(shape) - D:]) != tuple(ns):
        raise Invalid(f"normalized_shape {ns} is not the trailing shape of input {shape}")
    w = b = None
    if weight is not None:
        w, wshape, _ = float_values(weight, "weight")
        if wshape != tuple(ns):
            raise Invalid("weight must have shape normalized_shape")
    if bias is not None:
        b, bshape, _ = float_values(bias, "bias")
        if bshape != tuple(ns):
            raise Invalid("bias must have shape normalized_shape")
    outer = tuple(shape[:len(shape) - D])
    return ns, outer, prod(outer), prod(ns), w, b


def _rstd(radicand):
    if radicand <= 0:
        raise Undefined("variance + eps <= 0: the reciprocal square root is undefined over the reals")
    return R(1) / rl.sqrt(R(radicand))


def native_layer_norm(input, normalized_shape, weight, bias, eps):
    """(out, mean, rstd): biased variance, rstd = 1/sqrt(var + eps); mean and rstd of shape outer + [1]*D."""
    x, shape, dtype = float_values(input, "input")
    ns, outer, M, N, w, b = _layer_norm_geometry(shape, normalized_shape, weight, bias)
    e = real_scalar(eps, "eps")
    if N == 0 and M > 0:
        raise Undefined("mean of an empty normalisation set")
    out, means, rstds = [None] * len(x), [], []
    for m in range(M):
        row = x[m * N:(m + 1) * N]
        mu = sum(row, Fraction(0)) / N
        var = sum(((v - mu) ** 2 for v in row), Fraction(0)) / N
        r = _rstd(var + e)
        for j, v in enumerate(row):
            y = R(v - mu) * r
            if w is not None:
                y = y * w[j]
            if b is not None:
                y = y + b[j]
            out[m * N + j] = y
        means.append(mu)
        rstds.append(r)
    stat_shape = outer + (1,) * len(ns)
    stat_dtype = dtype                    # metadata only, decision NM.ATEN_DTYPES
    return [ten(out, shape, dtype), ten(means, stat_shape, stat_dtype), ten(rstds, stat_shape, stat_dtype)]


def native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask):
    """(grad_input, grad_weight, grad_bias) from the supplied mean and rstd (decision NM.ATEN_BWD_SUPPLIED)."""
    x, shape, dtype = float_values(input, "input")
    g, gshape, gdtype = float_values(grad_out, "grad_out")
    if gshape != shape:
        raise Invalid("grad_out must have the input's shape")
    ns, outer, M, N, w, b = _layer_norm_geometry(shape, normalized_shape, weight, bias)
    mu, _, _ = float_values(mean, "mean")
    rs, _, _ = float_values(rstd, "rstd")
    if len(mu) != M or len(rs) != M:
        raise Invalid(f"mean and rstd must hold one value per normalisation set ({M})")
    if not isinstance(output_mask, (list, tuple)) or len(output_mask) != 3:
        raise Invalid("output_mask must be a list of three booleans")
    mask = [bool_arg(v, "output_mask element") for v in output_mask]
    if mask[1] and weight is None:
        raise Unsupported("output_mask[1] without weight: the result position is not specified")
    if mask[2] and bias is None:
        raise Unsupported("output_mask[2] without bias: the result position is not specified")
    dx = [None] * len(x)
    dw = [Fraction(0)] * N
    db = [Fraction(0)] * N
    for m in range(M):
        lo = m * N
        r = rs[m]
        xh = [(x[lo + j] - mu[m]) * r for j in range(N)]
        h = [g[lo + j] * (w[j] if w is not None else 1) for j in range(N)]
        sh = sum(h, Fraction(0))
        shx = sum((a * c for a, c in zip(h, xh)), Fraction(0))
        for j in range(N):
            dx[lo + j] = r / N * (N * h[j] - sh - xh[j] * shx)
            dw[j] += g[lo + j] * xh[j]
            db[j] += g[lo + j]
    return [ten(dx, shape, dtype) if mask[0] else None,
            ten(dw, ns, weight.dtype) if mask[1] else None,
            ten(db, ns, bias.dtype) if mask[2] else None]


def rms_norm(input, normalized_shape, weight=None, eps=None):
    """y = x / sqrt(mean(x^2) + eps) * weight over the trailing normalized_shape dims."""
    x, shape, dtype = float_values(input, "input")
    ns, outer, M, N, w, _ = _layer_norm_geometry(shape, normalized_shape, weight, None)
    if eps is None:
        if dtype not in RMS_DEFAULT_EPS:
            raise Unsupported(f"rms_norm eps=None for {dtype}: documentation (torch.finfo(x.dtype).eps) and the "
                              "PyTorch 2.10 native default (float32 epsilon) differ; decision NM.ATEN_RMS_EPS_NONE")
        e = RMS_DEFAULT_EPS[dtype]
    else:
        e = real_scalar(eps, "eps")
    if N == 0 and M > 0:
        raise Undefined("mean of an empty normalisation set")
    out = [None] * len(x)
    for m in range(M):
        row = x[m * N:(m + 1) * N]
        r = _rstd(sum((v * v for v in row), Fraction(0)) / N + e)
        for j, v in enumerate(row):
            y = R(v) * r
            out[m * N + j] = y * w[j] if w is not None else y
    return ten(out, shape, dtype)


# ------------------------------------------------------------------ EMB
def embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    """grad_weight[k] = sum of grad rows at positions with index k (k != padding_idx, literal comparison),
    each divided by the occurrence count of k when scale_grad_by_freq."""
    g, gshape, dtype = float_values(grad_output, "grad_output")
    ix, ishape = int_values(indices, "indices", ("i64", "i32"))
    V = int_arg(num_weights, "num_weights", lo=0)
    pad = int_arg(padding_idx, "padding_idx")
    freq = bool_arg(scale_grad_by_freq, "scale_grad_by_freq")
    if len(gshape) != len(ishape) + 1 or gshape[:-1] != ishape:
        raise Invalid("grad must have the indices' shape followed by the embedding dimension")
    Dm = gshape[-1]
    if any(not 0 <= k < V for k in ix):
        raise Invalid("embedding backward: index outside [0, num_weights)")
    counts = {}
    for k in ix:
        counts[k] = counts.get(k, 0) + 1
    out = [Fraction(0)] * (V * Dm)
    for p, k in enumerate(ix):
        if k == pad:
            continue
        c = counts[k] if freq else 1
        for d in range(Dm):
            out[k * Dm + d] += g[p * Dm + d] / c
    return ten(out, (V, Dm), dtype)


def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    """The composite: sparse=False is embedding_dense_backward; sparse=True (a sparse COO result) is outside."""
    if bool_arg(sparse, "sparse"):
        raise Unsupported("embedding_backward with sparse=True returns a sparse gradient layout, outside the profile")
    return embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq)


# ------------------------------------------------------------------ NLL
REDUCTIONS = {0: "none", 1: "mean", 2: "sum"}


def _nll_inputs(self, target, weight, reduction, ignore_index, *, values=True):
    x, shape, dtype = float_values(self, "self", allow_ignored=not values)
    if len(shape) not in (1, 2):
        raise Invalid("nll_loss_forward/backward: self must be 1D or 2D")
    t, tshape = int_values(target, "target")
    if len(tshape) > 1:
        raise Invalid("target must be 0D or 1D")
    if len(shape) == 2:
        if len(tshape) != 1 or tshape[0] != shape[0]:
            raise Invalid("target must have shape (N) for self of shape (N, C)")
    elif tshape not in ((), (1,)):
        raise Invalid("for 1D self, target must be 0D or of size 1")
    C = shape[-1]
    w = None
    if weight is not None:
        w, wshape, _ = float_values(weight, "weight")
        if wshape != (C,):
            raise Invalid("weight must be 1D with one value per class")
    red = int_arg(reduction, "reduction")
    if red not in REDUCTIONS:
        raise Invalid(f"reduction {red} is not one of 0 (none), 1 (mean), 2 (sum)")
    ign = int_arg(ignore_index, "ignore_index")
    for v in t:
        if v != ign and not 0 <= v < C:
            raise Invalid(f"target {v} is out of bounds for {C} classes")
    return x, shape, dtype, t, C, w, red, ign


def nll_loss_forward(self, target, weight, reduction, ignore_index):
    """(output, total_weight); decisions NL.TOTAL_WEIGHT, NL.EMPTY_MEAN."""
    x, shape, dtype, t, C, w, red, ign = _nll_inputs(self, target, weight, reduction, ignore_index)
    losses, tw = [], Fraction(0)
    for n, c in enumerate(t):
        if c == ign:
            losses.append(Fraction(0))
            continue
        wc = w[c] if w is not None else Fraction(1)
        losses.append(-wc * x[n * C + c])
        tw += wc
    if red == 0 and len(shape) == 2:
        return [ten(losses, (shape[0],), dtype), ten([Fraction(0)], (), dtype)]
    s = sum(losses, Fraction(0))
    if red == 1:
        if tw == 0:
            raise Undefined("mean reduction with total weight 0 (every target ignored or weights summing to 0)")
        s = s / tw
    return [ten([s], (), dtype), ten([tw], (), dtype)]


def nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight):
    """grad_input: -weight[t] * grad_output (divided by the supplied total_weight for mean) at each non-ignored
    target position, 0 elsewhere (decisions NL.BWD_SUPPLIED, NL.BWD_EMPTY)."""
    _, shape, dtype, t, C, w, red, ign = _nll_inputs(self, target, weight, reduction, ignore_index, values=False)
    go, goshape, _ = float_values(grad_output, "grad_output")
    tws, twshape, _ = float_values(total_weight, "total_weight")
    if len(tws) != 1:
        raise Invalid("total_weight must have exactly one element")
    per_sample = red == 0 and len(shape) == 2
    if per_sample:
        if goshape != (shape[0],):
            raise Invalid("grad_output must have shape (N) for reduction none")
    elif len(goshape) > 1 or len(go) != 1:
        raise Invalid("grad_output must be a single element")
    rows = shape[0] if len(shape) == 2 else 1
    out = [Fraction(0)] * (rows * C)
    for n, c in enumerate(t):
        if c == ign:
            continue
        gval = go[n] if per_sample else go[0]
        if red == 1:
            if tws[0] == 0:
                raise Undefined("mean reduction backward with supplied total_weight 0 and a non-ignored target")
            gval = gval / tws[0]
        out[n * C + c] = -(w[c] if w is not None else 1) * gval
    return ten(out, shape, dtype)


OPS = {
    "aten.native_layer_norm": native_layer_norm,
    "aten.native_layer_norm_backward": native_layer_norm_backward,
    "aten.rms_norm": rms_norm,
    "aten.embedding_backward": embedding_backward,
    "aten.embedding_dense_backward": embedding_dense_backward,
    "aten.nll_loss_forward": nll_loss_forward,
    "aten.nll_loss_backward": nll_loss_backward,
}


def decode_argument(op, name, value, arguments):
    """Contract-specific decoding: nll_loss_backward does not read the values of self (only its shape and dtype),
    so retained NaN storage there is allowed; everything else uses the shared decoder."""
    from .aten_adapter import decode
    if isinstance(value, dict) and ("t" in value or "v" in value):
        return decode(value, allow_ignored_nan=(op == "aten.nll_loss_backward" and name == "self"))
    return decode(value)
