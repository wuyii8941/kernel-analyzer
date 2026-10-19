"""Adapter for contracts/CONV.yaml. Reference: kernel_spec.aten.convolution / convolution_backward; the mutants are
independent direct implementations of the misreadings (1-D only, enough for their witnesses)."""
from __future__ import annotations

import math
import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CONV.yaml").read_text())["covers"]
FINITE_FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _decode_argument(op, name, value, arguments):
    decoded = decode(value)
    if not isinstance(decoded, A.Ten):
        return decoded
    if decoded.dtype not in FINITE_FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {decoded.dtype!r} is outside the finite floating CONV profile")
    for item in decoded.a.flat:
        finite = item.finite if isinstance(item, R) else (math.isfinite(item) if isinstance(item, float) else True)
        if not finite:
            if isinstance(item, float) and math.isnan(item):
                raise Invalid(f"{name} contains NaN")
            raise Unsupported(f"{name} contains an infinite value outside the finite CONV profile")
    if name in {"g", "x", "w"}:
        if decoded.a.ndim == 5:
            raise Unsupported("3-D convolution is outside the reviewed CONV profile")
        if decoded.a.ndim not in (3, 4):
            raise Invalid("the reviewed low-level convolution profile requires batched 1-D or 2-D tensors")
    return decoded


reference = make_reference(COVERS, profiles=("real",), argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, profiles=("real",), argument_decoder=_decode_argument)


def _conv1d(x, w, bias, stride, pad_left, pad_right, dilation, flip=False, groups=1, ignore_groups=False):
    x, w = A._floatify(x), A._floatify(w)
    N, Cin, L = x.shape
    Cout, cpg, K = w.shape
    xs = [[[R(0)] * pad_left + list(x.a[n, c]) + [R(0)] * pad_right for c in range(Cin)] for n in range(N)]
    Lp = L + pad_left + pad_right
    Lout = (Lp - dilation * (K - 1) - 1) // stride + 1
    out = np.empty((N, Cout, Lout), dtype=object)
    for n in range(N):
        for co in range(Cout):
            for o in range(Lout):
                s = R(0) if bias is None else A._floatify(bias).a[co]
                chans = range(Cin) if ignore_groups else range((co // (Cout // groups)) * cpg, (co // (Cout // groups)) * cpg + cpg)
                for c in chans:
                    for k in range(K):
                        kk = K - 1 - k if flip else k
                        s = s + xs[n][c][o * stride + k * dilation] * w.a[co, 0 if ignore_groups else c - min(chans), kk]
                out[n, co, o] = s
    return A.Ten(out, x.dtype)


def _flipped(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    return _conv1d(x, w, bias, stride[0], padding[0], padding[0], dilation[0], flip=True, groups=groups)


def _left_pad(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    return _conv1d(x, w, bias, stride[0], 2 * padding[0], 0, dilation[0], groups=groups)


def _no_dilation(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    return _conv1d(x, w, bias, stride[0], padding[0], padding[0], 1, groups=groups)


def _no_groups(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    return _conv1d(x, w, bias, stride[0], padding[0], padding[0], dilation[0], groups=1, ignore_groups=True)


def _bias_first_position(g, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    gi, gw, gb = A.convolution_backward(
        g, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask)
    g = A._floatify(g)
    return [gi, gw, A.Ten(np.array([g.a[0, co, 0] for co in range(g.shape[1])], dtype=object), g.dtype)]


def _m(**ops):
    return make_reference(COVERS, extra=ops, profiles=("real",), argument_decoder=_decode_argument)


MUTANTS = {
    "CV.M1": _m(**{"aten.convolution": _flipped}),
    "CV.M2": _m(**{"aten.convolution": _left_pad}),
    "CV.M3": _m(**{"aten.convolution": _no_dilation}),
    "CV.M4": _m(**{"aten.convolution": _no_groups}),
    "CV.M5": _m(**{"aten.convolution_backward": _bias_first_position}),
}
READINGS = {}
