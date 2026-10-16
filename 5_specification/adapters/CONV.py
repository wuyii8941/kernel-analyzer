"""Adapter for contracts/CONV.yaml. Reference: kernel_spec.aten.convolution / convolution_backward; the mutants are
independent direct implementations of the misreadings (1-D only, enough for their witnesses)."""
from __future__ import annotations

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CONV.yaml").read_text())["covers"]
reference = make_reference(COVERS)


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


def _bias_first_position(g, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask=(True, True, True)):
    gi, gw, gb = A.convolution_backward(g, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding, groups)
    g = A._floatify(g)
    return [gi, gw, A.Ten(np.array([g.a[0, co, 0] for co in range(g.shape[1])], dtype=object), g.dtype)]


def _m(**ops):
    return make_reference(COVERS, extra=ops)


MUTANTS = {
    "CV.M1": _m(**{"aten.convolution": _flipped}),
    "CV.M2": _m(**{"aten.convolution": _left_pad}),
    "CV.M3": _m(**{"aten.convolution": _no_dilation}),
    "CV.M4": _m(**{"aten.convolution": _no_groups}),
    "CV.M5": _m(**{"aten.convolution_backward": _bias_first_position}),
}
READINGS = {}
