"""Reference for contracts/POOL.yaml: avg_pool{1,2,3}d, max_pool{1,2,3}d, adaptive_max_pool1d forward values, the
maximiser sets of max pooling, and the uniquely determined part of the VJP against an explicit upstream.

Real-number semantics (kernel_spec.real.R): exact rationals and +-inf. Positions are coordinate tuples within one plane
(channel). Every rule below is a clause or decision of POOL.yaml; nothing here is taken from an implementation.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

from .e2_common import NAN, NAN_REQUIRED, flag, ints, is_nan, nested, plain, tensor
from .num import Invalid, NotEstablished, Undefined, Unsupported, integer
from .real import NEG, R

OPS = {"avg_pool1d": ("avg", 1), "avg_pool2d": ("avg", 2), "avg_pool3d": ("avg", 3), "max_pool1d": ("max", 1),
       "max_pool2d": ("max", 2), "max_pool3d": ("max", 3), "adaptive_max_pool1d": ("adaptive", 1)}
PROFILES = ("real", "nan_prop")


def out_len(L, k, s, p, d, ceil_mode, literal_1d=False):
    """POOL.C2: floor((L + 2p - d(k-1) - 1)/s) + 1, or with ceil instead of floor; in ceil mode the last window is
    dropped when it would start in the right padding ((L_out - 1) s >= L + p). literal_1d: the diagnostic reading of
    the MaxPool1d page's written formula ceil((L + 2p - d(k-1) - 1 + (s - 1))/s) + 1 (decision POOL.CEIL_FORMULA)."""
    num = L + 2 * p - d * (k - 1) - 1
    if not ceil_mode:
        n = num // s + 1
    else:
        n = (math.ceil(Fraction(num + s - 1, s)) if literal_1d else math.ceil(Fraction(num, s))) + 1
        if (n - 1) * s >= L + p:
            n -= 1
    return n


def _params(inp, kind, nd):
    k = ints(inp["kernel"], nd, "kernel", 1)
    s = k if inp.get("stride") in (None, []) else ints(inp["stride"], nd, "stride", 1)
    p = ints(inp.get("padding", 0), nd, "padding", 0)
    d = ints(inp.get("dilation", 1), nd, "dilation", 1) if kind == "max" else (1,) * nd
    if kind == "avg" and "dilation" in inp and inp["dilation"] not in (1, [1], None):
        raise Invalid("average pooling has no dilation parameter")
    ceil_mode = flag(inp.get("ceil_mode", False), "ceil_mode")
    for i in range(nd):
        if 2 * p[i] > d[i] * (k[i] - 1) + 1:              # POOL.C12: pad at most half of the effective kernel size
            raise Invalid(f"padding {p[i]} exceeds half of the effective kernel size {d[i] * (k[i] - 1) + 1}")
    return k, s, p, d, ceil_mode


def _upstream(inp, shape):
    if "upstream" not in inp:
        raise Invalid("upstream (the cotangent of the output) is required")
    u = tensor(inp["upstream"], "upstream")
    if u.shape != shape:
        raise Invalid(f"upstream shape {u.shape} does not match the output shape {shape}")
    for v in u.reshape(-1):
        if not v.finite:
            raise Invalid("upstream must be finite")
    return u


def evaluate(profile, inp, divisor_reading="R2", nan_reading=None, literal_1d=False, min_norm=False):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    op = inp.get("op")
    if op not in OPS:
        raise Unsupported(f"operation {op!r}")
    kind, nd = OPS[op]
    x = tensor(inp["x"], "x", ndim=(nd + 1, nd + 2), allow_nan=True)
    if 0 in x.shape:
        raise Invalid("empty input")
    nan_mode = nan_reading or ("prop" if profile == "nan_prop" else "undefined")
    has_nan = any(is_nan(v) for v in x.reshape(-1))
    if has_nan and (kind == "avg" or nan_mode == "undefined"):
        raise Undefined("NaN in the input: the real-number target is undefined (POOL.A3 / POOL.C10)")
    lead, Ls = x.shape[:-nd], x.shape[-nd:]
    if kind == "adaptive":
        osz = ints(inp["output_size"], 1, "output_size", 1)
        windows = [[tuple(range(o * Ls[0] // osz[0], ((o + 1) * Ls[0] + osz[0] - 1) // osz[0])) for o in range(osz[0])]]
        outs = osz
        P = None
    else:
        k, s, p, d, ceil_mode = _params(inp, kind, nd)
        if kind == "avg" and inp.get("divisor_override") is not None:
            if nd == 1:
                raise Invalid("avg_pool1d has no divisor_override parameter")
            div_o = integer(inp["divisor_override"], "divisor_override")
            if div_o == 0:
                raise Invalid("divisor_override must be nonzero")
        else:
            div_o = None
        cip = flag(inp.get("count_include_pad", True), "count_include_pad") if kind == "avg" else None
        outs = tuple(out_len(Ls[i], k[i], s[i], p[i], d[i], ceil_mode, literal_1d and nd == 1 and kind == "max")
                     for i in range(nd))
        if any(o < 1 for o in outs):
            raise Invalid(f"output size {outs} is too small")
        windows = [[tuple(s[i] * o - p[i] + d[i] * m for m in range(k[i])) for o in range(outs[i])] for i in range(nd)]
        P = [(-p[i], Ls[i] + p[i]) for i in range(nd)]
    oshape = lead + tuple(outs)
    u = _upstream(inp, oshape)
    out = np.empty(oshape, dtype=object)
    arg = np.empty(oshape, dtype=object)
    grad = np.empty(x.shape, dtype=object)
    for i in np.ndindex(x.shape):
        grad[i] = R(0)
    for plane in np.ndindex(*lead):
        for o in itertools.product(*[range(n) for n in outs]):
            cells = list(itertools.product(*[windows[i][o[i]] for i in range(nd)]))
            real = [c for c in cells if all(0 <= c[i] < Ls[i] for i in range(nd))]
            v = u[plane + o].v
            if kind == "avg":
                total = R(0)
                for c in real:
                    total = total + x[plane + c]
                if div_o is not None:
                    div = div_o
                elif not cip:
                    div = len(real)
                elif divisor_reading == "R1":
                    div = math.prod(k)
                else:
                    div = sum(1 for c in cells if all(P[i][0] <= c[i] < P[i][1] for i in range(nd)))
                if div == 0:
                    raise Undefined("window without any counted element")
                out[plane + o] = total / div
                for c in real:
                    grad[plane + c] = grad[plane + c] + R(v) / div
                continue
            vals = [(c, x[plane + c]) for c in real]
            nans = [c for c, val in vals if is_nan(val)]
            if nans and nan_mode == "prop":
                out[plane + o] = NAN
                arg[plane + o] = frozenset(nans)
                continue                                   # NaN windows route nothing determined (POOL.A3)
            vals = [(c, val) for c, val in vals if not is_nan(val)]
            if not vals:
                if nans:
                    raise NotEstablished("all real values of a window are NaN under R_ignore")
                out[plane + o] = R(NEG)                    # empty window: only -inf padding (POOL.A5)
                arg[plane + o] = frozenset()
                continue
            best = vals[0][1]
            for _, val in vals[1:]:
                if val > best:
                    best = val
            S = frozenset(c for c, val in vals if val.same(best))
            out[plane + o] = best
            arg[plane + o] = S
            if len(S) == 1:
                (c,) = S
                grad[plane + c] = grad[plane + c] + R(v)
            elif min_norm:
                for c in S:
                    grad[plane + c] = grad[plane + c] + R(v) / len(S)
    fields = {"out": nested(out), "grad": nested(grad)}
    fields["argmax"] = [] if kind == "avg" else nested(arg, lambda a: a)
    return fields
