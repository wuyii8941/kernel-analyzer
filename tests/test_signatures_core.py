"""Per-signature evidence for the core arith / math / tt elementwise operations (DSL v2 rc3 04 W1): positive, boundary
and premise-violation cases on compiled-only kernels, checked against exact values (tests/signature_harness.py)."""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402

INF, NAN = float("inf"), float("nan")
FMAX, FSUB = float(np.finfo(np.float32).max), float(np.nextafter(np.float32(0), np.float32(1)))
I32MIN, I32MAX = -2 ** 31, 2 ** 31 - 1


def m(x):
    return mp.mpf(float(x))


def real(f):
    def g(*a):
        if any(isinstance(x, mp.mpf) and mp.isnan(x) for x in a):
            return mp.nan
        try:
            v = f(*a)
        except (ValueError, ZeroDivisionError):
            return None
        return v
    return g


def ext_div(a, b):
    if b == 0:
        return mp.nan if a == 0 else (mp.inf if (a > 0) == (mp.sign(b) >= 0) else -mp.inf)
    return a / b


def c_trunc(q):
    return mp.floor(q) if q >= 0 else mp.ceil(q)


def wrap32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def iop(f):
    """integer reference on Python ints; None = undefined (poison / division by zero)."""
    def g(*a):
        a = [int(x) for x in a]
        try:
            r = f(*a)
        except ZeroDivisionError:
            return None
        return r
    return g


def c_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


# name: (expr, in dtypes, out dtype, reference, is_int, (positive, boundary, premise) input columns)
FLOAT2 = ([1.0, -2.5, 7.0, 0.375], [3.0, 0.75, -2.0, 1.5])
SPEC = {
    "arith.addf": ("a0 + a1", ("fp32", "fp32"), "fp32", real(lambda x, y: x + y), False,
                   (FLOAT2, ([0.0, FMAX, FSUB, -0.0], [-0.0, FMAX, -FSUB, INF]), ([NAN, INF], [1.0, -INF]))),
    "arith.subf": ("a0 - a1", ("fp32", "fp32"), "fp32", real(lambda x, y: x - y), False,
                   (FLOAT2, ([0.0, FMAX, FSUB], [0.0, -FMAX, FSUB]), ([NAN, INF], [1.0, INF]))),
    "arith.mulf": ("a0 * a1", ("fp32", "fp32"), "fp32", real(lambda x, y: x * y), False,
                   (FLOAT2, ([0.0, FMAX, FSUB, -0.0], [5.0, FMAX, FSUB, 3.0]), ([NAN, INF], [1.0, 0.0]))),
    "arith.divf": ("a0 / a1", ("fp32", "fp32"), "fp32", real(ext_div), False,
                   (FLOAT2, ([0.0, FMAX, FSUB], [3.0, FSUB, FMAX]), ([1.0, 0.0, NAN], [0.0, 0.0, 1.0]))),
    "tt.precise_divf": ("tl.div_rn(a0, a1)", ("fp32", "fp32"), "fp32", real(ext_div), False,
                        (FLOAT2, ([0.0, FMAX], [3.0, FSUB]), ([1.0, 0.0], [0.0, 0.0]))),
    "arith.remf": ("a0 % a1", ("fp32", "fp32"), "fp32", real(lambda x, y: None if y == 0 else x - y * c_trunc(x / y)),
                   False, (FLOAT2, ([0.0, FSUB], [3.0, 1.0]), ([1.0], [0.0]))),
    "arith.negf": ("-a0", ("fp32",), "fp32", real(lambda x: -x), False,
                   (([1.0, -2.5, 0.375],), ([0.0, -0.0, FMAX, INF],), ([NAN],))),
    "math.absf": ("tl.abs(a0)", ("fp32",), "fp32", real(abs), False,
                  (([1.0, -2.5, 0.375],), ([0.0, -0.0, -FMAX, -INF],), ([NAN],))),
    "math.floor": ("tl.floor(a0)", ("fp32",), "fp32", real(mp.floor), False,
                   (([1.5, -2.5, 0.375],), ([0.0, -0.0, FMAX, -INF],), ([NAN],))),
    "math.ceil": ("tl.ceil(a0)", ("fp32",), "fp32", real(mp.ceil), False,
                  (([1.5, -2.5, 0.375],), ([0.0, -0.0, FMAX, INF],), ([NAN],))),
    "math.sqrt": ("tl.sqrt(a0)", ("fp32",), "fp32", real(lambda x: None if x < 0 else mp.sqrt(x)), False,
                  (([0.25, 2.0, 9.0],), ([0.0, -0.0, FSUB, FMAX, INF],), ([-1.0, -0.5],))),
    "tt.precise_sqrt": ("tl.sqrt_rn(a0)", ("fp32",), "fp32", real(lambda x: None if x < 0 else mp.sqrt(x)), False,
                        (([0.25, 2.0, 9.0],), ([0.0, FSUB, FMAX, INF],), ([-1.0],))),
    "math.rsqrt": ("tl.rsqrt(a0)", ("fp32",), "fp32",
                   real(lambda x: mp.inf if x == 0 else (None if x < 0 else 1 / mp.sqrt(x))), False,
                   (([0.25, 2.0, 9.0],), ([FSUB, FMAX, INF],), ([-1.0],))),
    "math.exp": ("tl.exp(a0)", ("fp32",), "fp32", real(mp.exp), False,
                 (([0.5, -1.25, 3.0],), ([0.0, FMAX, -FMAX, INF, -INF],), ([NAN],))),
    "math.exp2": ("tl.exp2(a0)", ("fp32",), "fp32", real(lambda x: mp.power(2, x)), False,
                  (([0.5, -1.25, 3.0],), ([0.0, 200.0, -200.0, INF, -INF],), ([NAN],))),
    "math.log": ("tl.log(a0)", ("fp32",), "fp32",
                 real(lambda x: -mp.inf if x == 0 else (None if x < 0 else mp.log(x))), False,
                 (([0.5, 2.0, 10.0],), ([FSUB, 1.0, FMAX, INF, 0.0],), ([-1.0, -INF],))),
    "math.log2": ("tl.log2(a0)", ("fp32",), "fp32",
                  real(lambda x: -mp.inf if x == 0 else (None if x < 0 else mp.log(x, 2))), False,
                  (([0.5, 2.0, 10.0],), ([FSUB, 1.0, FMAX, INF],), ([-1.0],))),
    "math.sin": ("tl.sin(a0)", ("fp32",), "fp32", real(lambda x: None if mp.isinf(x) else mp.sin(x)), False,
                 (([0.5, -1.25, 3.0],), ([0.0, -0.0, 1e4],), ([INF, NAN],))),
    "math.cos": ("tl.cos(a0)", ("fp32",), "fp32", real(lambda x: None if mp.isinf(x) else mp.cos(x)), False,
                 (([0.5, -1.25, 3.0],), ([0.0, -0.0, 1e4],), ([INF, NAN],))),
    "math.erf": ("tl.erf(a0)", ("fp32",), "fp32", real(mp.erf), False,
                 (([0.5, -1.25, 3.0],), ([0.0, -0.0, FMAX, INF, -INF],), ([NAN],))),
    "math.fma": ("tl.fma(a0, a1, a2)", ("fp32", "fp32", "fp32"), "fp32", real(lambda x, y, z: x * y + z), False,
                 (([1.0, -2.5], [3.0, 0.75], [-2.0, 4.0]), ([0.0, FMAX], [5.0, 2.0], [-0.0, -FMAX]),
                  ([NAN, INF], [1.0, 0.0], [2.0, 1.0]))),
    "arith.maxnumf": ("tl.maximum(a0, a1)", ("fp32", "fp32"), "fp32",
                      lambda x, y: y if mp.isnan(x) else (x if mp.isnan(y) else max(x, y)), False,
                      (FLOAT2, ([0.0, FMAX, -INF], [-0.0, INF, -FMAX]), ([NAN, 1.0], [2.0, NAN]))),
    "arith.minnumf": ("tl.minimum(a0, a1)", ("fp32", "fp32"), "fp32",
                      lambda x, y: y if mp.isnan(x) else (x if mp.isnan(y) else min(x, y)), False,
                      (FLOAT2, ([0.0, FMAX, -INF], [-0.0, INF, -FMAX]), ([NAN, 1.0], [2.0, NAN]))),
    "arith.maximumf": ("tl.maximum(a0, a1, propagate_nan=tl.PropagateNan.ALL)", ("fp32", "fp32"), "fp32",
                       real(lambda x, y: max(x, y)), False,
                       (FLOAT2, ([0.0, FMAX, -INF], [-0.0, INF, -FMAX]), ([NAN, 1.0], [2.0, NAN]))),
    "arith.minimumf": ("tl.minimum(a0, a1, propagate_nan=tl.PropagateNan.ALL)", ("fp32", "fp32"), "fp32",
                       real(lambda x, y: min(x, y)), False,
                       (FLOAT2, ([0.0, FMAX, -INF], [-0.0, INF, -FMAX]), ([NAN, 1.0], [2.0, NAN]))),
    # ordered IEEE comparison: a NaN operand makes a0 < a1 false, so the select takes a1
    "arith.cmpf+select": ("tl.where(a0 < a1, a0, a1)", ("fp32", "fp32"), "fp32",
                          lambda x, y: y if (mp.isnan(x) or mp.isnan(y)) else (x if x < y else y),
                          False, (FLOAT2, ([0.0, FMAX], [-0.0, -FMAX]), ([NAN], [1.0]))),
    "arith.sitofp": ("a0.to(tl.float32)", ("int32",), "fp32", iop(lambda x: x), False,
                     (([3, -7, 100],), ([0, I32MIN, I32MAX],), ([0],))),
    "arith.fptosi": ("a0.to(tl.int32)", ("fp32",), "int32", real(lambda x: None if mp.isinf(x) or not
                                                                  (I32MIN <= c_trunc(x) <= I32MAX) else int(c_trunc(x))),
                     True, (([3.75, -7.25, 100.0],), ([0.0, -0.0, 2147483520.0, -2147483648.0],), ([3e9, -3e9],))),
    "arith.extf": ("a0.to(tl.float32)", ("fp16",), "fp32", real(lambda x: x), False,
                   (([1.5, -2.25, 0.125],), ([0.0, 65504.0, float(np.float16(6e-8))],), ([NAN],))),
    "arith.truncf": ("a0.to(tl.float16)", ("fp32",), "fp16", real(lambda x: x), False,
                     (([1.5, -2.25, 0.125],), ([0.0, 65504.0, 1e5],), ([NAN],))),
    "arith.bitcast": ("a0.to(tl.int32, bitcast=True)", ("fp32",), "int32",
                      lambda x: int(np.asarray([float(x)], np.float32).view(np.int32)[0]), True,
                      (([1.5, -2.25, 0.125],), ([0.0, -0.0, FMAX, INF],), ([FSUB],))),
    "arith.addi": ("a0 + a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(x + y)), True,
                   (([3, -7, 100], [5, 2, -300]), ([I32MAX, I32MIN, 0], [1, -1, 0]), ([I32MAX], [I32MAX]))),
    "arith.subi": ("a0 - a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(x - y)), True,
                   (([3, -7, 100], [5, 2, -300]), ([I32MIN, I32MAX], [1, -1]), ([I32MIN], [I32MAX]))),
    "arith.muli": ("a0 * a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(x * y)), True,
                   (([3, -7, 100], [5, 2, -300]), ([I32MAX, I32MIN, 0], [2, -1, 5]), ([I32MAX], [I32MAX]))),
    "arith.divsi": ("a0 // a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(c_div(x, y))), True,
                    (([7, -7, 100], [2, 2, -3]), ([0, I32MIN, I32MAX], [5, 1, -1]), ([5, 3], [0, 0]))),
    "arith.remsi": ("a0 % a1", ("int32", "int32"), "int32", iop(lambda x, y: x - y * c_div(x, y)), True,
                    (([7, -7, 100], [2, 2, -3]), ([0, I32MAX], [5, 7]), ([5], [0]))),
    "arith.andi": ("a0 & a1", ("int32", "int32"), "int32", iop(lambda x, y: x & y), True,
                   (([12, -7, 100], [10, 3, -1]), ([0, I32MIN], [I32MAX, -1]), ([0], [0]))),
    "arith.ori": ("a0 | a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(x | y)), True,
                  (([12, -7, 100], [10, 3, -1]), ([0, I32MIN], [I32MAX, -1]), ([0], [0]))),
    "arith.xori": ("a0 ^ a1", ("int32", "int32"), "int32", iop(lambda x, y: wrap32(x ^ y)), True,
                   (([12, -7, 100], [10, 3, -1]), ([0, I32MIN], [I32MAX, -1]), ([0], [0]))),
    "arith.shli": ("a0 << a1", ("int32", "int32"), "int32", iop(lambda x, y: None if not 0 <= y < 32 else wrap32(x << y)),
                   True, (([3, -7, 1], [2, 1, 30]), ([1, I32MAX], [31, 1]), ([1, 1], [32, -1]))),
    "arith.shrsi": ("a0 >> a1", ("int32", "int32"), "int32", iop(lambda x, y: None if not 0 <= y < 32 else x >> y),
                    True, (([12, -7, 100], [2, 1, 3]), ([I32MIN, -1], [31, 31]), ([1, 1], [32, -1]))),
    "arith.maxsi": ("tl.maximum(a0, a1)", ("int32", "int32"), "int32", iop(max), True,
                    (([12, -7, 100], [10, 3, -1]), ([I32MIN, I32MAX], [I32MAX, I32MIN]), ([0], [0]))),
    "arith.minsi": ("tl.minimum(a0, a1)", ("int32", "int32"), "int32", iop(min), True,
                    (([12, -7, 100], [10, 3, -1]), ([I32MIN, I32MAX], [I32MAX, I32MIN]), ([0], [0]))),
    "arith.cmpi+select": ("tl.where(a0 < a1, a0, a1)", ("int32", "int32"), "int32", iop(min), True,
                          (([12, -7, 100], [10, 3, -1]), ([I32MIN, I32MAX], [I32MAX, I32MIN]), ([0], [0]))),
    "math.absi": ("tl.abs(a0)", ("int32",), "int32", iop(lambda x: wrap32(abs(x))), True,
                  (([12, -7, 100],), ([0, I32MAX, I32MIN + 1],), ([I32MIN],))),
    "arith.extsi": ("a0.to(tl.int64)", ("int32",), "int64", iop(lambda x: x), True,
                    (([12, -7, 100],), ([0, I32MAX, I32MIN],), ([0],))),
    "arith.trunci": ("a0.to(tl.int32)", ("int64",), "int32", iop(wrap32), True,
                     (([12, -7, 100],), ([2 ** 31, -2 ** 31 - 1, 2 ** 40 + 5],), ([0],))),
}


def _cases():
    return [pytest.param(name, cat, id=f"{name}-{cat}") for name in SPEC for cat in ("positive", "boundary",
                                                                                      "premise_violation")]


@pytest.mark.parametrize("name,category", _cases())
def test_core_signature(name, category):
    expr, dts, out, ref, is_int, cols_by_cat = SPEC[name]
    cols = cols_by_cat[{"positive": 0, "boundary": 1, "premise_violation": 2}[category]]
    cols = [H.pad(c) for c in cols]
    inputs = [np.asarray(c, H.NP[d]) for c, d in zip(cols, dts)]
    lo, hi, st, _ = H.evaluate(name.replace(".", "_").replace("+", "_"), expr, dts, out, inputs)
    conv = (lambda v: int(v)) if any(d.startswith(("int", "uint")) for d in dts) else m
    if name == "arith.bitcast":
        conv = float  # keep the sign of zero (mpmath drops it)
    exact = [ref(*[conv(x[j]) for x in inputs]) for j in range(H.N)]
    exact = [(int(e) if is_int and e is not None and not (isinstance(e, mp.mpf) and (mp.isnan(e) or mp.isinf(e)))
              else e) for e in exact]
    bad = H.check_lanes(category, lo, hi, st, exact, is_int=is_int)
    assert not bad, f"{name} {category}: {bad[:4]}"
