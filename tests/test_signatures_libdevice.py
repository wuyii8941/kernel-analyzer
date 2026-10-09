"""Per-signature evidence for every registered libdevice symbol (DSL v2 rc3 04 W1): positive, boundary and premise
violation cases on compiled-only kernels, checked against mpmath (tests/signature_harness.py)."""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from kernel_analyzer.reference_eval import ttir_mapping as M  # noqa: E402

F32 = np.finfo(np.float32)
F64 = np.finfo(np.float64)


def _sub(dt):
    return float(np.nextafter(np.zeros(1, H.NP[dt]), np.ones(1, H.NP[dt]))[0])


def _max(dt):
    return float(F32.max if dt == "fp32" else F64.max)


def _m(x):
    return mp.mpf(float(x))


def _ext_div(a, b):
    if b == 0:
        return mp.nan if a == 0 or mp.isnan(a) else (mp.inf if (a > 0) == (mp.sign(b) >= 0) else -mp.inf)
    return a / b


def _real(f):
    def g(*a):
        if any(mp.isnan(x) for x in a):
            return mp.nan
        try:
            v = f(*a)
        except (ValueError, ZeroDivisionError):
            return None
        return None if isinstance(v, mp.mpc) and v.imag != 0 else v
    return g


def _log(base=None):
    def f(x):
        if x == 0:
            return -mp.inf
        if x < 0:
            return None
        if mp.isinf(x):
            return mp.inf
        return mp.log(x) if base is None else mp.log(x, base)
    return _real(f)


REF = {
    "exp": _real(lambda x: mp.exp(x)), "exp2": _real(lambda x: mp.power(2, x)), "exp10": _real(lambda x: mp.power(10, x)),
    "expm1": _real(lambda x: mp.expm1(x)), "log": _log(), "log2": _log(2), "log10": _log(10),
    "log1p": _real(lambda x: -mp.inf if x == -1 else (None if x < -1 else mp.log1p(x))),
    "sqrt": _real(lambda x: None if x < 0 else mp.sqrt(x)),
    "rsqrt": _real(lambda x: mp.inf if x == 0 else (None if x < 0 else 1 / mp.sqrt(x))),
    "sin": _real(lambda x: None if mp.isinf(x) else mp.sin(x)), "cos": _real(lambda x: None if mp.isinf(x) else mp.cos(x)),
    "tan": _real(lambda x: None if mp.isinf(x) else mp.tan(x)), "sinh": _real(mp.sinh), "cosh": _real(mp.cosh),
    "tanh": _real(mp.tanh), "asin": _real(lambda x: None if abs(x) > 1 else mp.asin(x)),
    "acos": _real(lambda x: None if abs(x) > 1 else mp.acos(x)), "atan": _real(mp.atan), "asinh": _real(mp.asinh),
    "acosh": _real(lambda x: None if x < 1 else mp.acosh(x)),
    "atanh": _real(lambda x: mp.inf if x == 1 else (-mp.inf if x == -1 else (None if abs(x) > 1 else mp.atanh(x)))),
    "erf": _real(mp.erf), "erfc": _real(mp.erfc), "cbrt": _real(lambda x: mp.cbrt(x) if x >= 0 else -mp.cbrt(-x)),
    "floor": _real(mp.floor), "ceil": _real(mp.ceil), "trunc": _real(lambda x: mp.floor(x) if x >= 0 else mp.ceil(x)),
    "round": _real(lambda x: mp.sign(x) * mp.floor(abs(x) + mp.mpf(1) / 2)), "roundeven": _real(mp.nint),
    "abs": _real(abs), "rcp": _real(lambda x: _ext_div(mp.mpf(1), x)),
    "pow": _real(lambda x, y: None if x < 0 and y != mp.floor(y) else (mp.mpf(0) if x == 0 and y > 0 else mp.power(x, y))),
    "add": _real(lambda x, y: x + y), "sub": _real(lambda x, y: x - y), "mul": _real(lambda x, y: x * y),
    "div": _real(_ext_div), "fma": _real(lambda x, y, z: x * y + z),
    "maxnum": lambda x, y: y if mp.isnan(x) else (x if mp.isnan(y) else max(x, y)),
    "minnum": lambda x, y: y if mp.isnan(x) else (x if mp.isnan(y) else min(x, y)),
    "copysign": _real(lambda x, y: abs(x) * (1 if mp.sign(y) >= 0 else -1)),
    "remf": _real(lambda x, y: None if y == 0 else x - y * (mp.floor(x / y) if x / y >= 0 else mp.ceil(x / y))),
    "saturate": _real(lambda x: min(max(x, mp.mpf(0)), mp.mpf(1))),
}
# predicates and integer abs (int32 results)
IREF = {
    "isnan": lambda x: int(bool(mp.isnan(x))), "isinf": lambda x: int(bool(mp.isinf(x))),
    "isfinite": lambda x: int(not (mp.isnan(x) or mp.isinf(x))),
    "signbit": None,  # needs the sign of zero: computed on the raw float in the test
    "abs": lambda x: ((abs(int(x)) + 2 ** 31) % 2 ** 32) - 2 ** 31,  # two's complement wrap: abs(INT_MIN) = INT_MIN
}
# inputs per domain: (positive, boundary, premise violation); "S" / "X" are replaced by the subnormal / largest value
DOMAIN = {
    "total": ([0.5, -1.25, 2.0, -0.375, 3.0, 1e-3], [0.0, -0.0, "S", "-S", "X", "-X", float("inf"), float("-inf")],
              [float("nan")]),
    "trig": ([0.5, -1.25, 2.0, -0.375, 3.0, 1e-3], [0.0, -0.0, "S", 100.0, -1e4], [float("inf"), float("nan")]),
    "pos": ([0.5, 2.0, 10.0, 1e-3, 3.75], ["S", 1.0, "X", float("inf"), 0.0, -0.0], [-1.0, -2.5, float("-inf")]),
    "gt_m1": ([0.5, -0.5, 2.0, 1e-4], [-1.0, 0.0, -0.0, "X", float("inf")], [-1.5, -3.0]),
    "nonneg": ([0.25, 2.0, 9.0, 1e-3], [0.0, -0.0, "S", "X", float("inf")], [-1.0, -0.5]),
    "pos_r": ([0.25, 2.0, 9.0], ["S", "X", float("inf"), 0.0], [-1.0, -4.0]),
    "unit": ([0.5, -0.5, 0.0, 0.25], [1.0, -1.0, -0.0, "S"], [1.5, -2.0, float("inf")]),
    "ge1": ([1.5, 2.0, 10.0], [1.0, "X", float("inf")], [0.5, -1.0]),
    "open_unit": ([0.5, -0.5, 0.25], [1.0, -1.0, 0.0, -0.0], [1.5, -2.0]),
}
UNARY_DOMAIN = {"log": "pos", "log2": "pos", "log10": "pos", "log1p": "gt_m1", "sqrt": "nonneg", "rsqrt": "pos_r",
                "asin": "unit", "acos": "unit", "acosh": "ge1", "atanh": "open_unit", "sin": "trig", "cos": "trig",
                "tan": "trig", "rcp": "pos_r"}
BINARY = {  # positive pairs, boundary pairs, premise pairs
    "pow": ([(0.5, 2.0), (2.0, -1.5), (3.0, 0.5)], [(0.0, 2.0), (2.0, 0.0), (1.0, 7.0)], [(-2.0, 0.5), (-1.0, 0.25)]),
    "div": ([(1.0, 3.0), (-2.5, 0.75), (7.0, -2.0)], [(0.0, 3.0), ("X", 0.5), ("S", 4.0)], [(1.0, 0.0), (0.0, 0.0)]),
    "remf": ([(7.0, 3.0), (-2.5, 0.75)], [(0.0, 3.0), ("S", 1.0)], [(1.0, 0.0)]),
    "default": ([(1.0, 3.0), (-2.5, 0.75), (7.0, -2.0)], [(0.0, -0.0), ("X", "X"), ("S", "-S")],
                [(float("nan"), 1.0)]),
}
TERNARY = ([(1.0, 3.0, -2.0), (0.5, -0.25, 4.0)], [(0.0, 5.0, -0.0), ("X", 2.0, "-X")], [(float("nan"), 1.0, 2.0)])


def _val(v, dt):
    if isinstance(v, str):
        x = _sub(dt) if v.endswith("S") else _max(dt)
        return -x if v.startswith("-") else x
    return v


# carrier kernels for registered symbols the 3.6.0 Python bindings do not expose: the carrier's TTIR / TTGIR is
# compiled and its symbol replaced by the target symbol (evaluation only; such symbols are not emitted by the frontend)
CARRIER = {("fp32",): ("exp", "__nv_expf"), ("fp64",): ("exp", "__nv_exp"),
           ("fp32", "fp32"): ("pow", "__nv_powf"), ("fp64", "fp64"): ("pow", "__nv_pow")}
ARITY = {"div": 2, "maxnum": 2, "minnum": 2, "pow": 2, "fma": 3, "add": 2, "sub": 2, "mul": 2, "copysign": 2,
         "remf": 2}


def _cases():
    table = H.libdevice_table()
    out = []
    for sym, internal in sorted(M.LIBDEVICE.items()):
        if sym in table:
            pyname, args, ret = table[sym]
            carrier = None
        else:
            dt = "fp64" if not sym.endswith("f") and "_fast_" not in sym else "fp32"
            args = (dt,) * ARITY.get(internal, 1)
            if args not in CARRIER:
                continue
            pyname, carrier = CARRIER[args]
            ret = dt
        if internal not in REF and internal not in IREF:
            continue
        if any(a not in ("fp32", "fp64", "int32") for a in args) or ret not in ("fp32", "fp64", "int32"):
            continue
        for cat in ("positive", "boundary", "premise_violation"):
            out.append(pytest.param(sym, pyname, args, ret, internal, cat, carrier,
                                    id=f"{sym}-{cat}"))
    return out


@pytest.mark.parametrize("sym,pyname,args,ret,internal,category,carrier", _cases())
def test_libdevice_signature(sym, pyname, args, ret, internal, category, carrier):
    k = {"positive": 0, "boundary": 1, "premise_violation": 2}[category]
    dt = args[0]
    if dt == "int32":
        cols = [[[3, -7, 100], [0, 2 ** 31 - 1, -2 ** 31 + 1], [-2 ** 31]][k]]
    elif len(args) == 1:
        vals = DOMAIN[UNARY_DOMAIN.get(internal, "total")][k]
        cols = [[_val(v, dt) for v in vals]]
    elif len(args) == 2:
        pairs = BINARY.get(internal, BINARY["default"])[k]
        cols = [[_val(p[i], dt) for p in pairs] for i in range(2)]
    else:
        cols = [[_val(t[i], dt) for t in TERNARY[k]] for i in range(3)]
    cols = [H.pad(c) for c in cols]
    inputs = [np.asarray(c, H.NP[d]) for c, d in zip(cols, args)]
    expr = f"libdevice.{pyname}(" + ", ".join(f"a{i}" for i in range(len(args))) + ")"
    lo, hi, st, ref = H.evaluate(sym, expr, args, ret, inputs,
                                 rename=(f'"{carrier}"', f'"{sym}"') if carrier else None)
    if internal in IREF and ret == "int32":
        if internal == "signbit":
            exact = [int(bool(np.signbit(x[j]))) for x in inputs[:1] for j in range(H.N)]
        elif dt == "int32":
            exact = [IREF[internal](int(inputs[0][j])) for j in range(H.N)]
        else:
            exact = [IREF[internal](_m(inputs[0][j])) for j in range(H.N)]
        bad = H.check_lanes(category, lo, hi, st, exact, is_int=True)
        assert not bad, f"{sym} {category}: {bad[:4]}"
        return
    exact = [REF[internal](*[_m(x[j]) for x in inputs]) for j in range(H.N)]
    bad = H.check_lanes(category, lo, hi, st, exact)
    assert not bad, f"{sym} {category}: {bad[:4]}"
