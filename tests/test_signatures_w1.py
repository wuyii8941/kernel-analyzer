"""W1 evidence completion (DSL v2 increment 8): positive / boundary / premise-violation cases with trigger evidence for the
supported signatures that had none.  Operations the installed 3.6.0 frontend never emits are tested on hand-written TTIR
(marked; the syntax of 3.6.0 / official main); the others on compiled-only kernels.  Expected values are independent
(mpmath, exact integers).  CPU only."""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_libdevice import DOMAIN, REF, UNARY_DOMAIN, _val  # noqa: E402
from test_signatures_structural import _check, _run  # noqa: E402
from test_signatures_sync import _eval  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
INF, NAN = float("inf"), float("nan")
mp.mp.prec = 200


def _unary_ttir(op, t="f32", out="f32", conv=""):
    """hand-written TTIR: z[i] = op(x[i]) for 8 lanes (conv: an extra conversion of the result, e.g. i1 -> i8)."""
    res = "%r" if not conv else "%c"
    extra = f"    %c = {conv}\n" if conv else ""
    return f"""module {{
  tt.func public @kernel(%X: !tt.ptr<{t}>, %Z: !tt.ptr<{out}>) {{
    %o = tt.make_range {{end = 8 : i32, start = 0 : i32}} : tensor<8xi32>
    %x = tt.splat %X : !tt.ptr<{t}> -> tensor<8x!tt.ptr<{t}>>
    %xp = tt.addptr %x, %o : tensor<8x!tt.ptr<{t}>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<{t}>>
    %r = {op}
{extra}    %z = tt.splat %Z : !tt.ptr<{out}> -> tensor<8x!tt.ptr<{out}>>
    %zp = tt.addptr %z, %o : tensor<8x!tt.ptr<{out}>>, tensor<8xi32>
    tt.store %zp, {res} : tensor<8x!tt.ptr<{out}>>
    tt.return
  }}
}}"""


def _binary_ttir(op, t="f32", out="f32"):
    return f"""module {{
  tt.func public @kernel(%X: !tt.ptr<{t}>, %Y: !tt.ptr<{t}>, %Z: !tt.ptr<{out}>) {{
    %o = tt.make_range {{end = 8 : i32, start = 0 : i32}} : tensor<8xi32>
    %x = tt.splat %X : !tt.ptr<{t}> -> tensor<8x!tt.ptr<{t}>>
    %xp = tt.addptr %x, %o : tensor<8x!tt.ptr<{t}>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<{t}>>
    %y = tt.splat %Y : !tt.ptr<{t}> -> tensor<8x!tt.ptr<{t}>>
    %yp = tt.addptr %y, %o : tensor<8x!tt.ptr<{t}>>, tensor<8xi32>
    %w = tt.load %yp : tensor<8x!tt.ptr<{t}>>
    %r = {op}
    %z = tt.splat %Z : !tt.ptr<{out}> -> tensor<8x!tt.ptr<{out}>>
    %zp = tt.addptr %z, %o : tensor<8x!tt.ptr<{out}>>, tensor<8xi32>
    tt.store %zp, %r : tensor<8x!tt.ptr<{out}>>
    tt.return
  }}
}}"""


def _inputs(internal, category):
    dom = DOMAIN[UNARY_DOMAIN.get(internal, "total")]
    vals = {"positive": dom[0], "boundary": dom[1], "premise_violation": dom[2]}[category]
    vals = [_val(v, "fp32") for v in vals]
    return np.asarray((vals * 8)[:8], np.float32)


# ---------------------------------------------------------------- math dialect, hand-written TTIR

MATH_UNARY = ["acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cosh", "erfc", "expm1", "log10", "log1p",
              "sinh", "tan", "tanh", "round", "roundeven", "trunc"]


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("fn", MATH_UNARY)
def test_math_unary_signature(fn, category):
    x = _inputs(fn, category)
    out, _ = _eval(_unary_ttir(f"math.{fn} %v : tensor<8xf32>"), {"X": ("f32", x), "Z": ("f32", np.zeros(8))})
    lo, hi, st = out["Z"]
    exact = [REF[fn](mp.mpf(float(v))) if not np.isnan(v) else mp.nan for v in x]
    _check(category if category != "premise_violation" or fn not in ("round", "roundeven", "trunc") else "boundary",
           lo, hi, st, exact)


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("fn", ["isnan", "isinf", "isfinite"])
def test_math_predicate_signature(fn, category):
    x = np.asarray({"positive": [0.5, -1.0, 2.0, 3.0, 1e-3, -7.0, 4.0, 1.0],
                    "boundary": [0.0, -0.0, 1e-45, 3.4e38, -3.4e38, INF, -INF, NAN],
                    "premise_violation": [NAN, NAN, INF, -INF, 1.0, 2.0, 3.0, 4.0]}[category], np.float32)
    out, _ = _eval(_unary_ttir(f"math.{fn} %v : tensor<8xf32>", out="i8", conv="arith.extui %r : tensor<8xi1> to tensor<8xi8>"),
                   {"X": ("f32", x), "Z": ("i8", np.zeros(8))})
    lo, hi, st = out["Z"]
    f = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite}[fn]
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(b) for b in f(x)]


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("fn", ["copysign", "powf"])
def test_math_binary_signature(fn, category):
    a = {"positive": [0.5, -2.0, 3.0, 1.5, 2.0, 0.25, 4.0, 9.0], "boundary": [0.0, -0.0, 1.0, 2.0, 1e-45, 3.4e38, 1.0, 0.5],
         "premise_violation": [-2.0, -1.0, NAN, 1.0, -3.0, -0.5, NAN, 2.0]}[category]
    b = {"positive": [2.0, 3.0, -0.5, 1.0, -1.0, 2.0, 0.5, 0.5], "boundary": [2.0, 1.0, 0.0, -0.0, 1.0, 0.5, 7.0, -3.0],
         "premise_violation": [0.5, 0.25, 1.0, NAN, 1.5, 0.5, 2.0, NAN]}[category]
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    out, _ = _eval(_binary_ttir(f"math.{fn} %v, %w : tensor<8xf32>"), {"X": ("f32", x), "Y": ("f32", y),
                                                                       "Z": ("f32", np.zeros(8))})
    lo, hi, st = out["Z"]
    ref = REF["copysign" if fn == "copysign" else "pow"]
    exact = []
    for p, q in zip(x, y):
        if np.isnan(p) or np.isnan(q):
            exact.append(mp.nan)
        elif fn == "copysign":
            exact.append(mp.mpf(abs(float(p))) * (-1 if np.signbit(q) else 1))
        else:
            exact.append(ref(mp.mpf(float(p)), mp.mpf(float(q))))
    for j, (p, q) in enumerate(zip(x, y)):   # IEEE pow: pow(1, y) = 1 and pow(x, 0) = 1 even for NaN operands
        if fn == "powf" and (p == 1.0 or q == 0.0):
            exact[j] = mp.mpf(1)
    if fn == "copysign":   # the sign of zero decides: compare exactly
        for j, e in enumerate(exact):
            if st[j] == H.ST_OK:
                assert mp.mpf(float(lo[j])) <= e <= mp.mpf(float(hi[j]))
        return
    _check(category, lo, hi, st, exact)


# ---------------------------------------------------------------- arith operations the 3.6.0 frontend does not emit

def _int_case(category):
    a = {"positive": [7, -7, 9, -9, 100, 1, 0, 13], "boundary": [2 ** 31 - 1, -2 ** 31, 0, -1, 1, 6, -6, 2 ** 31 - 1],
         "premise_violation": [5, 6, 7, 8, -2 ** 31, 1, 2, 3]}[category]
    b = {"positive": [2, 2, -4, -4, 7, 3, 5, 5], "boundary": [1, 1, 3, 2, -1, 3, 3, 2 ** 31 - 1],
         "premise_violation": [0, 0, 1, 2, -1, 0, 1, 1]}[category]
    return np.asarray(a, np.int32), np.asarray(b, np.int32)


def _ceil(p, q):
    return -((-p) // q)


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("op", ["ceildivsi", "ceildivui", "floordivsi"])
def test_arith_division_variants_signature(op, category):
    a, b = _int_case(category)
    out, _ = _eval(_binary_ttir(f"arith.{op} %v, %w : tensor<8xi32>", t="i32", out="i32"),
                   {"X": ("i32", a), "Y": ("i32", b), "Z": ("i32", np.zeros(8))})
    lo, hi, st = out["Z"]
    for j, (p, q) in enumerate(zip(a.tolist(), b.tolist())):
        if op == "ceildivui":
            p, q = p % 2 ** 32, q % 2 ** 32
        if q == 0 or (op != "ceildivui" and p == -2 ** 31 and q == -1):
            assert st[j] != H.ST_OK, (op, p, q)      # division by zero / signed overflow: no value
            continue
        want = _ceil(p, q) if op.startswith("ceil") else p // q
        assert st[j] == H.ST_OK and (int(lo[j]) - want) % 2 ** 32 == 0, (op, p, q, int(lo[j]), want)


@pytest.mark.parametrize("category", CATS)
def test_arith_casts_and_negf_signature(category):
    x = {"positive": [0.5, -1.25, 2.0, 3.0, 7.75, 1.0, 100.0, 0.0],
         "boundary": [0.0, -0.0, 1e-45, 3.4e38, -3.4e38, 4294967040.0, 0.999, INF],
         "premise_violation": [-1.0, -0.5, NAN, 4294967296.0, -INF, 1e38, 2.0, 3.0]}[category]
    x = np.asarray(x, np.float32)
    out, _ = _eval(_unary_ttir("arith.negf %v : tensor<8xf32>"), {"X": ("f32", x), "Z": ("f32", np.zeros(8))})
    lo, hi, st = out["Z"]
    _check("boundary" if category == "premise_violation" else category, lo, hi, st,
           [mp.nan if np.isnan(v) else (-mp.mpf(float(v)) if np.isfinite(v) else (-mp.inf if v > 0 else mp.inf))
            for v in x])
    # fptoui: defined where trunc(x) fits in [0, 2^32)
    out, _ = _eval(_unary_ttir("arith.fptoui %v : tensor<8xf32> to tensor<8xi32>", out="i32"),
                   {"X": ("f32", x), "Z": ("i32", np.zeros(8))})
    lo, hi, st = out["Z"]
    for j, v in enumerate(x.tolist()):
        if np.isfinite(v) and 0 <= int(v) < 2 ** 32 and v > -1:
            assert st[j] == H.ST_OK and int(lo[j]) % 2 ** 32 == int(v)
        else:
            assert st[j] != H.ST_OK
    # bitcast f32 -> i32 of exactly representable points
    out, _ = _eval(_unary_ttir("arith.bitcast %v : tensor<8xf32> to tensor<8xi32>", out="i32"),
                   {"X": ("f32", x), "Z": ("i32", np.zeros(8))})
    lo, hi, st = out["Z"]
    bits = x.view(np.int32)
    for j, v in enumerate(x.tolist()):
        if not np.isnan(v):
            assert st[j] == H.ST_OK and int(lo[j]) == int(bits[j])
    # index_cast i32 -> i64 (sign extension)
    ints = np.asarray({"positive": [1, 2, 3, 4, 5, 6, 7, 8], "boundary": [2 ** 31 - 1, -2 ** 31, 0, -1, 1, 2, 3, 4],
                       "premise_violation": [-5, -6, 7, 8, 9, 10, 11, 12]}[category], np.int32)
    out, _ = _eval(_unary_ttir("arith.index_cast %v : tensor<8xi32> to tensor<8xi64>", t="i32", out="i64"),
                   {"X": ("i32", ints), "Z": ("i64", np.zeros(8))})
    lo, hi, st = out["Z"]
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(v) for v in ints]


# ---------------------------------------------------------------- ub.poison, tt.unsplat (hand-written TTIR)

POISON = """module {
  tt.func public @kernel(%X: !tt.ptr<f32>, %M: !tt.ptr<i8>, %Z: !tt.ptr<f32>) {
    %o = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %x = tt.splat %X : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %x, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<f32>>
    %m = tt.splat %M : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %mp = tt.addptr %m, %o : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %mv = tt.load %mp : tensor<8x!tt.ptr<i8>>
    %c0 = arith.constant dense<0> : tensor<8xi8>
    %mb = arith.cmpi ne, %mv, %c0 : tensor<8xi8>
    %p = ub.poison : tensor<8xf32>
    %r = arith.select %mb, %v, %p : tensor<8xi1>, tensor<8xf32>
    %z = tt.splat %Z : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %zp = tt.addptr %z, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %zp, %r : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_poison_signature(category):
    x = np.arange(8, dtype=np.float32) + 0.5
    m = {"positive": [1] * 8, "boundary": [1, 0, 1, 0, 1, 1, 1, 1], "premise_violation": [0] * 8}[category]
    out, _ = _eval(POISON, {"X": ("f32", x), "M": ("i8", m), "Z": ("f32", np.zeros(8))})
    lo, hi, st = out["Z"]
    for j in range(8):
        if m[j]:
            assert st[j] == H.ST_OK and lo[j] == x[j]
        else:
            assert st[j] != H.ST_OK          # a poison value reaching memory has no reference value


UNSPLAT = """module {
  tt.func public @kernel(%X: !tt.ptr<i64>, %Z: !tt.ptr<i64>) {
    %v = tt.load %X : !tt.ptr<i64>
    %t = tt.splat %v : i64 -> tensor<1xi64>
    %s = tt.unsplat %t : tensor<1xi64>
    %c = arith.constant 3 : i64
    %r = arith.muli %s, %c : i64
    tt.store %Z, %r : !tt.ptr<i64>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_unsplat_signature(category):
    v = {"positive": 7, "boundary": -2 ** 63 // 3, "premise_violation": 2 ** 62}[category]
    out, _ = _eval(UNSPLAT, {"X": ("i64", [v]), "Z": ("i64", [0])})
    lo, hi, st = out["Z"]
    want = ((v * 3 + 2 ** 63) % 2 ** 64) - 2 ** 63    # i64 multiplication wraps
    assert st[0] == H.ST_OK and int(lo[0]) == want


# ---------------------------------------------------------------- reduce combiners (compiled)

COMBINERS = {
    "and": ("a & b", "int32"), "or": ("a | b", "int32"), "max_int": ("tl.maximum(a, b)", "int32"),
    "min_int": ("tl.minimum(a, b)", "int32"), "max_uint": ("tl.maximum(a, b)", "uint32"),
    "min_uint": ("tl.minimum(a, b)", "uint32"), "prod_int": ("a * b", "int32"), "prod": ("a * b", "fp32"),
    "max_nan": ("tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)", "fp32"),
    "min_nan": ("tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)", "fp32"),
    "max_select": ("tl.where(a > b, a, b)", "fp32"), "min_select": ("tl.where(a < b, a, b)", "fp32"),
}


def _combiner_data(kind, dtype, category):
    if dtype == "fp32":
        d = {"positive": [0.5, 1.5, -2.0, 3.0, 1.25, -0.75, 2.0, 1.0],
             "boundary": [0.0, -0.0, 1e-45, 3.4e38, -3.4e38, 1.0, 1.0, 2.0],
             "premise_violation": [NAN, 1.0, 2.0, INF, -INF, 0.5, 3.0, 4.0]}[category]
        if kind == "prod" and category == "boundary":
            d = [1e-20, 1e-20, 1e20, 1e20, -1.0, 1.0, 2.0, 0.5]
        return np.asarray(d, np.float32)
    lo, hi = (0, 2 ** 32) if dtype == "uint32" else (-2 ** 31, 2 ** 31)
    d = {"positive": [5, 3, 12, 7, 9, 1, 2, 6], "boundary": [lo, hi - 1, 0, 1, hi - 1, lo, 2, 3],
         "premise_violation": [hi - 1, hi - 1, hi - 1, 2, 3, 5, 7, 11]}[category]
    return np.asarray(d, np.int64 if dtype == "uint32" else np.int32).astype(np.uint32 if dtype == "uint32" else np.int32)


def _fold(kind, vals):
    import functools
    if kind in ("max_nan", "min_nan") and any(mp.isnan(v) for v in vals):
        return None
    f = {"and": lambda a, b: a & b, "or": lambda a, b: a | b, "max_int": max, "min_int": min, "max_uint": max,
         "min_uint": min, "prod_int": lambda a, b: a * b, "prod": lambda a, b: a * b, "max_nan": max, "min_nan": min,
         "max_select": max, "min_select": min}[kind]
    return functools.reduce(f, vals)


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("kind", sorted(COMBINERS))
def test_reduce_combiner_signature(kind, category):
    expr, dtype = COMBINERS[kind]
    data = _combiner_data(kind, dtype, category)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
            f"    r = tl.reduce(x, 0, _comb)\n    tl.store(out + tl.arange(0, 1), r + tl.zeros([1], dtype=r.dtype))\n"
            "\n\n@triton.jit\ndef _comb(a, b):\n" f"    return {expr}\n")
    lo, hi, st = _run(f"comb_{kind}", body, {"x_ptr": (dtype, data), "out": (dtype, np.zeros(1))})
    if dtype == "fp32":
        if kind in ("max_select", "min_select", "prod") and any(np.isnan(data)) | any(np.isinf(data)):
            return                                    # order dependent / special: only containment is meaningful
        vals = [mp.mpf(float(v)) if np.isfinite(v) else (mp.nan if np.isnan(v) else (mp.inf if v > 0 else -mp.inf))
                for v in data]
        want = _fold(kind, vals)
        if want is None:
            assert st[0] == H.ST_NAN or st[0] != H.ST_OK
            return
        _check("boundary" if category != "positive" else "positive", lo, hi, st, [want])
        return
    vals = [int(v) for v in data]
    want = _fold(kind, vals)
    if kind == "prod_int":
        want = ((want + 2 ** 31) % 2 ** 32) - 2 ** 31
    if dtype == "uint32":
        assert st[0] == H.ST_OK and int(lo[0]) % 2 ** 32 == want % 2 ** 32, (kind, int(lo[0]), want)
    else:
        assert st[0] == H.ST_OK and int(lo[0]) == want, (kind, int(lo[0]), want)


# ---------------------------------------------------------------- inline asm patterns (compiled)

ASM = {"ex2.approx.f32 $0, $1;": lambda x: mp.power(2, x), "lg2.approx.f32 $0, $1;": lambda x: None if x <= 0 else mp.log(x, 2),
       "rcp.approx.f32 $0, $1;": lambda x: None if x == 0 else 1 / x, "rcp.rn.f32 $0, $1;": lambda x: None if x == 0 else 1 / x,
       "rsqrt.approx.f32 $0, $1;": lambda x: None if x <= 0 else 1 / mp.sqrt(x),
       "sqrt.approx.f32 $0, $1;": lambda x: None if x < 0 else mp.sqrt(x), "sqrt.rn.f32 $0, $1;": lambda x: None if x < 0 else mp.sqrt(x),
       "sin.approx.f32 $0, $1;": mp.sin, "cos.approx.f32 $0, $1;": mp.cos, "tanh.approx.f32 $0, $1;": mp.tanh,
       "mov.b32 $0, $1;": lambda x: x}


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("asm", sorted(ASM))
def test_inline_asm_pattern_signature(asm, category):
    x = np.asarray({"positive": [0.5, 1.25, 2.0, 3.0, 0.75, 1.5, 4.0, 0.125],
                    "boundary": [1.0, 2.0 ** -20, 64.0, 1e-3, 100.0, 0.5, 2.0, 8.0],
                    "premise_violation": [-1.0, 0.0, -0.5, NAN, -2.0, 1.0, 2.0, 3.0]}[category], np.float32)
    cons = "=r,r" if asm.startswith("mov") else "=f,f"
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
            f"    y = tl.inline_asm_elementwise(\"{asm}\", \"{cons}\", [x], "
            "dtype=tl.float32, is_pure=True, pack=1)\n    tl.store(out + i, y)\n")
    lo, hi, st = _run("asm_" + asm.split()[0].replace(".", "_"), body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(8))})
    exact = []
    for v in x:
        if np.isnan(v):
            exact.append(mp.nan)
            continue
        e = ASM[asm](mp.mpf(float(v)))
        exact.append(e)
    _check(category if category != "premise_violation" else "premise_violation", lo, hi, st, exact)


# ---------------------------------------------------------------- part 2: categories for rules that had positive cases only

@pytest.mark.parametrize("category", CATS)
def test_program_ids_signature(category):
    grid = {"positive": (2, 1, 1), "boundary": (2, 3, 4), "premise_violation": (4, 1, 1)}[category]
    n_out = 24 if category != "premise_violation" else 2      # a program id beyond the buffer: out of bounds
    body = ("    a = tl.program_id(0)\n    b = tl.program_id(1)\n    c = tl.program_id(2)\n"
            "    lin = a + tl.num_programs(0) * (b + tl.num_programs(1) * c)\n"
            "    tl.store(out + lin, a + 10 * b + 100 * c + 1000 * tl.num_programs(2))\n")
    lo, hi, st = _run("pids", body, {"out": ("int32", np.zeros(n_out))}, grid=grid)
    gx, gy, gz = grid
    for c in range(gz):
        for b in range(gy):
            for a in range(gx):
                lin = a + gx * (b + gy * c)
                if lin < n_out:
                    assert st[lin] == H.ST_OK and int(lo[lin]) == a + 10 * b + 100 * c + 1000 * gz


@pytest.mark.parametrize("category", CATS)
def test_pointer_integer_round_trip_signature(category):
    off = {"positive": 2, "boundary": 7, "premise_violation": 4096}[category]     # elements past the start
    body = ("    p = x_ptr.to(tl.int64, bitcast=True)\n"
            f"    q = (p + 4 * {off}).to(tl.pointer_type(tl.float32), bitcast=True)\n"
            "    tl.store(out + tl.arange(0, 1), tl.load(q) + tl.zeros([1], dtype=tl.float32))\n")
    x = np.arange(8, dtype=np.float32) * 1.5
    lo, hi, st = _run(f"p2i_{off}", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(1))})
    if off < 8:
        assert st[0] == H.ST_OK and lo[0] == x[off]
    else:
        assert st[0] != H.ST_OK                   # an address outside every captured storage


@pytest.mark.parametrize("category", CATS)
def test_atomic_cas_categories(category):
    if category == "premise_violation":
        # a relaxed spin lock: no acquire, so the critical section races
        body = ("    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, 0, 1, sem=\"relaxed\") == 1:\n        pass\n"
                "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
        lo, hi, st = _run("cas_relaxed", body, {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                          out_name="data", grid=(4, 1, 1))
        assert (st != H.ST_OK).all()
        return
    init = 0 if category == "positive" else 5      # boundary: the compare fails, nothing is swapped
    body = "    old = tl.atomic_cas(lock, 0, 9)\n    tl.store(out, old)\n"
    lo, hi, st = _run("cas_one", body, {"lock": ("int32", [init]), "out": ("int32", [0])})
    assert st[0] == H.ST_OK and int(lo[0]) == init
    lo, hi, st = _run("cas_one", body, {"lock": ("int32", [init]), "out": ("int32", [0])}, out_name="lock")
    assert st[0] == H.ST_OK and int(lo[0]) == (9 if init == 0 else init)


@pytest.mark.parametrize("category", CATS)
def test_barrier_signature(category):
    barrier = "    tl.debug_barrier()\n" if category != "premise_violation" else ""
    extra = "    tl.debug_barrier()\n" if category == "boundary" else ""
    body = ("    i = tl.arange(0, N)\n    tl.store(tmp + i, tl.load(x_ptr + i) * 2.0)\n" + barrier + extra +
            "    j = (i + 1) % N\n    tl.store(out + i, tl.load(tmp + j))\n")
    x = np.arange(8, dtype=np.float32)
    lo, hi, st = _run("barrier_" + category, body, {"x_ptr": ("fp32", x), "tmp": ("fp32", np.zeros(8)),
                                                    "out": ("fp32", np.zeros(8))})
    if category == "premise_violation":
        assert (st != H.ST_OK).any()               # another thread's store read without a barrier
    else:
        assert (st == H.ST_OK).all() and [float(v) for v in lo] == [float(x[(i + 1) % 8] * 2) for i in range(8)]


def _welford_body():
    return ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    w = tl.load(w_ptr + i)\n"
            "    m, s, n = tl.reduce((x, tl.zeros_like(x), w), 0, _welf)\n"
            "    tl.store(out + tl.arange(0, 1), s / n + tl.zeros([1], dtype=tl.float32))\n"
            "\n\n@triton.jit\ndef _welf(m1, s1, w1, m2, s2, w2):\n"
            "    w = w1 + w2\n    r = tl.where(w == 0.0, 0.0, w2 / w)\n    d = m2 - m1\n"
            "    return m1 + d * r, s1 + s2 + d * d * w1 * r, w\n")


@pytest.mark.parametrize("category", CATS)
def test_welford_categories(category):
    x = {"positive": [0.5, 1.5, -2.0, 3.0, 1.25, -0.75, 2.0, 1.0], "boundary": [2.0] * 8,
         "premise_violation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}[category]
    w = [1.0] * 8 if category != "premise_violation" else [0.0] * 8   # all weights zero: no variance exists
    lo, hi, st = _run("welf", _welford_body(), {"x_ptr": ("fp32", np.asarray(x, np.float32)),
                                                "w_ptr": ("fp32", np.asarray(w, np.float32)), "out": ("fp32", np.zeros(1))})
    if category == "premise_violation":
        assert st[0] != H.ST_OK
        return
    from fractions import Fraction as Fr
    xs = [Fr(float(v)) for v in x]
    mean = sum(xs) / 8
    var = sum((v - mean) ** 2 for v in xs) / 8
    assert st[0] == H.ST_OK and Fr(float(lo[0])) <= var <= Fr(float(hi[0]))


@pytest.mark.parametrize("category", CATS)
def test_generic_combine_categories(category):
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    r = tl.reduce(x, 0, _gen)\n"
            "    tl.store(out + tl.arange(0, 1), r + tl.zeros([1], dtype=r.dtype))\n"
            "\n\n@triton.jit\ndef _gen(a, b):\n    return a + 2 * b\n")
    x = np.asarray({"positive": [1, 2, 3, 4, 5, 6, 7, 8], "boundary": [0, 0, 0, 0, 0, 0, 0, 1],
                    "premise_violation": [1, 2, 3, 4, 5, 6, 7, 8]}[category], np.int32)
    lo, hi, st = _run("gen_comb", body, {"x_ptr": ("int32", x), "out": ("int32", np.zeros(1))},
                      drop_ttgir=(category == "premise_violation"))
    if category == "premise_violation":
        assert st[0] != H.ST_OK                      # no TTGIR: no trusted combination order
        return
    # the 3.6.0 lowering: lanes 0..7 hold one element each (lanes 8..31 replicate), butterfly combine(own, partner)
    lanes = [int(x[l % 8]) for l in range(32)]
    stride = 16
    while stride >= 1:
        lanes = [lanes[l] + 2 * lanes[l ^ stride] for l in range(32)]
        stride //= 2
    if len(set(lanes)) == 1:
        assert st[0] == H.ST_OK and int(lo[0]) == lanes[0]
    else:
        assert st[0] != H.ST_OK


# hand-written TTIR: 3.6.0 emits tt.assert only in debug mode; cf.assert comes from lowered control flow
ASSERT_TTIR = """module {
  tt.func public @kernel(%X: !tt.ptr<f32>, %Z: !tt.ptr<f32>) {
    %o = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %x = tt.splat %X : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %x, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<f32>>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %pos = arith.cmpf ogt, %v, %c0 : tensor<8xf32>
    tt.assert %pos, "positive" : tensor<8xi1>
    %s = tt.load %X : !tt.ptr<f32>
    %c00 = arith.constant 0.000000e+00 : f32
    %spos = arith.cmpf ogt, %s, %c00 : f32
    cf.assert %spos, "first positive"
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %v : tensor<8xf32>
    %two = arith.constant dense<2.000000e+00> : tensor<8xf32>
    %r = arith.mulf %v, %two : tensor<8xf32>
    %z = tt.splat %Z : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %zp = tt.addptr %z, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %zp, %r : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_assert_and_print_signature(category):
    x = np.asarray({"positive": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "boundary": [1e-45] * 8,
                    "premise_violation": [-1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}[category], np.float32)
    out, ref = _eval(ASSERT_TTIR, {"X": ("f32", x), "Z": ("f32", np.zeros(8))})
    violated = any("violates assert" in k for k in ref.reasons)
    assert violated == (category == "premise_violation")
    lo, hi, st = out["Z"]
    assert (st == H.ST_OK).all() and [float(v) for v in lo] == [float(v) * 2 for v in x]


@pytest.mark.parametrize("category", CATS)
def test_gather_signature(category):
    idx = np.asarray({"positive": [7, 6, 5, 4, 3, 2, 1, 0], "boundary": [0, 0, 7, 7, 3, 3, 0, 7],
                      "premise_violation": [0, 1, 2, 8, 4, 5, 6, -1]}[category], np.int32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    k = tl.load(k_ptr + i)\n"
            "    tl.store(out + i, tl.gather(x, k, 0))\n")
    x = np.arange(8, dtype=np.float32) + 0.5
    lo, hi, st = _run("gather", body, {"x_ptr": ("fp32", x), "k_ptr": ("int32", idx), "out": ("fp32", np.zeros(8))})
    for j, k in enumerate(idx.tolist()):
        if 0 <= k < 8:
            assert st[j] == H.ST_OK and lo[j] == x[k]
        else:
            assert st[j] != H.ST_OK                  # an index outside the source: no value


@pytest.mark.parametrize("category", CATS)
def test_noinline_call_signature(category):
    # noinline functions take scalars: pass the pointers (the official test_noinline pattern)
    body = ("    _f(x_ptr, out)\n"
            "\n\n@triton.jit(noinline=True)\ndef _f(xp, op):\n    i = tl.arange(0, 8)\n    v = tl.load(xp + i)\n"
            "    tl.store(op + i, 1.0 / v + v)\n")
    x = np.asarray({"positive": [1.0, 2.0, 4.0, 0.5, 8.0, 0.25, 3.0, 5.0], "boundary": [1e-38, 3.4e38, 1.0, -1.0, 2.0, -2.0, 0.5, 4.0],
                    "premise_violation": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}[category], np.float32)
    lo, hi, st = _run("noinline", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(8))})
    exact = [None if v == 0 else 1 / mp.mpf(float(v)) + mp.mpf(float(v)) for v in x]
    _check(category, lo, hi, st, exact)


# ---------------------------------------------------------------- part 3

@pytest.mark.parametrize("category", CATS)
def test_index_castui_signature(category):
    ints = np.asarray({"positive": [1, 2, 3, 4, 5, 6, 7, 8], "boundary": [2 ** 31 - 1, -2 ** 31, 0, -1, 1, 2, 3, 4],
                       "premise_violation": [-5, -6, 7, 8, 9, 10, 11, 12]}[category], np.int32)
    out, _ = _eval(_unary_ttir("arith.index_castui %v : tensor<8xi32> to tensor<8xi64>", t="i32", out="i64"),
                   {"X": ("i32", ints), "Z": ("i64", np.zeros(8))})
    lo, hi, st = out["Z"]
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(v) % 2 ** 32 for v in ints]   # zero extension


CLAMPF = """module {
  tt.func public @kernel(%X: !tt.ptr<f32>, %L: !tt.ptr<f32>, %U: !tt.ptr<f32>, %Z: !tt.ptr<f32>) {
    %o = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %x = tt.splat %X : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %xp = tt.addptr %x, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %xp : tensor<8x!tt.ptr<f32>>
    %l = tt.splat %L : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %lp = tt.addptr %l, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %lo = tt.load %lp : tensor<8x!tt.ptr<f32>>
    %u = tt.splat %U : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %up = tt.addptr %u, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %hi = tt.load %up : tensor<8x!tt.ptr<f32>>
    %r = math.clampf %v to [%lo, %hi] : tensor<8xf32>
    %z = tt.splat %Z : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %zp = tt.addptr %z, %o : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %zp, %r : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_math_clampf_signature(category):
    x = {"positive": [-3.0, -0.5, 0.0, 0.5, 2.0, 5.0, -1.0, 1.0], "boundary": [-1.0, 1.0, -0.0, 0.0, 3.4e38, -3.4e38, 1e-45, -1e-45],
         "premise_violation": [NAN, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}[category]
    lo_b = [-1.0] * 8
    hi_b = [1.0] * 8 if category != "premise_violation" else [1.0, 1.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # min > max: poison
    out, _ = _eval(CLAMPF, {"X": ("f32", x), "L": ("f32", lo_b), "U": ("f32", hi_b), "Z": ("f32", np.zeros(8))})
    lo, hi, st = out["Z"]
    for j in range(8):
        if np.isnan(x[j]) or lo_b[j] > hi_b[j]:
            assert st[j] != H.ST_OK
        else:
            assert st[j] == H.ST_OK and lo[j] == hi[j] == min(max(np.float32(x[j]), lo_b[j]), hi_b[j])


BR = """module {
  tt.func public @kernel(%X: !tt.ptr<i32>, %Z: !tt.ptr<i32>) {
    %v = tt.load %X : !tt.ptr<i32>
    %c0 = arith.constant 0 : i32
    %neg = arith.cmpi slt, %v, %c0 : i32
    cf.cond_br %neg, ^bb1, ^bb2(%v : i32)
  ^bb1:  // pred: ^bb0
    %n = arith.subi %c0, %v : i32
    cf.br ^bb2(%n : i32)
  ^bb2(%a: i32):  // 2 preds: ^bb0, ^bb1
    tt.store %Z, %a : !tt.ptr<i32>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_unconditional_branch_signature(category):
    v = {"positive": -7, "boundary": -2 ** 31 + 1, "premise_violation": -2 ** 31}[category]   # 0 - INT_MIN wraps
    out, _ = _eval(BR, {"X": ("i32", [v]), "Z": ("i32", [0])})
    lo, hi, st = out["Z"]
    assert st[0] == H.ST_OK and int(lo[0]) == ((-v + 2 ** 31) % 2 ** 32) - 2 ** 31


@pytest.mark.parametrize("category", CATS)
def test_contended_cas_serialization_categories(category):
    if category == "premise_violation":
        body = ("    pid = tl.program_id(0)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
                "    f = tl.load(flag)\n    if f == 0:\n        tl.store(out, pid)\n        tl.store(flag, 1)\n"
                "    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
        lo, hi, st = _run("lock_first", body, {"out": ("int32", [-1]), "flag": ("int32", [0]), "lock": ("int32", [0])},
                          grid=(4, 1, 1))
        assert st[0] != H.ST_OK                       # the two orders disagree
        return
    p = 2 if category == "boundary" else 8
    body = ("    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    lo, hi, st = _run("lock_add", body, {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])}, out_name="data",
                      grid=(p, 1, 1))
    assert (st == H.ST_OK).all() and (lo == p).all()


def _cummax_body(comb):
    return ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
            "    v, k = tl.associative_scan((x, i), 0, _cm)\n    tl.store(out + i, k)\n"
            "\n\n@triton.jit\ndef _cm(v1, i1, v2, i2):\n" + comb)


@pytest.mark.parametrize("category", CATS)
def test_multi_operand_scan_signature(category):
    if category == "premise_violation":
        # not associative: (v, i) -> (v1 + 2 v2, i2); the two bracketings disagree
        comb = "    return v1 + 2 * v2, i2\n"
        x = np.arange(1, 9, dtype=np.int32)
        body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
                "    v, k = tl.associative_scan((x, i), 0, _cm)\n    tl.store(out + i, v)\n"
                "\n\n@triton.jit\ndef _cm(v1, i1, v2, i2):\n" + comb)
        lo, hi, st = _run("scan_nonassoc", body, {"x_ptr": ("int32", x), "out": ("int32", np.zeros(8))})
        assert (st[2:] != H.ST_OK).any()
        return
    comb = "    take = v2 > v1\n    return tl.where(take, v2, v1), tl.where(take, i2, i1)\n"
    x = np.asarray({"positive": [3, 1, 4, 1, 5, 9, 2, 6], "boundary": [7, 7, 7, 7, 7, 7, 7, 7]}[category], np.int32)
    lo, hi, st = _run("scan_cummax", _cummax_body(comb), {"x_ptr": ("int32", x), "out": ("int32", np.zeros(8))})
    want, best, bi = [], None, None
    for j, v in enumerate(x.tolist()):
        if best is None or v > best:
            best, bi = v, j
        want.append(bi)
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == want
