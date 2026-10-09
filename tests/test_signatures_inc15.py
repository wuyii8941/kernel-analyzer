"""DSL v2 increment 15: bit counting (math.ctlz / cttz / ctpop, libdevice __nv_clz / clzll / popc / popcll), remf
beyond point divisors, and the integer-representation defect (signed comparisons, division, shifts, extensions and
compare-and-swap on values kept modulo 2^width).  Synthetic captures, CPU only; expected values computed directly."""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_int_sets import _compile, _launch  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
NP = {"i32": np.int32, "i64": np.int64, "i8": np.int8}
TORCH = {"i32": "int32", "i64": "int64", "i8": "int8", "u8": "uint8", "u32": "uint32", "f32": "float32"}


def _eval_ttir(ttir, bufs):
    """bufs: ordered {name: (torch dtype, array)}; one program."""
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    args, base = [], 1 << 20
    for i, (name, (dt, arr)) in enumerate(bufs.items()):
        arr = np.ascontiguousarray(arr)
        raw = arr.reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type=None, dtype=dt,
                                shape=arr.shape, stride=None, element_size=arr.itemsize, data_ptr=base * (i + 1),
                                storage_ptr=base * (i + 1), storage_nbytes=raw.size, storage_id=i, before=raw,
                                after=raw.copy()))
    launch = CapturedLaunch(index=0, kernel_name="k", kernel_hash="", grid=(1, 1, 1), args=args, asm={"ttir": ttir},
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    return ref, {k: base * (i + 1) for i, k in enumerate(bufs)}


def _count(name, x, width):
    u = int(x) & ((1 << width) - 1)
    if name == "ctpop":
        return bin(u).count("1")
    if name == "ctlz":
        return width - u.bit_length()
    return (u & -u).bit_length() - 1 if u else width


MATH = """module {
  tt.func public @k(%x: !tt.ptr<T>, %o: !tt.ptr<T>) attributes {noinline = false} {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %half = arith.constant dense<MASKN> : tensor<8xi32>
    %m = arith.cmpi slt, %r, %half : tensor<8xi32>
    %p = tt.splat %x : !tt.ptr<T> -> tensor<8x!tt.ptr<T>>
    %pp = tt.addptr %p, %r : tensor<8x!tt.ptr<T>>, tensor<8xi32>
    %v = tt.load %pp, %m : tensor<8x!tt.ptr<T>>
    %c = math.OP %v : tensor<8xT>
    %q = tt.splat %o : !tt.ptr<T> -> tensor<8x!tt.ptr<T>>
    %qq = tt.addptr %q, %r : tensor<8x!tt.ptr<T>>, tensor<8xi32>
    tt.store %qq, %c : tensor<8x!tt.ptr<T>>
    tt.return
  }
}
"""


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("op", ["ctlz", "cttz", "ctpop"])
@pytest.mark.parametrize("ty", ["i32", "i64"])
def test_math_bit_counting_signature(op, ty, category):
    width = int(ty[1:])
    rng = np.random.default_rng(15)
    x = {"positive": rng.integers(1, 1 << 30, 8),
         "boundary": [0, -1, -(1 << (width - 1)), 1, (1 << (width - 1)) - 1, 2, 1 << 20, -2],
         "premise_violation": rng.integers(1, 1 << 30, 8)}[category]
    x = np.asarray(x, NP[ty])
    # premise violation: lanes 4..7 are masked off without `other` (undefined input)
    ttir = MATH.replace("T", ty).replace("OP", op).replace("MASKN", "4" if category == "premise_violation" else "8")
    ref, ids = _eval_ttir(ttir, {"x": (TORCH[ty], x), "o": (TORCH[ty], np.zeros(8, NP[ty]))})
    assert ref.rules.get("path.program_aborted", 0) == 0
    buf = ref.buffers[ids["o"]]
    n = 4 if category == "premise_violation" else 8
    for i in range(n):
        assert buf.st[i] == H.ST_OK and int(buf.lo[i]) == _count(op, x[i], width), (i, int(x[i]), int(buf.lo[i]))
    if category == "premise_violation":
        assert (buf.st[4:] != H.ST_OK).all()


LIB = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i, mask=i < LIM)\n"
       "    tl.store(out + i, tl.extra.cuda.libdevice.FN(x))\n")


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("fn,dtype", [("clz", "int32"), ("clz", "int64"), ("popc", "int32"), ("popc", "int64")])
def test_libdevice_bit_counting_signature(fn, dtype, category):
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    from test_signatures_structural import C
    width = 64 if dtype == "int64" else 32
    rng = np.random.default_rng(16)
    x = {"positive": rng.integers(1, 1 << 30, C), "boundary": ([0, -1, -(1 << (width - 1)), 1] * C)[:C],
         "premise_violation": rng.integers(1, 1 << 30, C)}[category]
    lim = C // 2 if category == "premise_violation" else C
    body = LIB.replace("FN", fn).replace("LIM", str(lim))
    bufs = {"x_ptr": (dtype, x), "out": ("int32", np.zeros(C))}
    asm = _compile(f"lib_{fn}_{dtype}_{lim}", body, bufs, {})
    sym = {"clz": "__nv_clz", "popc": "__nv_popc"}[fn] + ("ll" if dtype == "int64" else "")
    assert sym in asm["ttir"]
    ref = evaluate_sequence([_launch(0, "k", asm, bufs, {}, (1, 1, 1))]).launches[0]
    buf = ref.buffers[2 << 20]
    op = {"clz": "ctlz", "popc": "ctpop"}[fn]
    for i in range(lim):
        assert buf.st[i] == H.ST_OK and int(buf.lo[i]) == _count(op, x[i], width), (i, x[i], buf.lo[i])
    if category == "premise_violation":
        assert (buf.st[lim:C] != H.ST_OK).all()


REMF = """module {
  tt.func public @k(%x: !tt.ptr<f32>, %y: !tt.ptr<i64>, %o: !tt.ptr<f32>) attributes {noinline = false} {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %p = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %pp = tt.addptr %p, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %pp : tensor<8x!tt.ptr<f32>>
    %py = tt.splat %y : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %ppy = tt.addptr %py, %r : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    %w = tt.load %ppy : tensor<8x!tt.ptr<i64>>
    %wf = arith.sitofp %w : tensor<8xi64> to tensor<8xf32>
    %z = arith.remf %v, %wf : tensor<8xf32>
    %q = tt.splat %o : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %qq = tt.addptr %q, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %qq, %z : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
"""


@pytest.mark.parametrize("category", CATS)
def test_remf_with_a_converted_int64_divisor(category):
    """test_bin_op (official) pattern: float % int64.  A large int64 converted to f32 is an interval; |x| < |y|
    gives fmod(x, y) = x exactly.  Boundary: |x| equal to the divisor's lower end.  Premise violation: divisor 0."""
    rng = np.random.default_rng(17)
    x = (rng.standard_normal(8) * 100).astype(np.float32)
    y = rng.integers(-(1 << 62), 1 << 62, 8).astype(np.int64)
    y[0] = (1 << 62) + 12345                                   # not representable in f32 or f64: an interval
    if category == "boundary":
        y[:4] = [3, -3, 7, 1]                                  # small point divisors: exact fmod
        x[:4] = [3.0, -7.5, 6.5, -0.0]
    if category == "premise_violation":
        y[2] = 0
    ref, ids = _eval_ttir(REMF, {"x": ("float32", x), "y": ("int64", y), "o": ("float32", np.zeros(8, np.float32))})
    buf = ref.buffers[ids["o"]]
    for i in range(8):
        if category == "premise_violation" and i == 2:
            assert buf.st[i] != H.ST_OK                        # category D: outside the domain, not established
            continue
        want = float(np.fmod(np.float64(x[i]), np.float64(y[i]))) if abs(int(y[i])) < (1 << 24) else float(x[i])
        assert buf.st[i] == H.ST_OK and buf.lo[i] <= want <= buf.hi[i], (i, x[i], y[i], buf.lo[i], buf.hi[i])
        if abs(int(y[i])) >= (1 << 24):
            assert buf.lo[i] == buf.hi[i] == float(x[i])       # exact: the result is x itself


REP = """module {
  tt.func public @k(%x: !tt.ptr<i8>, %o: !tt.ptr<i32>) attributes {noinline = false} {
    %c = arith.constant dense<-56> : tensor<4xi8>
    %three = arith.constant dense<3> : tensor<4xi8>
    %one = arith.constant dense<1> : tensor<4xi8>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %p = tt.splat %x : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %pp = tt.addptr %p, %r : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %v = tt.load %pp : tensor<4x!tt.ptr<i8>>
    %e = arith.cmpi eq, %v, %c : tensor<4xi8>
    %lt = arith.cmpi slt, %v, %one : tensor<4xi8>
    %d = arith.divsi %v, %three : tensor<4xi8>
    %h = arith.shrsi %v, %one : tensor<4xi8>
    %mx = arith.maxsi %v, %one : tensor<4xi8>
    %e32 = arith.extui %e : tensor<4xi1> to tensor<4xi32>
    %lt32 = arith.extui %lt : tensor<4xi1> to tensor<4xi32>
    %d32 = arith.extsi %d : tensor<4xi8> to tensor<4xi32>
    %h32 = arith.extsi %h : tensor<4xi8> to tensor<4xi32>
    %m32 = arith.extsi %mx : tensor<4xi8> to tensor<4xi32>
    %q = tt.splat %o : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %q0 = tt.addptr %q, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %four = arith.constant dense<4> : tensor<4xi32>
    %eight = arith.constant dense<8> : tensor<4xi32>
    %twelve = arith.constant dense<12> : tensor<4xi32>
    %sixteen = arith.constant dense<16> : tensor<4xi32>
    %q1 = tt.addptr %q0, %four : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q2 = tt.addptr %q0, %eight : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q3 = tt.addptr %q0, %twelve : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %q4 = tt.addptr %q0, %sixteen : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %q0, %e32 : tensor<4x!tt.ptr<i32>>
    tt.store %q1, %lt32 : tensor<4x!tt.ptr<i32>>
    tt.store %q2, %d32 : tensor<4x!tt.ptr<i32>>
    tt.store %q3, %h32 : tensor<4x!tt.ptr<i32>>
    tt.store %q4, %m32 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
"""


@pytest.mark.parametrize("category", CATS)
def test_signed_operations_on_unsigned_storage(category):
    """i8 values read from uint8 / int8 storage: comparisons, division, arithmetic shift, max and sign extension follow
    the signed interpretation of the bit pattern, whatever the storage's representation."""
    raw = {"positive": [200, 1, 0, 128], "boundary": [255, 127, 128, 0], "premise_violation": [200, 3, 9, 128]}[category]
    dt = "int8" if category == "premise_violation" else "uint8"      # the same bytes through a signed storage
    arr = np.asarray(raw, np.uint8)
    ref, ids = _eval_ttir(REP, {"x": (dt, arr.view(np.int8) if dt == "int8" else arr),
                                "o": ("int32", np.zeros(20, np.int32))})
    buf = ref.buffers[ids["o"]]
    s = [int(v) - 256 if v >= 128 else int(v) for v in raw]
    want = ([int(v == -56) for v in s] + [int(v < 1) for v in s] + [int(np.trunc(v / 3)) for v in s] +
            [v >> 1 for v in s] + [max(v, 1) for v in s])
    assert (buf.st[:20] == H.ST_OK).all() and [int(v) for v in buf.lo[:20]] == want


CAS = ("    pid = tl.program_id(0)\n    c = tl.load(c_ptr + pid)\n    n = tl.load(n_ptr + pid)\n"
       "    old = tl.atomic_cas(z + pid, c, n)\n    tl.store(out + pid, old)\n")


@pytest.mark.parametrize("category", CATS)
def test_compare_and_swap_compares_integers_exactly(category):
    """uint32 storage holding values >= 2^31 and int64 values that float64 cannot tell apart."""
    from test_signatures_int_sets import _run
    if category == "positive":       # uint32 0xFFFFFFFE against the i32 value -2 (the same bits): swaps
        dt, z, c, n, want_old, want_z = "uint32", 0xFFFFFFFE, 0xFFFFFFFE, 5, 0xFFFFFFFE, 5
    elif category == "boundary":     # int64: 2^62 + 1 vs 2^62 (equal in float64): must not swap
        dt, z, c, n, want_old, want_z = "int64", (1 << 62) + 1, 1 << 62, 7, (1 << 62) + 1, (1 << 62) + 1
    else:                            # int64: an exact match of a large value: swaps
        dt, z, c, n, want_old, want_z = "int64", (1 << 62) + 1, (1 << 62) + 1, 7, (1 << 62) + 1, 7
    bufs = {"z": (dt, [z]), "c_ptr": (dt, [c]), "n_ptr": (dt, [n]), "out": (dt, [0])}
    ref, ids = _run("cas_exact_" + dt, CAS, bufs, (1, 1, 1))
    width = 32 if dt == "uint32" else 64
    w = lambda v: ((int(v) + (1 << (width - 1))) & ((1 << width) - 1)) - (1 << (width - 1))
    zb, ob = ref.buffers[ids["z"]], ref.buffers[ids["out"]]
    assert ob.st[0] == H.ST_OK and w(ob.lo[0]) == w(want_old), (int(ob.lo[0]), want_old)
    assert zb.st[0] == H.ST_OK and w(zb.lo[0]) == w(want_z), (int(zb.lo[0]), want_z)


def test_integer_undefined_behavior_has_a_reason():
    ttir = REP.replace("%d = arith.divsi %v, %three", "%d = arith.remsi %v, %c")   # divisor -56: fine
    ttir = ttir.replace("%c = arith.constant dense<-56> : tensor<4xi8>", "%c = arith.constant dense<-1> : tensor<4xi8>")
    ref, ids = _eval_ttir(ttir, {"x": ("int8", np.asarray([-128, 5, 0, 7], np.int8)),
                                 "o": ("int32", np.zeros(20, np.int32))})
    buf = ref.buffers[ids["o"]]
    assert buf.st[8] != H.ST_OK and (buf.st[9:12] == H.ST_OK).all()          # -128 % -1: undefined behavior
    assert any("INT_MIN / -1" in r for r in ref.reasons)
