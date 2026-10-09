"""Atomic load / store / poll, histogram and approximate division (DSL v2 increment 5, rc3 02 5.3, 6.5, 6.8, 6.9).

The installed Triton 3.6.0 does not emit tt.atomic_load / atomic_store / atomic_poll / approx_divf, so these tests
evaluate TTIR text: kernels marked "official main" are the TTIR the official main build (e50b186e) produced for the
official unit tests (.cache/dsl_v2/w0_dump, location annotations removed); the others are written by hand in the same
syntax and say so.  Synthetic captures, CPU only.  Expected values are computed independently (exact integers,
enumerated interleavings, mpmath)."""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

import signature_harness as H

CATS = ("positive", "boundary", "premise_violation")
NP = {"i32": np.int32, "i64": np.int64, "f32": np.float32, "i16": np.int16, "i8": np.int8}
TORCH = {"i32": "int32", "i64": "int64", "f32": "float32", "i16": "int16", "i8": "int8"}


def _eval(ttir, bufs, grid=(1, 1, 1)):
    """bufs: ordered {param name: (elem, array)}; returns {name: (lo, hi, st)} and the launch reference."""
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    args, base = [], 1 << 20
    for i, (name, (elem, arr)) in enumerate(bufs.items()):
        arr = np.ascontiguousarray(np.asarray(arr, NP[elem]))
        raw = arr.reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type="*" + elem,
                                dtype=TORCH[elem], shape=arr.shape, stride=None, element_size=arr.itemsize,
                                data_ptr=base * (i + 1), storage_ptr=base * (i + 1), storage_nbytes=raw.size,
                                storage_id=i, before=raw, after=raw.copy()))
    launch = CapturedLaunch(index=0, kernel_name="k", kernel_hash="", grid=grid, args=args, asm={"ttir": ttir, "ttgir": ""},
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = {}
    for i, name in enumerate(bufs):
        b = ref.buffers[base * (i + 1)]
        st = np.where(np.asarray(b.written), np.asarray(b.st), H.ST_NE)
        out[name] = (np.asarray(b.lo), np.asarray(b.hi if b.hi is not None else b.lo), st)
    return out, ref


# ---------------------------------------------------------------- atomic load / store

def _load_store_ttir(t):
    # hand-written, the shape of the official test_atomic_load_store (vector, odd lanes)
    return f"""module {{
  tt.func public @kernel(%src: !tt.ptr<{t}>, %dst: !tt.ptr<{t}>) {{
    %c2 = arith.constant dense<2> : tensor<8xi32>
    %c1 = arith.constant dense<1> : tensor<8xi32>
    %offs = tt.make_range {{end = 8 : i32, start = 0 : i32}} : tensor<8xi32>
    %r = arith.remsi %offs, %c2 : tensor<8xi32>
    %mask = arith.cmpi eq, %r, %c1 : tensor<8xi32>
    %s = tt.splat %src : !tt.ptr<{t}> -> tensor<8x!tt.ptr<{t}>>
    %sp = tt.addptr %s, %offs : tensor<8x!tt.ptr<{t}>>, tensor<8xi32>
    %v = tt.atomic_load acquire, gpu, %sp, %mask : (tensor<8x!tt.ptr<{t}>>, tensor<8xi1>) -> tensor<8x{t}>
    %d = tt.splat %dst : !tt.ptr<{t}> -> tensor<8x!tt.ptr<{t}>>
    %dp = tt.addptr %d, %offs : tensor<8x!tt.ptr<{t}>>, tensor<8xi32>
    tt.atomic_store release, gpu, %dp, %v, %mask : tensor<8x!tt.ptr<{t}>>
    tt.return
  }}
}}"""


# official main: test_atomic_load_store[int16-scalar] (offset 1)
SCALAR_LOAD_STORE = """module {
  tt.func public @kernel(%src: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offsets = arith.constant 1 : i32
    %mask = arith.constant true
    %value = tt.addptr %src, %offsets : !tt.ptr<i16>, i32
    %value_0 = tt.atomic_load acquire, gpu, %value, %mask : (!tt.ptr<i16>, i1) -> i16
    %0 = tt.addptr %dst, %offsets : !tt.ptr<i16>, i32
    tt.atomic_store release, gpu, %0, %value_0, %mask : !tt.ptr<i16>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_atomic_load_store_signature(category):
    if category == "positive":
        src = np.arange(8, dtype=np.int32) * 3 - 7
        out, _ = _eval(_load_store_ttir("i32"), {"src": ("i32", src), "dst": ("i32", np.full(8, 42))})
        lo, hi, st = out["dst"]
        for i in range(8):
            if i % 2:
                assert st[i] == H.ST_OK and int(lo[i]) == int(src[i])
            else:                                  # masked off: not written, the captured 42 stays
                assert st[i] == H.ST_NE and int(lo[i]) == 42
    elif category == "boundary":
        src = np.asarray([0.0, 3.4e38, -0.0, 1e-45, -3.4e38, 1.0, 2.0, -1e-45], np.float32)
        out, _ = _eval(_load_store_ttir("f32"), {"src": ("f32", src), "dst": ("f32", np.zeros(8))})
        lo, hi, st = out["dst"]
        for i in range(1, 8, 2):
            assert st[i] == H.ST_OK and lo[i] == hi[i] == src[i]
        out, _ = _eval(SCALAR_LOAD_STORE, {"src": ("i16", [5, -32768, 7]), "dst": ("i16", [1, 2, 3])})
        lo, hi, st = out["dst"]
        assert st[1] == H.ST_OK and int(lo[1]) == -32768
    else:
        # two programs store different values to the same addresses: the final value is set-valued
        ttir = _two_program_store("f32")
        out, ref = _eval(ttir, {"val": ("f32", [1.5, -2.25]), "x": ("f32", [9.0])}, grid=(2, 1, 1))
        lo, hi, st = out["x"]
        assert st[0] == H.ST_OK and lo[0] <= -2.25 and 1.5 <= hi[0]
        assert any(r.startswith("set:") for r in ref.reasons)
        out, _ = _eval(_two_program_store("i32"), {"val": ("i32", [1, 2]), "x": ("i32", [9])}, grid=(2, 1, 1))
        assert out["x"][2][0] != H.ST_OK


def _two_program_store(t):
    # hand-written: every program atomically stores val[pid] to x
    return f"""module {{
  tt.func public @kernel(%val: !tt.ptr<{t}>, %x: !tt.ptr<{t}>) {{
    %true = arith.constant true
    %pid = tt.get_program_id x : i32
    %vp = tt.addptr %val, %pid : !tt.ptr<{t}>, i32
    %v = tt.load %vp : !tt.ptr<{t}>
    tt.atomic_store relaxed, gpu, %x, %v, %true : !tt.ptr<{t}>
    tt.return
  }}
}}"""


def _store_then_load(t):
    # hand-written: program 0 atomically stores val to x, program 1 atomically loads x into out
    return f"""module {{
  tt.func public @kernel(%val: !tt.ptr<{t}>, %x: !tt.ptr<{t}>, %out: !tt.ptr<{t}>) {{
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %is0 = arith.cmpi eq, %pid, %c0 : i32
    scf.if %is0 {{
      %v = tt.load %val : !tt.ptr<{t}>
      tt.atomic_store relaxed, gpu, %x, %v, %true : !tt.ptr<{t}>
    }} else {{
      %r = tt.atomic_load relaxed, gpu, %x, %true : (!tt.ptr<{t}>, i1) -> {t}
      tt.store %out, %r : !tt.ptr<{t}>
    }}
    tt.return
  }}
}}"""


def test_atomic_load_of_a_concurrently_stored_address_is_a_set():
    out, ref = _eval(_store_then_load("f32"), {"val": ("f32", [4.5]), "x": ("f32", [-1.0]), "out": ("f32", [0.0])},
                     grid=(2, 1, 1))
    lo, hi, st = out["out"]
    assert st[0] == H.ST_OK and lo[0] <= -1.0 and 4.5 <= hi[0]          # either the old or the stored value
    assert not any("race" in r for r in ref.reasons)                     # atomic against atomic: no race
    out, _ = _eval(_store_then_load("i32"), {"val": ("i32", [4]), "x": ("i32", [-1]), "out": ("i32", [0])},
                   grid=(2, 1, 1))
    assert out["out"][2][0] != H.ST_OK                                  # integer set: not a point
    out, _ = _eval(_store_then_load("i32"), {"val": ("i32", [4]), "x": ("i32", [4]), "out": ("i32", [0])},
                   grid=(2, 1, 1))
    assert out["out"][2][0] == H.ST_OK and int(out["out"][0][0]) == 4  # every interleaving reads 4


# ---------------------------------------------------------------- atomic poll

# official main: test_atomic_poll_waits_for_remote_cta
REMOTE_CTA = """module {
  tt.func public @kernel(%flag: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %payload: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c42_i32 = arith.constant 42 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = tt.atomic_poll acquire, gpu, %flag, %c1_i32 : !tt.ptr<i32>, i32 -> i1
      %3 = tt.load %payload : !tt.ptr<i32>
      tt.store %out, %3 : !tt.ptr<i32>
    } else {
      tt.store %payload, %c42_i32 : !tt.ptr<i32>
      %2 = tt.atomic_rmw exch, release, gpu, %flag, %c1_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
    }
    tt.return
  }
}"""


def _poll_tensor(timeout):
    # hand-written, the shape of the official test_atomic_poll_tensor_results (block 8)
    t = " timeout %t" if timeout else ""
    tdef = "    %t = arith.constant 0 : i64\n" if timeout else ""
    return f"""module {{
  tt.func public @kernel(%flags: !tt.ptr<i32>, %out: !tt.ptr<i8>) {{
{tdef}    %c1 = arith.constant dense<1> : tensor<8xi32>
    %offs = tt.make_range {{end = 8 : i32, start = 0 : i32}} : tensor<8xi32>
    %e = arith.addi %offs, %c1 : tensor<8xi32>
    %f = tt.splat %flags : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %fp = tt.addptr %f, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %m = tt.atomic_poll acquire, gpu, %fp, %e{t} : tensor<8x!tt.ptr<i32>>, tensor<8xi32> -> tensor<8xi1>
    %o = tt.splat %out : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %op = tt.addptr %o, %offs : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %mi = arith.extui %m : tensor<8xi1> to tensor<8xi8>
    tt.store %op, %mi : tensor<8x!tt.ptr<i8>>
    tt.return
  }}
}}"""


@pytest.mark.parametrize("category", CATS)
def test_atomic_poll_signature(category):
    flags = np.arange(1, 9, dtype=np.int32)
    if category == "positive":
        out, _ = _eval(_poll_tensor(False), {"flags": ("i32", flags), "out": ("i8", np.zeros(8))})
        lo, hi, st = out["out"]
        assert (st == H.ST_OK).all() and (lo == 1).all()
    elif category == "boundary":
        flags[::2] = 0                            # with a zero timeout: matched exactly where flag == expected
        out, _ = _eval(_poll_tensor(True), {"flags": ("i32", flags), "out": ("i8", np.zeros(8))})
        lo, hi, st = out["out"]
        assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(f == i + 1) for i, f in enumerate(flags)]
    else:
        flags[3] = 0                              # no timeout, never matched, nobody writes it: no termination
        out, ref = _eval(_poll_tensor(False), {"flags": ("i32", flags), "out": ("i8", np.zeros(8))})
        assert (out["out"][2] != H.ST_OK).all() and ref.aborted


def test_message_passing_through_a_poll_orders_the_payload():
    out, ref = _eval(REMOTE_CTA, {"flag": ("i32", [0]), "payload": ("i32", [0]), "out": ("i32", [0])},
                     grid=(2, 1, 1))
    lo, hi, st = out["out"]
    assert st[0] == H.ST_OK and int(lo[0]) == 42
    assert any("assumed:atomic_poll terminates" in r for r in ref.reasons)
    assert not any("race" in r for r in ref.reasons)
    assert ref.rules.get("execution.happens_before_edges") == 1
    assert out["flag"][2][0] == H.ST_OK and int(out["flag"][0][0]) == 1
    # a relaxed poll does not acquire: the payload read races with the write
    out, ref = _eval(REMOTE_CTA.replace("atomic_poll acquire", "atomic_poll relaxed"),
                     {"flag": ("i32", [0]), "payload": ("i32", [0]), "out": ("i32", [0])}, grid=(2, 1, 1))
    assert out["out"][2][0] != H.ST_OK


def test_poll_dependency_cycle_is_not_established_and_does_not_hang():
    # hand-written: program 0 waits for a (set by program 1 after it saw b), program 1 waits for b (set by program 0
    # after it saw a): a deadlock on any device
    ttir = """module {
  tt.func public @kernel(%a: !tt.ptr<i32>, %b: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %c7 = arith.constant 7 : i32
    %c0 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %is0 = arith.cmpi eq, %pid, %c0 : i32
    scf.if %is0 {
      %p = tt.atomic_poll acquire, gpu, %a, %c1 : !tt.ptr<i32>, i32 -> i1
      tt.atomic_store release, gpu, %b, %c1, %true : !tt.ptr<i32>
    } else {
      %p = tt.atomic_poll acquire, gpu, %b, %c1 : !tt.ptr<i32>, i32 -> i1
      tt.atomic_store release, gpu, %a, %c1, %true : !tt.ptr<i32>
    }
    %o = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    tt.store %o, %c7 : !tt.ptr<i32>
    tt.return
  }
}"""
    out, ref = _eval(ttir, {"a": ("i32", [0]), "b": ("i32", [0]), "out": ("i32", [0, 0])}, grid=(2, 1, 1))
    assert (out["out"][2] != H.ST_OK).all() and len(ref.aborted) == 2


# ---------------------------------------------------------------- histogram

# official main: test_histogram_mask (2048 inputs, mask offset < 1024, 8 bins)
HISTOGRAM = """module {
  tt.func public @histogram_kernel(%x_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %z_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %mask = arith.constant dense<1024> : tensor<2048xi32>
    %offset1 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %offset2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %mask_0 = arith.cmpi slt, %offset1, %mask : tensor<2048xi32>
    %x = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    %x_1 = tt.addptr %x, %offset1 : tensor<2048x!tt.ptr<i32>>, tensor<2048xi32>
    %x_2 = tt.load %x_1 : tensor<2048x!tt.ptr<i32>>
    %z = tt.histogram %x_2, %mask_0 : tensor<2048xi32> -> tensor<8xi32>
    %0 = tt.splat %z_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %1 = tt.addptr %0, %offset2 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %1, %z : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_histogram_signature(category):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 8, 2048).astype(np.int32)
    if category == "boundary":
        x[:16] = 0
        x[16:32] = 7
        x[32:40] = [8, 9, -1, -100, 2 ** 31 - 1, -2 ** 31, 8, 8]  # counted positions outside [0, 8): dropped
        x[1024:] = rng.integers(-100, 100, 1024)  # masked off: never counted, whatever the value
    if category == "premise_violation":
        # an input without an established value: the official test kernel loads it from memory, so make the load
        # read an element the reference cannot establish (a NaN-free int has a value; use a masked load instead)
        out, _ = _eval(HISTOGRAM.replace("%x_2 = tt.load %x_1 : tensor<2048x!tt.ptr<i32>>",
                                         "%x_2 = tt.load %x_1, %mask_0 : tensor<2048x!tt.ptr<i32>>")
                       .replace("tt.histogram %x_2, %mask_0", "tt.histogram %x_2"),
                       {"x_ptr": ("i32", x), "z_ptr": ("i32", np.zeros(8))})
        assert (out["z_ptr"][2] != H.ST_OK).all()   # lanes 1024.. are undefined (masked load without other)
        return
    out, _ = _eval(HISTOGRAM, {"x_ptr": ("i32", x), "z_ptr": ("i32", np.zeros(8))})
    lo, hi, st = out["z_ptr"]
    expect = [int((x[:1024] == b).sum()) for b in range(8)]
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == expect


# ---------------------------------------------------------------- approximate division

# official main: test_approx_divf (128 lanes)
APPROX_DIV = """module {
  tt.func public @kernel(%X: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Y: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %x = tt.splat %X : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %x_0 = tt.addptr %x, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %x_1 = tt.load %x_0 : tensor<128x!tt.ptr<f32>>
    %y = tt.splat %Y : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %y_2 = tt.addptr %y, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %y_3 = tt.load %y_2 : tensor<128x!tt.ptr<f32>>
    %0 = tt.splat %Z : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %2 = tt.approx_divf %x_1, %y_3 : tensor<128xf32>
    tt.store %1, %2 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("form", ["approx_divf", "div.full.f32"])
def test_approximate_division_signature(category, form):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(128).astype(np.float32)
    y = (rng.standard_normal(128) + 3.0).astype(np.float32)
    if category == "boundary":
        x[:4] = [0.0, -0.0, 3.4e38, 1e-45]
        y[:4] = [2.0 ** -126, 2.0 ** 126, 3.0, 1.0]
    if category == "premise_violation":
        y[:4] = 0.0                                 # x / 0: no real value
    ttir = APPROX_DIV
    if form == "div.full.f32":
        ttir = APPROX_DIV.replace(
            "tt.approx_divf %x_1, %y_3 : tensor<128xf32>",
            'tt.elementwise_inline_asm "div.full.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, '
            'pure = true} %x_1, %y_3 : tensor<128xf32>, tensor<128xf32> -> tensor<128xf32>')
    out, _ = _eval(ttir, {"X": ("f32", x), "Y": ("f32", y), "Z": ("f32", np.zeros(128))})
    lo, hi, st = out["Z"]
    exact = [None if float(b) == 0.0 else mp.mpf(float(a)) / mp.mpf(float(b)) for a, b in zip(x, y)]
    bad = H.check_lanes(category, lo, hi, st, exact)
    assert not bad, bad[:4]
