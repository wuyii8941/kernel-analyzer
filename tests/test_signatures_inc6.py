"""DSL v2 increment 6: argmax / argmin over NaN along the lowering tree, bit casts to and from bf16 / fp8, and tuple
arguments in the capture.  Compiled-only kernels on synthetic captures (CPU only), except the capture test, which
launches a tiny kernel on the GPU when one is present."""
from __future__ import annotations

import math

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import _run  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
NAN, INF = float("nan"), float("inf")


def _official_combine(is_max, tie_break_left):
    """standard.py _argmax_combine / _argmin_combine (3.6.0 and official main are the same), on Python floats."""
    def comb(a, b):
        (v1, i1), (v2, i2) = a, b
        tie = tie_break_left and v1 == v2 and i1 < i2
        better = (v1 > v2) if is_max else (v1 < v2)
        take_first = better or tie
        return (v1, i1) if take_first else (v2, i2)
    return comb


def _lane_butterfly(combine, leaves):
    """The 3.6.0 lowering for 8 elements, 1 warp: one element per lane on lanes 0..7, lanes 8..31 replicate
    lanes 0..7 (the blocked layout wraps), then the butterfly combine(own, lane ^ stride) for stride 16 .. 1.
    Returns the result every lane holds (a non-commutative combine leaves different lanes with different results)."""
    lanes = [leaves[l % len(leaves)] for l in range(32)]
    stride = 16
    while stride >= 1:
        lanes = [combine(lanes[l], lanes[l ^ stride]) for l in range(32)]
        stride //= 2
    return lanes


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("fn", ["argmax", "argmin"])
def test_arg_reduction_over_nan_follows_the_lowering_tree(category, fn):
    x = {"positive": [NAN, 6.0, 8.0, 1.0, 2.0, 3.0, -1.0, 0.5],
         "boundary": [1.0, NAN, NAN, 8.0, 8.0, -INF, INF, 0.0],
         "premise_violation": [NAN] * 8}[category]
    body = (f"    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    r = tl.{fn}(x, axis=0)\n"
            "    tl.store(out + tl.arange(0, 1), r + tl.zeros([1], dtype=r.dtype))\n")
    lo, hi, st = _run(f"nan_{fn}", body, {"x_ptr": ("fp32", np.asarray(x, np.float32)), "out": ("int32", np.zeros(1))})
    comb = _official_combine(fn == "argmax", True)
    lanes = {r[1] for r in _lane_butterfly(comb, [(float(v), i) for i, v in enumerate(x)])}
    if len(lanes) == 1:                # every lane holds the same index: established and equal to it
        assert st[0] == H.ST_OK and int(lo[0]) == lanes.pop(), (int(lo[0]), lanes)
    else:                              # which lane's result is consumed is not modelled: not established (sound)
        assert st[0] != H.ST_OK, (int(lo[0]), lanes)


@pytest.mark.parametrize("category", CATS)
def test_bitcast_bf16_and_fp8_signature(category):
    import torch
    if category == "positive":
        f = torch.tensor([1.0, -2.5, 0.15625, 3.0, -0.0, 7.5, 0.5, 1024.0], dtype=torch.bfloat16)
    elif category == "boundary":
        f = torch.tensor([3.3895e38, -3.3895e38, 1e-38, INF, -INF, 0.0, -0.0, 9.1835e-41], dtype=torch.bfloat16)
    else:
        f = torch.tensor([NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.bfloat16)
    bits = f.view(torch.int16).numpy()
    # int16 -> bf16 (bit cast) -> fp32: the value is the decoded bit pattern, NaN stays NaN
    body = ("    i = tl.arange(0, N)\n    b = tl.load(x_ptr + i)\n"
            "    tl.store(out + i, b.to(tl.bfloat16, bitcast=True).to(tl.float32))\n")
    lo, hi, st = _run("bc_bf16", body, {"x_ptr": ("int16", bits), "out": ("fp32", np.zeros(8))})
    for j, v in enumerate(f.float().numpy()):
        if math.isnan(v):
            assert st[j] == H.ST_NAN
        elif math.isinf(v):
            assert st[j] == (H.ST_PINF if v > 0 else H.ST_NINF)
        else:
            assert st[j] == H.ST_OK and lo[j] == hi[j] == v
    # fp8 e5m2: int8 -> fp8 (bit cast) -> fp32
    g = torch.tensor([1.0, -2.0, 0.25, 57344.0, -57344.0, 1.5258789e-05, 0.0, 3.0]).to(torch.float8_e5m2)
    if category == "premise_violation":
        g[0] = float("nan")
    b8 = g.view(torch.int8).numpy()
    body = ("    i = tl.arange(0, N)\n    b = tl.load(x_ptr + i)\n"
            "    tl.store(out + i, b.to(tl.float8e5, bitcast=True).to(tl.float32))\n")
    lo, hi, st = _run("bc_f8", body, {"x_ptr": ("int8", b8), "out": ("fp32", np.zeros(8))})
    for j, v in enumerate(g.float().numpy()):
        if math.isnan(v):
            assert st[j] == H.ST_NAN
        else:
            assert st[j] == H.ST_OK and lo[j] == hi[j] == v
    # bf16 -> int16 (bit cast) of an exactly representable point: definite bits
    body = ("    i = tl.arange(0, N)\n    v = tl.load(x_ptr + i)\n"
            "    tl.store(out + i, v.to(tl.bfloat16).to(tl.int16, bitcast=True))\n")
    vals = np.asarray([1.0, -2.5, 0.15625, 3.0, 0.0, 7.5, 0.5, 1024.0], np.float32)
    lo, hi, st = _run("bc_bf16_back", body, {"x_ptr": ("fp32", vals), "out": ("int16", np.zeros(8))})
    want = torch.tensor(vals).to(torch.bfloat16).view(torch.int16).numpy()
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(v) for v in want]


def test_tuple_arguments_are_flattened_like_the_frontend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to launch")
    import triton.language as tl

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, evaluate_sequence

    @triton.jit
    def k(out, shape, strides, ts):
        i = tl.arange(0, 8)
        a = tl.load(ts[0] + i)
        b = tl.load(ts[1] + i)
        tl.store(out + i * strides[0] + shape[1], a + b + shape[0])

    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    x = torch.arange(8, dtype=torch.int32, device="cuda")
    y = x * 10
    TritonLaunchRecorder.install_hook()
    rec = TritonLaunchRecorder()
    with rec:
        k[(1,)](out, (3, 5), (2, 1), (x, y))
        torch.cuda.synchronize()
    launch = rec.launches[-1]
    names = [(a.name, a.constexpr) for a in launch.args]
    assert ("shape.0", False) in names and ("ts.1", False) in names
    ref = evaluate_sequence([launch]).launches[0]
    buf = next(b for b in ref.buffers.values() if b.name == "out")
    w = np.asarray(buf.written)
    assert (np.asarray(buf.st)[w] == ST_OK).all()
    assert [int(v) for v in np.asarray(buf.lo)[w]] == [int(v) for v in out.cpu().numpy()[w]]


# ---------------------------------------------------------------- map_elementwise (official main TTIR)

# official main: test_map_elementwise (a three-way comparison with branches, pack 1)
MAP_CMP = """module {
  tt.func public @kernel(%X: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %Y: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %Z: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %x = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %x_0 = tt.splat %X : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %x_1 = tt.addptr %x_0, %x : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %x_2 = tt.load %x_1 : tensor<128x!tt.ptr<i32>>
    %y = tt.splat %Y : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %y_3 = tt.addptr %y, %x : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %y_4 = tt.load %y_3 : tensor<128x!tt.ptr<i32>>
    %z = "tt.map_elementwise"(%x_2, %y_4) <{pack = 1 : i32}> ({
    ^bb0(%z_5: i32, %z_6: i32):
      %2 = arith.cmpi slt, %z_5, %z_6 : i32
      cf.cond_br %2, ^bb2(%c-1_i32 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %3 = arith.cmpi eq, %z_5, %z_6 : i32
      cf.cond_br %3, ^bb2(%c0_i32 : i32), ^bb2(%c1_i32 : i32)
    ^bb2(%4: i32):  // 3 preds: ^bb0, ^bb1, ^bb1
      tt.map_elementwise.return %4 : i32
    }) : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %0 = tt.splat %Z : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %1 = tt.addptr %0, %x : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %1, %z : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}"""

# official main: test_map_elementwise_pack (unsigned divide and remainder, pack 2, two results)
MAP_DIVREM = """module {
  tt.func public @kernel(%A: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %B: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %C: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %a = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %a_0 = tt.splat %A : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %a_1 = tt.addptr %a_0, %a : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    %a_2 = tt.load %a_1 : tensor<512x!tt.ptr<i32>>
    %b = tt.splat %B : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %b_3 = tt.addptr %b, %a : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    %b_4 = tt.load %b_3 : tensor<512x!tt.ptr<i32>>
    %0:2 = "tt.map_elementwise"(%a_2, %b_4) <{pack = 2 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32):
      %5 = arith.divui %arg4, %arg6 : i32
      %6 = arith.divui %arg5, %arg7 : i32
      %7 = arith.remui %arg4, %arg6 : i32
      %8 = arith.remui %arg5, %arg7 : i32
      tt.map_elementwise.return %5, %6, %7, %8 : i32, i32, i32, i32
    }) : (tensor<512xi32>, tensor<512xi32>) -> (tensor<512xi32>, tensor<512xi32>)
    %1 = tt.splat %C : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %2 = tt.addptr %1, %a : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    tt.store %2, %0#0 : tensor<512x!tt.ptr<i32>>
    %3 = tt.splat %D : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %4 = tt.addptr %3, %a : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    tt.store %4, %0#1 : tensor<512x!tt.ptr<i32>>
    tt.return
  }
}"""


@pytest.mark.parametrize("category", CATS)
def test_map_elementwise_signature(category):
    from test_signatures_sync import _eval
    rng = np.random.default_rng(9)
    x = rng.integers(-50, 50, 128).astype(np.int32)
    y = rng.integers(-50, 50, 128).astype(np.int32)
    if category == "boundary":
        x[:4] = [-2 ** 31, 2 ** 31 - 1, 7, -2 ** 31]
        y[:4] = [2 ** 31 - 1, -2 ** 31, 7, -2 ** 31]
    out, _ = _eval(MAP_CMP, {"X": ("i32", x), "Y": ("i32", y), "Z": ("i32", np.zeros(128))})
    lo, hi, st = out["Z"]
    assert (st == H.ST_OK).all()
    assert [int(v) for v in lo] == [(-1 if a < b else (0 if a == b else 1)) for a, b in zip(x, y)]
    a = rng.integers(0, 2 ** 31, 512).astype(np.int64)
    b = rng.integers(1, 1000, 512).astype(np.int64)
    if category == "boundary":
        a[:2], b[:2] = [2 ** 32 - 1, 0], [1, 7]               # the largest unsigned value, a zero numerator
    if category == "premise_violation":
        b[3] = 0                                               # unsigned division by zero: no value
    a32, b32 = (a & 0xFFFFFFFF).astype(np.uint32).view(np.int32), b.astype(np.uint32).view(np.int32)
    out, _ = _eval(MAP_DIVREM, {"A": ("i32", a32), "B": ("i32", b32), "C": ("i32", np.zeros(512)),
                                "D": ("i32", np.zeros(512))})
    for name, f in (("C", lambda p, q: p // q), ("D", lambda p, q: p % q)):
        lo, hi, st = out[name]
        for i in range(512):
            p, q = int(a[i]) & 0xFFFFFFFF, int(b[i])
            if q == 0:
                assert st[i] != H.ST_OK
            else:
                assert st[i] == H.ST_OK and int(lo[i]) % 2 ** 32 == f(p, q), (name, i)
