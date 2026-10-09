"""Blackwell scaled tcgen05 MMA (DSL v2 increment 13, rc3 04 W3; device validation pending): ttng.tc_gen5_mma_scaled
in the sm_100 TTGIR of the official kernels in tests/data/nvidia_ttgir (README there).  Each TTGIR reference is checked
against exact rationals and against the reference of the same kernel's TTIR (tt.dot_scaled).  Synthetic captures,
CPU only."""
from __future__ import annotations

from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest

import signature_harness as H

DATA = Path(__file__).parent / "data" / "nvidia_ttgir"
CATS = ("positive", "boundary", "premise_violation")
DT = {np.dtype(np.float32): "float32", np.dtype(np.uint8): "uint8"}
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ir(name, level):
    return (DATA / f"{name}.{level}").read_text()


def _eval(asm, bufs, scalars=()):
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    args, base = [], 1 << 24
    for i, (name, spec) in enumerate(bufs.items()):
        dtype, arr = spec if isinstance(spec, tuple) else (DT[np.asarray(spec).dtype], spec)
        arr = np.ascontiguousarray(arr)
        raw = arr.reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type=None, dtype=dtype,
                                shape=arr.shape, stride=None, element_size=arr.itemsize, data_ptr=base * (i + 1),
                                storage_ptr=base * (i + 1), storage_nbytes=raw.size, storage_id=i, before=raw,
                                after=raw.copy()))
    for name, value in scalars:
        args.append(CapturedArg(index=len(args), name=name, kind="int", constexpr=False, signature_type="i32",
                                value=value))
    launch = CapturedLaunch(index=0, kernel_name="k", kernel_hash="", grid=(1, 1, 1), args=args, asm=asm,
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = {}
    for i, name in enumerate(bufs):
        b = ref.buffers[base * (i + 1)]
        st = np.where(np.asarray(b.written), np.asarray(b.st), H.ST_NE)
        out[name] = (np.asarray(b.lo), np.asarray(b.hi if b.hi is not None else b.lo), st)
    return out, ref


def _both(name, bufs, scalars=(), ttgir=None, out_name="out"):
    g, gref = _eval({"ttgir": ttgir or _ir(name, "sm100.ttgir")}, bufs, scalars)
    t, _ = _eval({"ttir": _ir(name, "ttir"), "ttgir": _ir(name, "sm100.ttgir")}, bufs, scalars)
    lo1, hi1, s1 = g[out_name]
    lo2, hi2, s2 = t[out_name]
    both = (s1 == H.ST_OK) & (s2 == H.ST_OK)
    assert not ((hi1[both] < lo2[both]) | (hi2[both] < lo1[both])).any()
    return g, gref


def _f8(raw, fmt):
    import torch
    t = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    return torch.from_numpy(np.ascontiguousarray(raw, dtype=np.uint8)).view(t).to(torch.float64).numpy()


def _scales(rng, shape, category):
    s = rng.integers(122, 133, shape).astype(np.uint8)
    if category == "boundary":
        s[0, 0], s[0, 1] = 0, 254                                # 2^-127 and 2^127
    if category == "premise_violation":
        s[0, 0] = 255                                            # the E8M0 NaN marker
    return s


@pytest.mark.parametrize("category", CATS)
def test_scaled_mma_fp8_signature(category):
    rng = np.random.default_rng(11)
    a = rng.integers(0, 0x7F, (128, 64)).astype(np.uint8)        # e4m3fn without the NaN codes
    b = rng.integers(0x80, 0xFF, (64, 128)).astype(np.uint8)     # negative e4m3fn values
    if category == "boundary":
        a[1, :], b[:, 1] = 0x7E, 0xFE                            # 448 and -448, the largest finite e4m3fn
    sa, sb = _scales(rng, (128, 2), category), rng.integers(124, 131, (128, 2)).astype(np.uint8)
    bufs = {"a_base": a, "b_base": b, "a_scale": sa, "b_scale": sb, "out": np.zeros(128 * 128, np.float32)}
    assert "ttng.tc_gen5_mma_scaled" in _ir("simple_dot_mxfp", "sm100.ttgir")
    out, ref = _both("simple_dot_mxfp", bufs)
    assert ref.rules.get("nvidia.tc_gen5_mma_scaled", 0) == 1
    lo, hi, st = out["out"]
    av, bv = _f8(a, "e4m3"), _f8(b, "e4m3")
    for i, j in ((0, 0), (0, 9), (1, 1), (5, 77), (127, 127)):
        e = i * 128 + j
        if category == "premise_violation" and i == 0:
            assert st[e] != H.ST_OK
            continue
        exact = sum(Fr(float(av[i, k])) * Fr(2) ** (int(sa[i, k // 32]) - 127) *
                    Fr(float(bv[k, j])) * Fr(2) ** (int(sb[j, k // 32]) - 127) for k in range(64))
        assert st[e] == H.ST_OK and Fr(float(lo[e])) <= exact <= Fr(float(hi[e])), (i, j)


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("variant,n,stages", [("mxfp4_matmul_n128_s1", 128, 1), ("mxfp4_matmul_n128_s3", 128, 3),
                                              ("mxfp4_matmul_n256_s1", 256, 1), ("mxfp4_matmul_n256_s2", 256, 2)])
def test_scaled_mma_fp4_pipelines_signature(variant, n, stages, category):
    rng = np.random.default_rng(12)
    M, K = 128, 256 * max(2, stages)                             # enough k-tiles for the pipelined loop to iterate
    a = rng.integers(0, 0x7C, (M, K)).astype(np.uint8)           # e5m2 finite codes
    b = rng.integers(0, 256, (n, K // 2)).astype(np.uint8)       # [N, K/2], two e2m1 per byte along K, low first
    if category == "boundary":
        a[2, :] = 0x7B                                           # 57344, the largest finite e5m2
    sa = rng.integers(124, 131, (M, K // 32)).astype(np.uint8)
    sb = _scales(rng, (n, K // 32), category)
    bufs = {"a_ptr": ("float8_e5m2", a), "b_ptr": b, "output_ptr": np.zeros(M * n, np.float32), "a_scale": sa,
            "b_scale": sb}
    scalars = [("M", M), ("N", n), ("K", K), ("stride_scale", K // 32), ("stride_am", K), ("stride_cm", n)]
    out, ref = _both(variant, bufs, scalars, out_name="output_ptr")
    assert ref.rules.get("nvidia.tc_gen5_mma_scaled", 0) == K // 256
    lo, hi, st = out["output_ptr"]
    av = _f8(a, "e5m2")

    def bval(col, k):
        byte = int(b[col, k // 2])
        return Fr(E2M1[(byte >> 4) if k % 2 else (byte & 0xF)])
    for i, j in ((0, 0), (2, 0), (2, 5), (64, n - 1), (127, 3)):
        e = i * n + j
        if category == "premise_violation" and j == 0:
            assert st[e] != H.ST_OK                              # column 0's first scale block is NaN
            continue
        exact = sum(Fr(float(av[i, k])) * Fr(2) ** (int(sa[i, k // 32]) - 127) *
                    bval(j, k) * Fr(2) ** (int(sb[j, k // 32]) - 127) for k in range(K))
        assert st[e] == H.ST_OK and Fr(float(lo[e])) <= exact <= Fr(float(hi[e])), (i, j)


def test_scaled_mma_result_needs_the_barrier_wait():
    """Variant: the barrier wait after the asynchronous scaled MMA is removed; the tensor-memory accumulator is read
    before its completion is observed, so nothing stored from it is established."""
    rng = np.random.default_rng(13)
    lines = [ln for ln in _ir("simple_dot_mxfp", "sm100.ttgir").splitlines() if "ttng.wait_barrier" not in ln]
    bufs = {"a_base": rng.integers(0, 0x7F, (128, 64)).astype(np.uint8),
            "b_base": rng.integers(0, 0x7F, (64, 128)).astype(np.uint8),
            "a_scale": np.full((128, 2), 127, np.uint8), "b_scale": np.full((128, 2), 127, np.uint8),
            "out": np.zeros(128 * 128, np.float32)}
    out, ref = _eval({"ttgir": "\n".join(lines)}, bufs)
    assert ref.rules.get("nvidia.tc_gen5_mma_scaled", 0) == 1
    assert (out["out"][2] != H.ST_OK).all()


@pytest.mark.parametrize("key", ["two_ctas", "multicast"])
def test_cta_pair_mma_is_not_evaluated_as_one_cta(key):
    """Variant: the scaled MMA carries two_ctas / multicast (operands distributed over a CTA pair): not modelled, the
    program aborts with the reason instead of computing a one-CTA product."""
    ttgir = _ir("simple_dot_mxfp", "sm100.ttgir").replace("{is_async}", "{is_async, %s}" % key)
    assert key in ttgir
    bufs = {"a_base": np.zeros((128, 64), np.uint8), "b_base": np.zeros((64, 128), np.uint8),
            "a_scale": np.full((128, 2), 127, np.uint8), "b_scale": np.full((128, 2), 127, np.uint8),
            "out": np.zeros(128 * 128, np.float32)}
    out, ref = _eval({"ttgir": ttgir}, bufs)
    assert any(key in r for r in ref.aborted.values())
    assert (out["out"][2] != H.ST_OK).all()
