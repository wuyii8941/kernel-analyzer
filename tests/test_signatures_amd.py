"""AMD gfx9xx TTGIR modules (DSL v2 increment 12, rc3 04 W3; device validation pending): buffer load / store / atomics,
buffer load to shared memory, in-thread transpose, scaled upcasts, scheduling hints.  The fixtures in
tests/data/amd_ttgir are official TTIR and the TTGIR the official AMD stages produce from it (README there); variants
below change one operand or instruction and say so.  Each TTGIR reference is checked against values computed here
(exact rationals) and against the reference of the same kernel's TTIR.  Synthetic captures, CPU only."""
from __future__ import annotations

from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest

import signature_harness as H

DATA = Path(__file__).parent / "data" / "amd_ttgir"
CATS = ("positive", "boundary", "premise_violation")
DT = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16", np.dtype(np.int32): "int32",
      np.dtype(np.int8): "int8", np.dtype(np.uint8): "uint8"}


def _ir(name, level):
    return (DATA / f"{name}.{level}").read_text()


def _eval(ir, level, bufs, scalars=(), grid=(1, 1, 1)):
    """bufs: ordered {name: array} or {name: (torch dtype, raw array)} for formats numpy lacks (bf16, fp8)."""
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
    asm = {"ttgir": ir} if level == "ttgir" else {"ttir": ir[0], "ttgir": ir[1]}
    launch = CapturedLaunch(index=0, kernel_name="k", kernel_hash="", grid=grid, args=args, asm=asm,
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = {}
    for i, name in enumerate(bufs):
        b = ref.buffers[base * (i + 1)]
        st = np.where(np.asarray(b.written), np.asarray(b.st), H.ST_NE)
        out[name] = (np.asarray(b.lo), np.asarray(b.hi if b.hi is not None else b.lo), st)
    return out, ref


def _both(name, ttgir_level, bufs, scalars=(), ttgir=None):
    """TTGIR reference (optionally a variant) and TTIR reference on the same capture; where both are complete their
    enclosures intersect."""
    g, gref = _eval(ttgir or _ir(name, ttgir_level), "ttgir", bufs, scalars)
    # the TTIR launch carries the TTGIR of its own compilation (combination order of generic combine regions)
    t, _ = _eval((_ir(name, "ttir"), _ir(name, ttgir_level)), "ttir", bufs, scalars)
    for k in g:
        lo1, hi1, s1 = g[k]
        lo2, hi2, s2 = t[k]
        both = (s1 == H.ST_OK) & (s2 == H.ST_OK)
        assert not ((hi1[both] < lo2[both]) | (hi2[both] < lo1[both])).any(), k
    return g, gref


def _encloses(lo, hi, st, idx, exact):
    return st[idx] == H.ST_OK and Fr(float(lo[idx])) <= exact <= Fr(float(hi[idx]))


def _f8(raw, fmt):
    import torch
    t = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    return torch.from_numpy(np.ascontiguousarray(raw, dtype=np.uint8)).view(t).to(torch.float64).numpy()


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---- A: buffer_load with mask / other --------------------------------------------------------------------------------

@pytest.mark.parametrize("category", CATS)
def test_buffer_load_mask_other_signature(category):
    x = {"positive": np.array([0.5, -2.0, 1.25, 9.0], np.float32),
         # -inf in the tile; the masked-off lane holds +inf and must not count
         "boundary": np.array([-np.inf, 3.0, 2.5, np.inf], np.float32),
         "premise_violation": np.array([1.0, 2.0], np.float32)}[category]   # lane 2 lies outside the capture
    bufs = {"x_ptr": x, "val_ptr": np.zeros(1, np.float32), "idx_ptr": np.zeros(1, np.int32)}
    out, ref = _both("argmax_masked_other", "gfx942.ttgir", bufs)
    assert ref.rules.get("amd.buffer_load", 0) >= 1
    (vlo, vhi, vst), (ilo, _, ist) = out["val_ptr"], out["idx_ptr"]
    if category == "premise_violation":
        assert vst[0] != H.ST_OK and ist[0] != H.ST_OK
        # variant: the same load without `other`: the masked-off lane is undefined (as for tt.load), so the maximum
        # over the tile has no value
        x4 = np.array([0.5, -2.0, 1.25, 9.0], np.float32)
        ttgir = _ir("argmax_masked_other", "gfx942.ttgir").replace("[%offsets], %mask_0, %x :", "[%offsets], %mask_0 :")
        assert "%mask_0 :" in ttgir
        out, _ = _eval(ttgir, "ttgir", {"x_ptr": x4, "val_ptr": np.zeros(1, np.float32),
                                        "idx_ptr": np.zeros(1, np.int32)})
        assert out["val_ptr"][2][0] != H.ST_OK
        return
    want = float(np.max(x[:3]))
    assert vst[0] == H.ST_OK and vlo[0] == vhi[0] == want
    assert ist[0] == H.ST_OK and ilo[0] == int(np.argmax(x[:3]))


# ---- B: buffer_store with mask ---------------------------------------------------------------------------------------

def _bf16(vals):
    return (np.asarray(vals, np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("category", CATS)
def test_buffer_store_mask_signature(category):
    a = _bf16(np.linspace(-3, 3, 64))
    n = {"positive": 64, "boundary": 63, "premise_violation": 64}[category]
    if category == "boundary":
        a[:2] = _bf16([3.3895313892515355e38, -0.0])          # the largest finite bf16, negative zero
    src = a[:32] if category == "premise_violation" else a  # lanes 32..63 read outside the capture
    out_init = _bf16(np.full(64, 7.0))
    bufs = {"out_ptr": ("bfloat16", out_init), "a_ptr": ("bfloat16", src)}
    out, ref = _both("masked_copy_bf16", "gfx942.ttgir", bufs, scalars=[("shape.0", n)])
    assert ref.rules.get("amd.buffer_store", 0) >= 1
    lo, hi, st = out["out_ptr"]
    vals = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if category == "premise_violation":
        assert (st[:32] == H.ST_OK).all() and np.array_equal(lo[:32], vals[:32])
        assert (st[32:] != H.ST_OK).all()
        return
    assert (st[:n] == H.ST_OK).all() and np.array_equal(lo[:n], vals[:n])
    if n < 64:   # the masked-off lane is not written: it keeps the captured value
        assert not ref.buffers[1 << 24].written[63] and ref.buffers[1 << 24].lo[63] == 7.0


# ---- C: buffer atomics -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("category", CATS)
def test_buffer_atomic_rmw_signature(category):
    rng = np.random.default_rng(3)
    val = rng.standard_normal(16).astype(np.float32)
    X = rng.standard_normal(8).astype(np.float32)
    if category == "boundary":
        val[0::2], val[1::2] = 3.0e38, -3.0e38                  # two updates of one address cancel
        X[:] = -0.0
    if category == "premise_violation":
        val[4] = np.nan                                         # an update outside the finite domain
    out, ref = _both("atomic_fadd_pairs", "gfx942.ttgir", {"X": X, "val": val})
    assert ref.rules.get("amd.buffer_atomic_rmw", 0) >= 1
    lo, hi, st = out["X"]
    for k in range(8):
        if category == "premise_violation" and k == 2:
            assert st[k] != H.ST_OK                             # X[2] receives the NaN
            continue
        exact = Fr(float(X[k])) + Fr(float(val[2 * k])) + Fr(float(val[2 * k + 1]))
        assert _encloses(lo, hi, st, k, exact), k


@pytest.mark.parametrize("category", CATS)
def test_buffer_atomic_cas_signature(category):
    index = {"positive": np.zeros(16, np.int32),                # every row swaps in its own number
             "boundary": np.where(np.arange(16) % 2 == 0, 0, 3).astype(np.int32),   # odd rows compare unequal
             "premise_violation": np.zeros(8, np.int32)}[category]                 # rows 8..15 outside the capture
    out, ref = _both("atomic_cas_rows", "gfx942.ttgir", {"index_ptr": index, "out_ptr": np.zeros(256, np.int32)})
    lo, _, st = out["index_ptr"]
    if category == "premise_violation":
        # rows 8..15 return no old value; the store addressed by it may write any element of out_ptr
        assert (st[:8] == H.ST_OK).all() and (out["out_ptr"][2] != H.ST_OK).all()
        return
    assert ref.rules.get("amd.buffer_atomic_cas", 0) >= 1
    want = np.where(index == 0, np.arange(16), index)
    assert (st == H.ST_OK).all() and np.array_equal(lo, want)
    olo, _, ost = out["out_ptr"]
    rows = sorted(set(int(v) for v in index))                   # out[old * 16 + col] = 5 for the returned old values
    for r in rows:
        assert (ost[r * 16:(r + 1) * 16] == H.ST_OK).all() and (olo[r * 16:(r + 1) * 16] == 5).all()


# ---- D: buffer_load_to_local (asynchronous copy into shared memory) --------------------------------------------------

@pytest.mark.parametrize("category", CATS)
def test_buffer_load_to_local_signature(category):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((16, 128)).astype(np.float32)
    b = rng.standard_normal((128, 16)).astype(np.float32)
    if category == "boundary":
        a[0, :], b[:, 0] = 3.0e38, 1.0 / 1024
        a[1, :] = 0.0
    ttgir = _ir("gather_dot_pipeline", "gfx950.ttgir")
    if category == "premise_violation":
        # variant: the wait inside the pipelined loop is removed; the next iteration reads copies not yet observed
        lines = ttgir.splitlines()
        k = next(i for i, ln in enumerate(lines) if "ttg.async_wait" in ln)
        name = lines[k].split("=")[0].strip()
        ttgir = "\n".join(ln for i, ln in enumerate(lines) if i != k).replace(f"{name}", "%a_43")
    bufs = {"a_ptr": a, "b_ptr": b, "output_ptr": np.zeros(256, np.float32)}
    scalars = [("stride_am", 128), ("stride_bk", 16), ("stride_cm", 16)]
    out, ref = (_eval(ttgir, "ttgir", bufs, scalars) if category == "premise_violation"
                else _both("gather_dot_pipeline", "gfx950.ttgir", bufs, scalars))
    lo, hi, st = out["output_ptr"]
    if category == "premise_violation":
        assert (st != H.ST_OK).any()
        return
    assert ref.rules.get("amd.buffer_load_to_local", 0) >= 1
    for i, j in ((0, 0), (1, 5), (7, 3), (15, 15)):
        exact = sum(Fr(float(a[i, k])) * Fr(float(b[k, j])) for k in range(128))
        assert _encloses(lo, hi, st, i * 16 + j, exact), (i, j)


# ---- E: in_thread_transpose ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("category", CATS)
def test_in_thread_transpose_signature(category):
    rng = np.random.default_rng(5)
    a = (rng.standard_normal((64, 64)) * 2).astype(np.float16)
    b = (rng.standard_normal((64, 64)) * 2).astype(np.float16)
    if category == "boundary":
        a[0, :], b[:, 0] = 65504.0, 1.0 / 1024
    if category == "premise_violation":
        b[3, 7] = np.nan
    ttgir = _ir("simple_dot_transpose", "gfx942.ttgir")
    assert "amdg.in_thread_transpose" in ttgir
    out, ref = _both("simple_dot_transpose", "gfx942.ttgir",
                     {"a_base": a, "b_base": b, "out": np.zeros(64 * 64, np.float16)})
    lo, hi, st = out["out"]
    for i, j in ((0, 0), (0, 7), (5, 3), (40, 7), (63, 63)):
        e = i * 64 + j
        if category == "premise_violation" and j == 7:
            assert st[e] != H.ST_OK                             # column 7 meets the NaN of b
            continue
        exact = sum(Fr(float(a[i, k])) * Fr(float(b[k, j])) for k in range(64))
        assert _encloses(lo, hi, st, e, exact), (i, j)


# ---- F: scaled upcasts -----------------------------------------------------------------------------------------------

def _scale_inputs(rng, shape, category):
    s = rng.integers(120, 135, shape).astype(np.uint8)
    if category == "boundary":
        s.reshape(-1)[0], s.reshape(-1)[1] = 0, 254             # 2^-127 (bf16 carrier subnormal) and 2^127
    if category == "premise_violation":
        s.reshape(-1)[0] = 255                                  # the E8M0 NaN marker
    return s


@pytest.mark.parametrize("category", CATS)
def test_scaled_upcast_fp8_signature(category):
    rng = np.random.default_rng(6)
    a = rng.integers(0, 0x7F, (128, 64)).astype(np.uint8)       # e4m3fn without its NaN codes (0x7F, 0xFF)
    b = rng.integers(0, 0x7F, (64, 128)).astype(np.uint8)
    a[a == 0x7F], b[b == 0x7F] = 0x7E, 0x7E
    sa, sb = _scale_inputs(rng, (128, 2), category), rng.integers(124, 131, (128, 2)).astype(np.uint8)
    bufs = {"a_base": a, "b_base": b, "a_scale": sa, "b_scale": sb, "out": np.zeros(128 * 128, np.float32)}
    out, ref = _both("simple_dot_mxfp8", "gfx942.ttgir", bufs)
    assert ref.rules.get("amd.scaled_upcast_fp8", 0) >= 1
    lo, hi, st = out["out"]
    av, bv = _f8(a, "e4m3"), _f8(b, "e4m3")
    for i, j in ((0, 0), (0, 77), (3, 5), (64, 127), (127, 1)):
        e = i * 128 + j
        if category == "premise_violation" and i == 0:
            assert st[e] != H.ST_OK                             # row 0's first scale block is NaN
            continue
        exact = sum(Fr(float(av[i, k])) * Fr(2) ** (int(sa[i, k // 32]) - 127) *
                    Fr(float(bv[k, j])) * Fr(2) ** (int(sb[j, k // 32]) - 127) for k in range(64))
        assert _encloses(lo, hi, st, e, exact), (i, j)


@pytest.mark.parametrize("category", CATS)
def test_scaled_upcast_fp4_signature(category):
    rng = np.random.default_rng(7)
    a = rng.integers(0, 0x7C, (128, 256)).astype(np.uint8)      # e5m2 finite codes only (0x7C.. are inf / NaN)
    b = rng.integers(0, 256, (128, 128)).astype(np.uint8)       # two e2m1 values per byte, low nibble first
    sa, sb = rng.integers(124, 131, (128, 8)).astype(np.uint8), _scale_inputs(rng, (128, 8), category)
    bufs = {"a_ptr": ("float8_e5m2", a), "b_ptr": b, "output_ptr": np.zeros(128 * 128, np.float32),
            "a_scale": sa, "b_scale": sb}
    scalars = [("M", 128), ("N", 128), ("K", 256), ("stride_scale", 8), ("stride_am", 256), ("stride_cm", 128)]
    out, ref = _both("mxfp8_mxfp4_matmul", "gfx942.ttgir", bufs, scalars)
    assert ref.rules.get("amd.scaled_upcast_fp4", 0) >= 1
    lo, hi, st = out["output_ptr"]
    av = _f8(a, "e5m2")

    def bval(n, k):
        byte = int(b[n, k // 2])
        return Fr(E2M1[(byte >> 4) if k % 2 else (byte & 0xF)])
    for i, j in ((0, 0), (5, 0), (17, 9), (127, 127)):
        e = i * 128 + j
        if category == "premise_violation" and j == 0:
            assert st[e] != H.ST_OK                             # column 0's first scale block is NaN
            continue
        exact = sum(Fr(float(av[i, k])) * Fr(2) ** (int(sa[i, k // 32]) - 127) *
                    bval(j, k) * Fr(2) ** (int(sb[j, k // 32]) - 127) for k in range(256))
        assert _encloses(lo, hi, st, e, exact), (i, j)


# ---- G: scheduling hints ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("category", CATS)
def test_scheduling_hints_signature(category):
    rng = np.random.default_rng(8)
    K = 512                                                     # 8 k-tiles: the pipelined loop runs 4 iterations
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    if category == "boundary":
        a[0, :], b[:, 0] = 65504.0, 65504.0                     # sums far above the f16 range: exact on the reals
    if category == "premise_violation":
        a[9, 300] = np.inf
    bufs = {"a_ptr": a, "b_ptr": b, "c_ptr": np.zeros(64 * 64, np.float32)}
    scalars = [("M", 64), ("N", 64), ("K", K), ("stride_am", K), ("stride_bk", 64), ("stride_cm", 64)]
    out, ref = _both("matmul_pipelined", "gfx942.ttgir", bufs, scalars)
    assert ref.rules.get("amd.scheduling_hint", 0) >= 4
    lo, hi, st = out["c_ptr"]
    for i, j in ((0, 0), (9, 2), (31, 40), (63, 63)):
        e = i * 64 + j
        if category == "premise_violation" and i == 9:
            assert st[e] != H.ST_OK
            continue
        exact = sum(Fr(float(a[i, k])) * Fr(float(b[k, j])) for k in range(K))
        assert _encloses(lo, hi, st, e, exact), (i, j)


# ---- the aborted-launch defect found by the TTIR / TTGIR cross-check (registered before increment 12) ----------------

def test_aborted_launch_does_not_keep_identical_rewrites_as_references():
    """The kernel rewrote out_ptr with the bytes it already held; the reference aborts before its store.  No element of
    out_ptr may keep the captured value as an established reference (only the static address flow shows that the
    kernel may write it: no byte changed and the reference wrote nothing)."""
    ttgir = _ir("masked_copy_bf16", "gfx942.ttgir").replace("%block = amdg.buffer_load",
                                                            "%block = amdg.masked_load")
    a = _bf16(np.linspace(-3, 3, 64))
    out, ref = _eval(ttgir, "ttgir", {"out_ptr": ("bfloat16", a.copy()), "a_ptr": ("bfloat16", a)},
                     scalars=[("shape.0", 64)])
    assert ref.aborted
    buf = ref.buffers[1 << 24]
    assert (np.asarray(buf.st) != H.ST_OK).all()
    assert any("in a buffer the kernel writes" in r for r in ref.reasons)
    # the input buffer is only read: its elements keep their established values
    assert (np.asarray(ref.buffers[2 << 24].st) == H.ST_OK).all()


def test_may_write_analysis_over_approximates_address_escape():
    from kernel_analyzer.reference_eval.ttir_eval import _may_write_params
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    fn = parse_ttir(_ir("masked_copy_bf16", "gfx942.ttgir")).entry()
    names, any_write = _may_write_params(fn)
    assert any_write and "%out_ptr" in names and "%a_ptr" not in names   # (the scalar bound feeds the store mask)
    fn = parse_ttir(_ir("atomic_cas_rows", "gfx942.ttgir")).entry()
    assert _may_write_params(fn) == ({"%index_ptr", "%out_ptr"}, True)
    # an address rebuilt from an integer may point anywhere: every parameter may be written
    text = _ir("masked_copy_bf16", "gfx942.ttgir").replace(
        "    tt.return", "    %q = tt.int_to_ptr %shape.0_0 : i64 -> !tt.ptr<bf16>\n    tt.return")
    assert _may_write_params(parse_ttir(text).entry()) == (None, True)
