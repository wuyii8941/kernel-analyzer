"""Gluon kernels from their TTGIR (DSL v2 increment 10, rc3 04 W3): shared memory, views, gather / scatter, atomic
scatter, asynchronous copies with mbarriers.  The TTGIR fixtures in tests/data/gluon_ttgir are what the official main
build (e50b186e) produced for official Gluon unit tests (python/test/gluon/test_core.py), location annotations
removed; the variants below change only operands or one instruction and say so.  Synthetic captures, CPU only;
expected values computed directly with numpy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import signature_harness as H

DATA = Path(__file__).parent / "data" / "gluon_ttgir"
CATS = ("positive", "boundary", "premise_violation")
NP = {"i32": np.int32, "f32": np.float32, "i8": np.int8}
TORCH = {"i32": "int32", "f32": "float32", "i8": "int8"}


def _eval(ttgir, bufs, scalars=(), grid=(1, 1, 1)):
    """A launch whose only IR is the TTGIR (as for Gluon kernels); bufs: ordered {name: (elem, array)}."""
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
    for name, value in scalars:
        args.append(CapturedArg(index=len(args), name=name, kind="int", constexpr=False, signature_type="i32",
                                value=value))
    launch = CapturedLaunch(index=0, kernel_name="k", kernel_hash="", grid=grid, args=args, asm={"ttgir": ttgir},
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = {}
    for i, name in enumerate(bufs):
        b = ref.buffers[base * (i + 1)]
        st = np.where(np.asarray(b.written), np.asarray(b.st), H.ST_NE)
        out[name] = (np.asarray(b.lo), np.asarray(b.hi if b.hi is not None else b.lo), st)
    return out, ref


@pytest.mark.parametrize("category", CATS)
def test_shared_scatter_signature(category):
    ttgir = (DATA / "shared_scatter.ttgir").read_text()
    vals = {"positive": np.arange(64, dtype=np.float32) * 0.5, "boundary": np.full(64, 3.4e38, np.float32),
            "premise_violation": np.arange(64, dtype=np.float32)}[category]
    if category == "premise_violation":
        # variant: the scatter index is the row index (every column of a row collides on one element)
        ttgir = ttgir.replace("ttg.local_scatter %smem[%offsets_2d_4]", "ttg.local_scatter %smem[%offsets_2d_3]")
    out, ref = _eval(ttgir, {"indices_ptr": ("i32", np.zeros(64)), "values_ptr": ("f32", vals),
                             "output_ptr": ("f32", np.zeros(64 * 64))})
    lo, hi, st = out["output_ptr"]
    if category == "premise_violation":
        assert (st != H.ST_OK).any()      # out-of-range or colliding scatter indices leave elements without a value
        return
    want = np.repeat(vals.astype(np.float64), 64)
    assert (st == H.ST_OK).all() and np.array_equal(lo, want)
    assert any("Membar" in r for r in ref.reasons)


@pytest.mark.parametrize("category", CATS)
def test_gather_from_a_padded_subslice_signature(category):
    ttgir = (DATA / "gather_padded_subslice.ttgir").read_text()
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((64, 64)).astype(np.float32)
    idx = {"positive": rng.integers(0, 16, (16, 16)), "boundary": np.tile([0, 15], (16, 8)),
           "premise_violation": rng.integers(0, 16, (16, 16))}[category].astype(np.int32)
    if category == "premise_violation":
        idx[3, 4], idx[7, 7] = 16, -1                              # outside the 16-row subslice
    out, _ = _eval(ttgir, {"matrix_ptr": ("f32", mat), "indices_ptr": ("i32", idx), "output_ptr": ("f32", np.zeros(256))})
    lo, hi, st = out["output_ptr"]
    sub = mat[48:64, 16:32]
    for i in range(16):
        for j in range(16):
            k = int(idx[i, j])
            if 0 <= k < 16:
                assert st[i * 16 + j] == H.ST_OK and lo[i * 16 + j] == sub[k, j]
            else:
                assert st[i * 16 + j] != H.ST_OK


@pytest.mark.parametrize("category", CATS)
def test_atomic_scatter_rmw_signature(category):
    ttgir = (DATA / "atomic_scatter_rmw_add.ttgir").read_text()
    rng = np.random.default_rng(2)
    idx = {"positive": rng.permutation(32)[:2 * 16].reshape(2, 16) % 32, "boundary": np.zeros((2, 16)),
           "premise_violation": rng.integers(0, 32, (2, 16))}[category].astype(np.int32)
    if category == "positive":
        idx = np.stack([np.arange(16), np.arange(16, 32)])        # distinct rows per lane of each column
    mask = np.ones((2, 16), np.int8)
    if category == "premise_violation":
        idx[0, 3] = 40                                            # outside the 32 rows
        mask[1, 5] = 0                                            # masked off: no update, no value
    out, _ = _eval(ttgir, {"values_ptr": ("i32", np.zeros(32)), "indices_ptr": ("i32", idx), "mask_ptr": ("i8", mask),
                           "old_ptr": ("i32", np.zeros(32)), "final_ptr": ("i32", np.zeros(32 * 16))})
    lo, hi, st = out["final_ptr"]
    want = np.zeros((32, 16), np.int64)
    for r in range(2):
        for c in range(16):
            if mask[r, c] and 0 <= idx[r, c] < 32:
                want[idx[r, c], c] += 1
    for e in range(32 * 16):
        row, col = divmod(e, 16)
        if category == "premise_violation" and ((row == 40) or col == 3):
            continue
        assert st[e] == H.ST_OK and int(lo[e]) == int(want[row, col]), (row, col)
    lo, hi, st = out["old_ptr"]
    for r in range(2):
        for c in range(16):
            j = r * 16 + c
            if category == "boundary":         # both lanes of a column hit row 0: each sees 0 or 1 (a set), not a point
                assert st[j] != H.ST_OK
            elif category == "positive":
                assert st[j] == H.ST_OK and int(lo[j]) == 0


@pytest.mark.parametrize("category", CATS)
def test_async_copy_with_mbarrier_signature(category):
    ttgir = (DATA / "async_copy_mbarrier.ttgir").read_text()
    rng = np.random.default_rng(3)
    inp = rng.standard_normal(32 * 32).astype(np.float32)
    xnumel = {"positive": 32, "boundary": 1, "premise_violation": 32}[category]
    if category == "premise_violation":
        # variant: drop the final arrive / wait pair, so the copy is never observed complete before the load
        lines = [ln for ln in ttgir.splitlines()]
        last_wait = max(i for i, ln in enumerate(lines) if "ttng.wait_barrier" in ln)
        del lines[last_wait]
        last_arrive = max(i for i, ln in enumerate(lines) if "ttng.arrive_barrier" in ln)
        del lines[last_arrive]
        ttgir = "\n".join(lines)
    out, _ = _eval(ttgir, {"out": ("f32", np.zeros(32 * 32)), "inp": ("f32", inp)}, scalars=[("xnumel", xnumel)])
    lo, hi, st = out["out"]
    if category == "premise_violation":
        assert (st != H.ST_OK).all()
        return
    for e in range(32 * 32):
        row = e // 32
        if row < xnumel:
            assert st[e] == H.ST_OK and lo[e] == inp[e]
        else:   # masked-off rows are zero-filled (the official test asserts zeros over the initial 7s)
            assert st[e] == H.ST_OK and lo[e] == 0.0


def test_mbarrier_wait_that_cannot_complete_is_not_established():
    # variant of the official kernel: the first arrive is removed, so the first wait (parity 0) has no arrival
    ttgir = (DATA / "async_copy_mbarrier.ttgir").read_text()
    lines = ttgir.splitlines()
    first = min(i for i, ln in enumerate(lines) if "ttng.arrive_barrier" in ln)
    del lines[first]
    out, ref = _eval("\n".join(lines), {"out": ("f32", np.zeros(32 * 32)), "inp": ("f32", np.ones(32 * 32))},
                     scalars=[("xnumel", 32)])
    assert ref.aborted and (out["out"][2] != H.ST_OK).all()
