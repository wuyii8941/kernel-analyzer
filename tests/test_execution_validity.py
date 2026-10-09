"""Execution validity checks of the reference evaluator (DSL v2 rc3 02 6.3, 04 W4), on compiled-only kernels.

The kernels are compiled for sm_86 (``triton.compile``, no launch) and evaluated on synthetic captures, so no racy
kernel is ever executed.  Covered:

* a value stored by one thread of a program and loaded back by the other threads without a barrier (the T5 pattern of
  the structure acceptance set): execution race, the dependent outputs are not established;
* the same store and load of a tensor in one layout (each thread reads its own element): ordered, allowed;
* the scalar round trip with a barrier in between: allowed;
* a program reading the slot another program of the same launch writes (the prog_26 pattern), in either evaluation
  order: execution race;
* negative control: programs reading a read-only input and writing disjoint outputs: no finding.

The lowering facts the thread check relies on are visible in the PTX of the scalar case: the store is predicated to
thread 0, the load is executed by every thread, and no bar.sync separates them.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_NE, ST_OK, evaluate_sequence  # noqa: E402

R, D = 8, 256


@triton.jit
def _scalar_roundtrip(x_ptr, buf_ptr, y_ptr, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + c)
    m = tl.sum(x, axis=0) / D
    tl.store(buf_ptr + r, m)
    m2 = tl.load(buf_ptr + r)
    tl.store(y_ptr + r * D + c, x - m2)


@triton.jit
def _scalar_roundtrip_barrier(x_ptr, buf_ptr, y_ptr, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + c)
    m = tl.sum(x, axis=0) / D
    tl.store(buf_ptr + r, m)
    tl.debug_barrier()
    m2 = tl.load(buf_ptr + r)
    tl.store(y_ptr + r * D + c, x - m2)


@triton.jit
def _tensor_roundtrip(x_ptr, buf_ptr, y_ptr, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + c)
    tl.store(buf_ptr + r * D + c, x * 3.0)
    t = tl.load(buf_ptr + r * D + c)
    tl.store(y_ptr + r * D + c, t + 1.0)


@triton.jit
def _neighbour(x_ptr, buf_ptr, y_ptr, R, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + c)
    m = tl.sum(x, axis=0) / D
    tl.store(buf_ptr + r, m)
    m2 = tl.load(buf_ptr + (r + 1) % R)
    tl.store(y_ptr + r * D + c, x - m2)


@triton.jit
def _disjoint(x_ptr, buf_ptr, y_ptr, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + c)
    tl.store(buf_ptr + r, tl.sum(x, axis=0))
    tl.store(y_ptr + r * D + c, x * 2.0)


def _launch(fn, with_R=False):
    sig = {"x_ptr": "*fp32", "buf_ptr": "*fp32", "y_ptr": "*fp32", "D": "constexpr"}
    if with_R:
        sig = {"x_ptr": "*fp32", "buf_ptr": "*fp32", "y_ptr": "*fp32", "R": "i32", "D": "constexpr"}
    ck = triton.compile(ASTSource(fn=fn, signature=sig, constexprs={"D": D}), target=GPUTarget("cuda", 86, 32),
                        options={"num_warps": 4})
    rng = np.random.default_rng(0)
    x = rng.standard_normal((R, D)).astype(np.float32)
    bufs = {"x_ptr": x, "buf_ptr": np.zeros(R * D if fn is _tensor_roundtrip else R, np.float32),
            "y_ptr": np.zeros((R, D), np.float32)}
    args, base = [], 1 << 20
    for i, (name, arr) in enumerate(bufs.items()):
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type="*fp32",
                                dtype="float32", shape=arr.shape, stride=None, element_size=4,
                                data_ptr=base * (i + 1), storage_ptr=base * (i + 1), storage_nbytes=raw.size,
                                storage_id=i, before=raw, after=raw.copy()))
    if with_R:
        args.append(CapturedArg(index=3, name="R", kind="int", constexpr=False, signature_type="i32", value=R))
    args.append(CapturedArg(index=len(args), name="D", kind="int", constexpr=True, signature_type="constexpr", value=D))
    launch = CapturedLaunch(index=0, kernel_name=fn.__name__, kernel_hash="", grid=(R, 1, 1), args=args,
                            asm={k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}, cubin_sha256=None, metadata={},
                            libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    y = ref.buffers[base * 3]
    return ref, y.st.reshape(R, D), ck


def test_scalar_roundtrip_without_barrier_is_an_execution_race():
    ref, st, ck = _launch(_scalar_roundtrip)
    assert ref.rules.get("execution.intra_program_cross_thread_race_lanes", 0) > 0
    assert (st == ST_NE).all()
    ptx = ck.asm["ptx"]
    store = [ln for ln in ptx.splitlines() if "st.global.b32 [" in ln and "{ %r" in ln]
    assert store and store[0].strip().startswith("@%p")   # predicated store (one thread)


def test_tensor_roundtrip_in_one_layout_is_ordered():
    ref, st, _ = _launch(_tensor_roundtrip)
    assert ref.rules.get("execution.same_program_write_read_same_thread_lanes", 0) == R * D
    assert ref.rules.get("execution.intra_program_cross_thread_race_lanes", 0) == 0
    assert (st == ST_OK).all()


def test_barrier_orders_the_scalar_roundtrip():
    ref, st, _ = _launch(_scalar_roundtrip_barrier)
    assert ref.rules.get("execution.intra_program_cross_thread_race_lanes", 0) == 0
    assert (st == ST_OK).all()


def test_reading_the_slot_another_program_writes_is_an_execution_race():
    ref, st, _ = _launch(_neighbour, with_R=True)
    # rows 0..R-2 read a slot written later in evaluation order (read then write); row R-1 reads slot 0, already
    # written (load after another program's write)
    assert ref.rules.get("execution.cross_program_read_write_race_programs", 0) == R - 1
    assert ref.rules.get("memory.cross_program_race_lanes", 0) >= 1
    assert (st == ST_NE).all()


def test_disjoint_programs_have_no_finding():
    ref, st, _ = _launch(_disjoint)
    for key in ("execution.cross_program_read_write_race_programs", "execution.intra_program_cross_thread_race_lanes",
                "memory.cross_program_race_lanes"):
        assert ref.rules.get(key, 0) == 0
    assert (st == ST_OK).all()
