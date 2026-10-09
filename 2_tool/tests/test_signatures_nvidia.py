"""NVIDIA Hopper / Blackwell TTGIR modules (DSL v2 increment 11, rc3 04 W3; device validation pending): wgmma
(warp_group_dot), tcgen05 (tc_gen5_mma with tensor memory and mbarriers), memdesc_trans, fp4_to_fp.  Kernels compiled
only (sm_90 / sm_100, no such device here); their TTGIR is evaluated on synthetic captures and checked against exact
rational dot products and against the reference of the same kernel's TTIR.  CPU only."""
from __future__ import annotations

import hashlib
import importlib.util
from fractions import Fraction as Fr

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
M = N = K = 64
_CACHE: dict = {}


def _kernel(src):
    H.KDIR.mkdir(parents=True, exist_ok=True)
    path = H.KDIR / f"nv_{hashlib.sha256(src.encode()).hexdigest()[:12]}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile(src, sig, arch):
    key = (src, arch)
    if key not in _CACHE:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        mod = _kernel(src)
        ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={}), target=GPUTarget("cuda", arch, 32),
                            options={"num_warps": 4})
        _CACHE[key] = {k: ck.asm[k] for k in ("ttir", "ttgir")}
    return _CACHE[key]


def _evaluate(asm, bufs, out_name):
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    args, base, ident = [], 1 << 20, {}
    for i, (name, (dt, arr)) in enumerate(bufs.items()):
        arr = np.ascontiguousarray(arr)
        raw = arr.reshape(-1).view(np.uint8).copy()
        ident[name] = base * (i + 1)
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type=None, dtype=dt,
                                shape=arr.shape, stride=None, element_size=arr.itemsize, data_ptr=base * (i + 1),
                                storage_ptr=base * (i + 1), storage_nbytes=raw.size, storage_id=i, before=raw,
                                after=raw.copy()))
    launch = CapturedLaunch(index=0, kernel_name="nv", kernel_hash="", grid=(1, 1, 1), args=args, asm=asm,
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = ref.buffers[ident[out_name]]
    st = np.where(np.asarray(out.written), np.asarray(out.st), H.ST_NE)
    return np.asarray(out.lo), np.asarray(out.hi), st, ref


DOT_SRC = '''import triton
import triton.language as tl


@triton.jit
def kernel(a_ptr, b_ptr, c_ptr):
    r = tl.arange(0, 64)
    a = tl.load(a_ptr + r[:, None] * 64 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 64 + r[None, :])
    c = tl.dot(a, {B})
    tl.store(c_ptr + r[:, None] * 64 + r[None, :], c)
'''
SIG = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"}


def _inputs(category):
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, K)) * 2).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 2).astype(np.float16)
    if category == "boundary":
        a[0, :], b[:, 0] = 65504.0, 1.0 / 1024   # the largest fp16 times small values
        a[1, :] = 0.0
    if category == "premise_violation":
        a[2, 5] = np.nan
        b[7, 3] = np.inf
    return a, b


def _exact(a, b, i, j):
    return sum(Fr(float(a[i, k])) * Fr(float(b[k, j])) for k in range(K))


def _check_dot(lo, hi, st, a, b, category):
    bad = []
    for i in (0, 1, 2, 5, 17, 40, 63):
        for j in (0, 3, 9, 31, 63):
            e = i * N + j
            if category == "premise_violation" and (i == 2 or j == 3):
                if st[e] == H.ST_OK:
                    bad.append((i, j, "complete with a NaN / inf operand"))
                continue
            if st[e] != H.ST_OK:
                bad.append((i, j, "not complete"))
                continue
            ex = _exact(a, b, i, j)
            if not (Fr(float(lo[e])) <= ex <= Fr(float(hi[e]))):
                bad.append((i, j, float(lo[e]), float(ex), float(hi[e])))
    assert not bad, bad[:4]


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("arch,op", [(90, "ttng.warp_group_dot"), (100, "ttng.tc_gen5_mma")])
def test_hardware_dot_signature(arch, op, category):
    asm = _compile(DOT_SRC.replace("{B}", "b"), SIG, arch)
    assert op in asm["ttgir"], f"{op} not in the sm_{arch} TTGIR"
    a, b = _inputs(category)
    bufs = {"a_ptr": ("float16", a), "b_ptr": ("float16", b), "c_ptr": ("float32", np.zeros((M, N), np.float32))}
    lo, hi, st, _ = _evaluate({"ttgir": asm["ttgir"]}, bufs, "c_ptr")
    _check_dot(lo, hi, st, a, b, category)
    # the same kernel's TTIR reference agrees where both are established
    lo0, hi0, st0, _ = _evaluate({"ttir": asm["ttir"], "ttgir": asm["ttgir"]}, bufs, "c_ptr")
    both = (st == H.ST_OK) & (st0 == H.ST_OK)
    assert not ((hi[both] < lo0[both]) | (hi0[both] < lo[both])).any()


@pytest.mark.parametrize("category", CATS)
def test_transposed_operand_signature(category):
    asm = _compile(DOT_SRC.replace("{B}", "tl.trans(b)"), SIG, 90)
    a, b = _inputs(category)
    bufs = {"a_ptr": ("float16", a), "b_ptr": ("float16", b), "c_ptr": ("float32", np.zeros((M, N), np.float32))}
    lo, hi, st, _ = _evaluate({"ttgir": asm["ttgir"]}, bufs, "c_ptr")
    bt = np.ascontiguousarray(b.T)
    if category == "premise_violation":   # the NaN / inf positions move with the transpose: row 2 and row 3 of b.T
        for i, j in ((2, 0), (0, 7)):
            assert st[i * N + j] != H.ST_OK
        return
    _check_dot(lo, hi, st, a, bt, category)


def test_blackwell_kernel_uses_tensor_memory_barriers_and_invalidation():
    asm = _compile(DOT_SRC.replace("{B}", "b"), SIG, 100)
    for name in ("ttng.tmem_alloc", "ttng.tmem_load", "ttng.init_barrier", "ttng.wait_barrier"):
        assert name in asm["ttgir"], name
    a, b = _inputs("positive")
    bufs = {"a_ptr": ("float16", a), "b_ptr": ("float16", b), "c_ptr": ("float32", np.zeros((M, N), np.float32))}
    lo, hi, st, ref = _evaluate({"ttgir": asm["ttgir"]}, bufs, "c_ptr")
    assert (st == H.ST_OK).all() and ref.rules.get("nvidia.tc_gen5_mma", 0) >= 1
    # removing the barrier wait leaves the accumulator unobserved: the tensor-memory load is not established
    lines = [ln for ln in asm["ttgir"].splitlines() if "ttng.wait_barrier" not in ln]
    lo, hi, st, ref = _evaluate({"ttgir": "\n".join(lines)}, bufs, "c_ptr")
    assert (st != H.ST_OK).all()
