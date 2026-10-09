"""tt.dot_scaled (DSL v2 increment 3, rc3 02 6.5 format bridge): exact MX decoding and the exact real dot, checked on a
compiled-only kernel against exact rational arithmetic.  Positive, boundary and premise-violation cases; fp8 e4m3 and
fp4 e2m1 operands with e8m0 scales."""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402

M = N = 32
K = 64


def _kernel_src(fmt):
    ka = K // 2 if fmt == "e2m1" else K
    return f'''import triton
import triton.language as tl


@triton.jit
def kernel(a_ptr, as_ptr, b_ptr, bs_ptr, out, N: tl.constexpr):
    rm = tl.arange(0, 32)
    ra = tl.arange(0, {ka})
    rk = tl.arange(0, {ka})
    rs = tl.arange(0, 2)
    a = tl.load(a_ptr + rm[:, None] * {ka} + ra[None, :])
    asc = tl.load(as_ptr + rm[:, None] * 2 + rs[None, :])
    b = tl.load(b_ptr + rk[:, None] * 32 + rm[None, :])
    bsc = tl.load(bs_ptr + rm[:, None] * 2 + rs[None, :])
    c = tl.dot_scaled(a, asc, "{fmt}", b, bsc, "{fmt}")
    tl.store(out + rm[:, None] * 32 + rm[None, :], c)
'''


_CACHE = {}


def _compile(fmt):
    if fmt in _CACHE:
        return _CACHE[fmt]
    import hashlib
    import importlib.util

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    src = _kernel_src(fmt)
    H.KDIR.mkdir(parents=True, exist_ok=True)
    path = H.KDIR / f"dscaled_{fmt}_{hashlib.sha256(src.encode()).hexdigest()[:10]}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    elt = "*fp8e4nv" if fmt == "e4m3" else "*u8"
    sig = {"a_ptr": elt, "as_ptr": "*u8", "b_ptr": elt, "bs_ptr": "*u8", "out": "*fp32", "N": "constexpr"}
    # fp8e4nv is not a 3.6.0 type on sm_86: compiled (not run) for sm_90; the reference uses the TTIR only
    ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": 32}), target=GPUTarget("cuda", 90, 32),
                        options={"num_warps": 4})
    _CACHE[fmt] = {k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}
    return _CACHE[fmt]


def _evaluate(fmt, a_raw, a_dtype, a_s, b_raw, b_s):
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    bufs = [("a_ptr", a_dtype, a_raw), ("as_ptr", "uint8", a_s), ("b_ptr", a_dtype, b_raw), ("bs_ptr", "uint8", b_s),
            ("out", "float32", np.zeros((M, N), np.float32))]
    args, base = [], 1 << 20
    for i, (name, dt, arr) in enumerate(bufs):
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type=None, dtype=dt,
                                shape=arr.shape, stride=None, element_size=arr.itemsize, data_ptr=base * (i + 1),
                                storage_ptr=base * (i + 1), storage_nbytes=raw.size, storage_id=i, before=raw,
                                after=raw.copy()))
    args.append(CapturedArg(index=5, name="N", kind="int", constexpr=True, signature_type="constexpr", value=32))
    launch = CapturedLaunch(index=0, kernel_name="dot_scaled", kernel_hash="", grid=(1, 1, 1), args=args,
                            asm=_compile(fmt), cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = ref.buffers[base * 5]
    st = np.where(out.written, out.st, H.ST_NE).reshape(M, N)
    return out.lo.reshape(M, N), out.hi.reshape(M, N), st


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _case(fmt, category, seed=0):
    rng = np.random.default_rng(seed)
    if fmt == "e4m3":
        av = torch.tensor(rng.standard_normal((M, K)) * 4).to(torch.float8_e4m3fn)
        bv = torch.tensor(rng.standard_normal((K, N)) * 4).to(torch.float8_e4m3fn)
        if category == "boundary":
            av[0, :] = 448.0
            av[1, :] = 0.0
        a_raw, b_raw = av.view(torch.uint8).numpy(), bv.view(torch.uint8).numpy()
        a_dec, b_dec = av.float().numpy().astype(np.float64), bv.float().numpy().astype(np.float64)
        dtype = "float8_e4m3fn"
    else:
        an = rng.integers(0, 16, (M, K))
        bn = rng.integers(0, 16, (K, N))
        a_raw = (an[:, 0::2] | (an[:, 1::2] << 4)).astype(np.uint8)
        b_raw = (bn[0::2, :] | (bn[1::2, :] << 4)).astype(np.uint8)
        a_dec, b_dec = np.array(E2M1)[an], np.array(E2M1)[bn]
        dtype = "uint8"
    a_s = rng.integers(120, 135, (M, K // 32)).astype(np.uint8)
    b_s = rng.integers(120, 135, (N, K // 32)).astype(np.uint8)
    if category == "boundary":
        a_s[2, :] = 0
        b_s[3, :] = 254
    if category == "premise_violation":
        a_s[0, 0] = 255  # e8m0 NaN
    return a_raw, dtype, a_s, b_raw, b_s, a_dec, b_dec


def _exact(a_dec, a_s, b_dec, b_s, i, j):
    if a_s[i, :].max() == 255 or b_s[j, :].max() == 255:
        return None
    total = Fr(0)
    for k in range(K):
        fa = Fr(float(a_dec[i, k])) * Fr(2) ** (int(a_s[i, k // 32]) - 127)
        fb = Fr(float(b_dec[k, j])) * Fr(2) ** (int(b_s[j, k // 32]) - 127)
        total += fa * fb
    return total


@pytest.mark.parametrize("fmt", ["e4m3", "e2m1"])
@pytest.mark.parametrize("category", ["positive", "boundary", "premise_violation"])
def test_dot_scaled_signature(fmt, category):
    a_raw, dtype, a_s, b_raw, b_s, a_dec, b_dec = _case(fmt, category)
    lo, hi, st = _evaluate(fmt, a_raw, dtype, a_s, b_raw, b_s)
    bad = []
    for i in range(M):
        for j in range(N):
            ex = _exact(a_dec, a_s, b_dec, b_s, i, j)
            if st[i, j] == H.ST_OK:
                if ex is None:
                    bad.append(f"({i},{j}) complete but undefined (NaN scale)")
                elif not (Fr(float(lo[i, j])) <= ex <= Fr(float(hi[i, j]))):
                    bad.append(f"({i},{j}) [{lo[i, j]}, {hi[i, j]}] misses {float(ex)}")
            elif category == "positive":
                bad.append(f"({i},{j}) not complete in a positive case")
    assert not bad, bad[:4]
