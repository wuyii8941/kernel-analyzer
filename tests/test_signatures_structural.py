"""Per-signature evidence for structural operations, reductions, scans, dot, atomics, control flow and unsigned integer
operations (DSL v2 rc3 04 W1), on compiled-only kernels evaluated on synthetic captures (CPU only)."""
from __future__ import annotations

import hashlib
import importlib.util
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402

INF, NAN = float("inf"), float("nan")
R, C = 4, 8  # 2-D shape for structural ops
U32 = 2 ** 32


def _kernel(name, body, params):
    """Write and import a kernel with the given parameters (pointer names) and body (indented 4 spaces)."""
    H.KDIR.mkdir(parents=True, exist_ok=True)
    src = ("import triton\nimport triton.language as tl\n\n\n@triton.jit\n"
           f"def kernel({', '.join(params)}, N: tl.constexpr):\n" + body)
    path = H.KDIR / f"s_{name}_{hashlib.sha256(src.encode()).hexdigest()[:12]}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(name, body, bufs, out_name="out", grid=(1, 1, 1), scalars=None, drop_ttgir=False):
    """bufs: ordered {name: (dtype, array)}; returns the reference (lo, hi, st) of ``out_name``."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    scalars = scalars or {}
    params = list(bufs) + list(scalars)
    mod = _kernel(name, body, params)
    sig = {k: H.SIG[d] for k, (d, _) in bufs.items()}
    sig.update({k: t for k, (t, _) in scalars.items()})
    sig["N"] = "constexpr"
    key = (name, body, tuple(sig.items()))
    if key not in _CACHE:
        ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": C}), target=GPUTarget("cuda", 86, 32),
                            options={"num_warps": 1})
        _CACHE[key] = {k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}
    asm = dict(_CACHE[key])
    if drop_ttgir:
        asm["ttgir"] = ""
    args, base, ident = [], 1 << 20, {}
    for i, (k, (d, arr)) in enumerate(bufs.items()):
        arr = np.ascontiguousarray(np.asarray(arr, H.NP[d]))
        raw = arr.reshape(-1).view(np.uint8).copy()
        ident[k] = base * (i + 1)
        args.append(CapturedArg(index=i, name=k, kind="tensor", constexpr=False, signature_type=H.SIG[d],
                                dtype=H.TORCH[d], shape=arr.shape, stride=None, element_size=arr.itemsize,
                                data_ptr=base * (i + 1), storage_ptr=base * (i + 1), storage_nbytes=raw.size,
                                storage_id=i, before=raw, after=raw.copy()))
    for k, (t, v) in scalars.items():
        args.append(CapturedArg(index=len(args), name=k, kind="int", constexpr=False, signature_type=t, value=v))
    args.append(CapturedArg(index=len(args), name="N", kind="int", constexpr=True, signature_type="constexpr", value=C))
    launch = CapturedLaunch(index=0, kernel_name=name, kernel_hash="", grid=grid, args=args, asm=asm, cubin_sha256=None,
                            metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = ref.buffers[ident[out_name]]
    # unwritten elements (e.g. a program that aborted before its store) keep the captured value: not a reference
    st = np.where(np.asarray(out.written), np.asarray(out.st), H.ST_NE)
    return np.asarray(out.lo), np.asarray(out.hi if out.hi is not None else out.lo), st


_CACHE: dict = {}


def _check(category, lo, hi, st, exact, is_int=False):
    bad = H.check_lanes(category, lo, hi, st, exact, is_int=is_int)
    assert not bad, bad[:4]


def _vals(category, pos, bnd, pre):
    return {"positive": pos, "boundary": bnd, "premise_violation": pre}[category]


CATS = ("positive", "boundary", "premise_violation")
RNG = np.random.default_rng(7)
POS = RNG.standard_normal(C).astype(np.float32)


# ---------------------------------------------------------------- reductions (one row of C elements)

def _reduce_case(expr, data, ref, dtype="fp32", is_int=False, drop_ttgir=False):
    body = (f"    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    r = {expr}\n    tl.store(out + tl.arange(0, 1), "
            "tl.reshape(r, (1,)) if False else r + tl.zeros([1], dtype=r.dtype))\n")
    lo, hi, st = _run("red_" + expr.replace(" ", "")[:20], body, {"x_ptr": (dtype, data), "out": (dtype, np.zeros(1))},
                      drop_ttgir=drop_ttgir)
    return lo, hi, st, [ref(data)] if ref is not None else None


def _fsum(d):
    if any(np.isnan(d)) or (np.inf in d and -np.inf in d):
        return mp.nan
    if np.inf in d:
        return mp.inf
    if -np.inf in d:
        return -mp.inf
    return mp.mpf(sum(Fr(float(v)) for v in d).numerator) / sum(Fr(float(v)) for v in d).denominator


def _fmax(d, nan="ignore"):
    if nan == "propagate" and any(np.isnan(d)):
        return mp.nan
    d = [v for v in d if not np.isnan(v)]
    return mp.nan if not d else mp.mpf(float(max(d)))


REDUCE = {
    "sum": ("tl.sum(x, axis=0)", _fsum, ([1.0, -2.0, 3.5, 0.25, 7.0, -1.5, 2.0, 4.0],
                                          [0.0, -0.0, 3.4e38, 3.4e38, 1e-45, -1e-45, 2.0, -2.0],
                                          [INF, -INF, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    "max": ("tl.max(x, axis=0)", _fmax, ([1.0, -2.0, 3.5, 0.25, 7.0, -1.5, 2.0, 4.0],
                                          [-INF, -0.0, 0.0, -3.4e38, 1e-45, -1.0, -2.0, -3.0],
                                          [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN])),
    "min": ("tl.min(x, axis=0)", lambda d: -_fmax(-np.asarray(d)) if not all(np.isnan(d)) else mp.nan,
            ([1.0, -2.0, 3.5, 0.25, 7.0, -1.5, 2.0, 4.0], [INF, -0.0, 0.0, 3.4e38, 1e-45, 1.0, 2.0, 3.0],
             [NAN] * 8)),
    "xor": ("tl.xor_sum(x, axis=0)", None, None),
}


@pytest.mark.parametrize("kind,category", [(k, c) for k in ("sum", "max", "min") for c in CATS],
                         ids=lambda v: str(v))
def test_reduce_signature(kind, category):
    expr, ref, cols = REDUCE[kind]
    data = np.asarray(_vals(category, *cols), np.float32)
    lo, hi, st, exact = _reduce_case(expr, data, ref)
    _check(category if category != "premise_violation" or kind == "sum" else "boundary", lo, hi, st, exact)


@pytest.mark.parametrize("category", CATS)
def test_reduce_int_signature(category):
    data = np.asarray(_vals(category, [3, -7, 100, 5, 0, 9, -2, 1], [2 ** 31 - 1, 1, 0, 0, 0, 0, 0, 0],
                            [-2 ** 31, -1, 0, 0, 0, 0, 0, 0]), np.int32)
    lo, hi, st, _ = _reduce_case("tl.sum(x, axis=0)", data, None, dtype="int32", is_int=True)
    s = int(np.sum(data.astype(np.int64)))
    _check(category, lo, hi, st, [((s + 2 ** 31) % U32) - 2 ** 31], is_int=True)


@pytest.mark.parametrize("category", CATS)
def test_reduce_xor_and_argmax_signature(category):
    data = np.asarray(_vals(category, [3, 7, 100, 5, 0, 9, 2, 1], [-1, 0, 2 ** 31 - 1, -2 ** 31, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]), np.int32)
    lo, hi, st, _ = _reduce_case("tl.xor_sum(x, axis=0)", data, None, dtype="int32", is_int=True)
    v = 0
    for d in data:
        v ^= int(d) & 0xFFFFFFFF
    _check(category, lo, hi, st, [v - U32 if v >= 2 ** 31 else v], is_int=True)
    f = np.asarray(_vals(category, [1.0, -2.0, 9.5, 0.25, 7.0, -1.5, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         [-INF, -INF, -INF, -INF, -INF, -INF, -INF, -INF]), np.float32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    r = tl.argmax(x, axis=0)\n"
            "    tl.store(out + tl.arange(0, 1), r + tl.zeros([1], dtype=r.dtype))\n")
    lo, hi, st = _run("argmax", body, {"x_ptr": ("fp32", f), "out": ("int32", np.zeros(1))})
    _check("boundary", lo, hi, st, [int(np.argmax(f))], is_int=True)  # ties: the smallest index (tl.argmax default)


def test_generic_reduce_without_ttgir_is_not_established_premise_violation():
    # premise of the generic-region route: a trusted order (TTGIR layout); without it the reference is not established
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    r = tl.reduce(x, 0, _comb)\n"
            "    tl.store(out + tl.arange(0, 1), r + tl.zeros([1], dtype=r.dtype))\n")
    body = body  # the combine function is defined in the kernel module below
    src_comb = "@triton.jit\ndef _comb(a, b):\n    return a * 0.5 + b\n\n\n"
    H.KDIR.mkdir(parents=True, exist_ok=True)
    global _kernel
    orig = _kernel

    def with_comb(name, body, params):
        src = ("import triton\nimport triton.language as tl\n\n\n" + src_comb + "@triton.jit\n"
               f"def kernel({', '.join(params)}, N: tl.constexpr):\n" + body)
        path = H.KDIR / f"s_{name}_{hashlib.sha256(src.encode()).hexdigest()[:12]}.py"
        if not path.exists():
            path.write_text(src)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    _kernel = with_comb
    try:
        data = POS
        lo, hi, st = _run("generic_reduce", body, {"x_ptr": ("fp32", data), "out": ("fp32", np.zeros(1))},
                          drop_ttgir=True)
        assert (st != H.ST_OK).all()
        lo, hi, st = _run("generic_reduce", body, {"x_ptr": ("fp32", data), "out": ("fp32", np.zeros(1))})
        assert (st == H.ST_OK).all()     # with the TTGIR order the order-specific reference is complete
    finally:
        _kernel = orig


# ---------------------------------------------------------------- scans

@pytest.mark.parametrize("category", CATS)
def test_scan_cumsum_signature(category):
    data = np.asarray(_vals(category, POS.tolist(), [0.0, -0.0, 3.4e38, 3.4e38, 1e-45, -1e-45, 1.0, -1.0],
                            [1.0, INF, -INF, 1.0, 2.0, 3.0, 4.0, 5.0]), np.float32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.store(out + i, tl.cumsum(x, axis=0))\n"
    lo, hi, st = _run("cumsum", body, {"x_ptr": ("fp32", data), "out": ("fp32", np.zeros(C))})
    exact = [_fsum(data[:k + 1]) for k in range(C)]
    _check(category, lo, hi, st, exact)


@pytest.mark.parametrize("category", CATS)
def test_scan_int_and_cumprod_signature(category):
    data = np.asarray(_vals(category, [3, -7, 100, 5, 0, 9, -2, 1], [2 ** 31 - 1, 1, 1, 0, 0, 0, 0, 0],
                            [-2 ** 31, -1, 0, 0, 0, 0, 0, 0]), np.int32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.store(out + i, tl.cumsum(x, axis=0))\n"
    lo, hi, st = _run("cumsum_i", body, {"x_ptr": ("int32", data), "out": ("int32", np.zeros(C))})
    acc, exact = 0, []
    for d in data:
        acc = ((acc + int(d) + 2 ** 31) % U32) - 2 ** 31
        exact.append(acc)
    _check(category, lo, hi, st, exact, is_int=True)
    f = np.asarray(_vals(category, [1.5, -2.0, 0.5, 3.0, 1.0, -1.0, 2.0, 0.25], [0.0, 3.4e38, 3.4e38, 1.0, 1.0, 1.0, 1.0,
                                                                                   1.0],
                         [INF, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.float32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.store(out + i, tl.cumprod(x, axis=0))\n"
    lo, hi, st = _run("cumprod", body, {"x_ptr": ("fp32", f), "out": ("fp32", np.zeros(C))})
    p, exact = mp.mpf(1), []
    for v in f:
        p = mp.nan if (mp.isinf(p) and v == 0) or (p == 0 and np.isinf(v)) else p * mp.mpf(float(v))
        exact.append(p)
    _check(category, lo, hi, st, exact)


# ---------------------------------------------------------------- dot (fp32, ieee)

@pytest.mark.parametrize("category", CATS)
def test_dot_ieee_signature(category):
    n = 16
    a = RNG.standard_normal((n, n)).astype(np.float32)
    b = RNG.standard_normal((n, n)).astype(np.float32)
    if category == "boundary":
        a[0, :] = 0.0
        b[:, 1] = 3.4e38 / 64
    if category == "premise_violation":
        a[0, 0], b[0, 0] = INF, 0.0
    body = ("    r = tl.arange(0, 16)\n    a = tl.load(a_ptr + r[:, None] * 16 + r[None, :])\n"
            "    b = tl.load(b_ptr + r[:, None] * 16 + r[None, :])\n"
            "    c = tl.dot(a, b, input_precision='ieee')\n    tl.store(out + r[:, None] * 16 + r[None, :], c)\n")
    lo, hi, st = _run("dot", body, {"a_ptr": ("fp32", a), "b_ptr": ("fp32", b), "out": ("fp32", np.zeros((n, n)))})
    exact = []
    for i in range(n):
        for j in range(n):
            terms = [(float(a[i, k]), float(b[k, j])) for k in range(n)]
            if any((np.isinf(x) and y == 0) or (np.isinf(y) and x == 0) for x, y in terms):
                exact.append(mp.nan)
            else:
                exact.append(mp.fsum(mp.mpf(x) * mp.mpf(y) for x, y in terms))
    _check(category, lo.reshape(-1), hi.reshape(-1), st.reshape(-1), exact)


# ---------------------------------------------------------------- shape and layout operations

@pytest.mark.parametrize("category", CATS)
def test_shape_ops_signature(category):
    x = np.asarray(_vals(category, POS.tolist(), [0.0, -0.0, INF, -INF, 1e-45, 3.4e38, 1.0, -1.0], [NAN] * 8),
                   np.float32)
    # broadcast / expand_dims / reshape / trans / join / split
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    m = x[:, None] + tl.zeros([N, 2], dtype=x.dtype)\n"
            "    t = tl.trans(m)\n    j = tl.join(x, x)\n    a, b = tl.split(j)\n    r = tl.reshape(t, (2 * N,))\n"
            "    tl.store(out + tl.arange(0, 2 * N), r)\n    tl.store(out2 + i, a + b)\n")
    bufs = {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(2 * C)), "out2": ("fp32", np.zeros(C))}
    lo, hi, st = _run("shape", body, bufs)
    exact = [mp.mpf(float(v)) if np.isfinite(v) else (mp.nan if np.isnan(v) else (mp.inf if v > 0 else -mp.inf))
             for v in np.concatenate([x, x])]
    _check(category, lo, hi, st, exact)
    lo, hi, st = _run("shape", body, bufs, out_name="out2")
    exact = [mp.mpf(float(v)) * 2 if np.isfinite(v) else (mp.nan if np.isnan(v) else (mp.inf if v > 0 else -mp.inf))
             for v in x]
    _check(category, lo, hi, st, exact)


@pytest.mark.parametrize("category", CATS)
def test_clamp_and_fp8_signature(category):
    x = np.asarray(_vals(category, POS.tolist(), [0.0, -0.0, INF, -INF, 1e-45, 3.4e38, 1.0, -1.0], [NAN] * 8),
                   np.float32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.store(out + i, tl.clamp(x, -1.0, 1.0))\n"
    lo, hi, st = _run("clamp", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(C))})
    # tl.clamp default propagate_nan=NONE: min(max(x, lo), hi) with NaN-ignoring max / min, so clamp(NaN) = lo
    exact = [mp.mpf(-1.0) if np.isnan(v) else mp.mpf(float(min(max(v, -1.0), 1.0))) for v in x]
    _check(category, lo, hi, st, exact)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
            "    tl.store(out + i, x.to(tl.float8e5).to(tl.float32))\n")
    lo, hi, st = _run("fp8", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(C))})
    exact = [mp.mpf(float(v)) if np.isfinite(v) else (mp.nan if np.isnan(v) else (mp.inf if v > 0 else -mp.inf))
             for v in x]
    _check("boundary" if category == "positive" else category, lo, hi, st, exact)  # conversions are identity on reals


# ---------------------------------------------------------------- unsigned integer operations

@pytest.mark.parametrize("category", CATS)
def test_unsigned_ops_signature(category):
    a = np.asarray(_vals(category, [7, 100, 12, 3000000000, 5, 9, 1, 2], [0, U32 - 1, 2 ** 31, 1, 0, 0, 0, 0],
                         [5, 3, 1, 1, 1, 1, 1, 1]), np.uint32)
    b = np.asarray(_vals(category, [2, 3, 5, 7, 1, 4, 1, 2], [1, U32 - 1, 2, 31, 1, 1, 1, 1],
                         [0, 0, 32, 40, 0, 0, 0, 0]), np.uint32)
    for op, f in (("a // b", lambda x, y: None if y == 0 else x // y), ("a % b", lambda x, y: None if y == 0 else x % y),
                  ("a >> b", lambda x, y: None if y >= 32 else x >> y), ("tl.maximum(a, b)", max),
                  ("tl.minimum(a, b)", min), ("tl.umulhi(a, b)", lambda x, y: (x * y) >> 32)):
        body = (f"    i = tl.arange(0, N)\n    a = tl.load(a_ptr + i)\n    b = tl.load(b_ptr + i)\n"
                f"    tl.store(out + i, {op})\n")
        lo, hi, st = _run("u_" + op[:10].replace(" ", "").replace("/", "d").replace("%", "m").replace(">", "s")
                          .replace("(", "").replace(",", ""), body,
                          {"a_ptr": ("uint32", a), "b_ptr": ("uint32", b), "out": ("uint32", np.zeros(C))})
        exact = [f(int(x), int(y)) for x, y in zip(a, b)]
        lo_u = np.asarray(lo, np.int64) % U32
        bad = H.check_lanes(category if op not in ("tl.maximum(a, b)", "tl.minimum(a, b)", "tl.umulhi(a, b)")
                            else "boundary", lo_u, lo_u, st, exact, is_int=True)
        assert not bad, (op, bad[:3])
    for op, f, dt_in, dt_out in (("a.to(tl.float32)", lambda x: mp.mpf(int(x)), "uint32", "fp32"),):
        body = f"    i = tl.arange(0, N)\n    a = tl.load(a_ptr + i)\n    tl.store(out + i, {op})\n"
        lo, hi, st = _run("uitofp", body, {"a_ptr": (dt_in, a), "out": (dt_out, np.zeros(C))})
        _check("boundary", lo, hi, st, [f(x) for x in a])


# ---------------------------------------------------------------- atomics and control flow

@pytest.mark.parametrize("category", CATS)
def test_atomic_add_signature(category):
    x = np.asarray(_vals(category, POS.tolist(), [0.0, -0.0, 3.4e38, 3.4e38, 1e-45, 1.0, -1.0, 2.0],
                         [INF, -INF, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.float32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.atomic_add(out + i * 0, x)\n"
    lo, hi, st = _run("atomic_add", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(1))})
    _check(category, lo, hi, st, [_fsum(x)])
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    old = tl.atomic_add(acc + i * 0, x)\n"
            "    tl.store(out + i, old)\n")
    lo, hi, st = _run("atomic_used", body, {"x_ptr": ("fp32", x), "acc": ("fp32", np.zeros(1)),
                                            "out": ("fp32", np.zeros(C))})
    assert (st != H.ST_OK).all()   # a used return value depends on the undeclared order: never complete


@pytest.mark.parametrize("category", CATS)
def test_control_flow_signature(category):
    x = np.asarray(_vals(category, POS.tolist(), [0.0, -0.0, 3.4e38, -3.4e38, 1e-45, 1.0, -1.0, 2.0],
                         [NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), np.float32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    acc = tl.zeros([N], dtype=tl.float32)\n"
            "    for k in range(3):\n        acc += x * (k + 1)\n    s = tl.sum(x, axis=0)\n"
            "    if s > 0:\n        acc = acc + 1.0\n    else:\n        acc = acc - 1.0\n"
            "    n = 0\n    while n < 2:\n        acc = acc * 0.5\n        n += 1\n    tl.store(out + i, acc)\n")
    lo, hi, st = _run("control", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(C))})
    s = _fsum(x)
    exact = []
    for v in x:
        if np.isnan(v):
            exact.append(None)
            continue
        a = mp.mpf(float(v)) * 6 + (1 if (not mp.isnan(s) and s > 0) else -1)  # NaN > 0 is false: the else branch
        exact.append(a / 4)
    _check(category if category != "premise_violation" else "boundary", lo, hi, st, exact)


def test_num_programs_and_pointer_casts_signature():
    x = POS
    body = ("    i = tl.arange(0, N)\n    p = (x_ptr.to(tl.int64, bitcast=True)).to(tl.pointer_type(tl.float32), "
            "bitcast=True)\n    x = tl.load(p + i)\n    n = tl.num_programs(0)\n    tl.store(out + i, x * n)\n")
    lo, hi, st = _run("nprog", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(C))})
    _check("positive", lo, hi, st, [mp.mpf(float(v)) for v in x])


@pytest.mark.parametrize("category", CATS)
def test_unsigned_atomics_and_wide_mulhi_signature(category):
    # atomic umax / umin on uint32 (unsigned comparison), and the high half of a 64 x 64 bit product
    v = np.asarray(_vals(category, [7, 100, 12, 3000000000, 5, 9, 1, 2], [0, U32 - 1, 2 ** 31, 1, 0, 0, 0, 0],
                         [2 ** 31, 2 ** 31 - 1, 0, 0, 0, 0, 0, 0]), np.uint32)
    for fn, red in (("tl.atomic_max", max), ("tl.atomic_min", min)):
        body = f"    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    {fn}(out + i * 0, x)\n"
        init = np.asarray([5], np.uint32)
        lo, hi, st = _run("uatomic_" + fn[-3:], body, {"x_ptr": ("uint32", v), "out": ("uint32", init)})
        exact = red([5] + [int(t) for t in v])
        lo_u = np.asarray(lo, np.int64) % U32
        bad = H.check_lanes("boundary", lo_u, lo_u, st, [exact], is_int=True)
        assert not bad, (fn, bad)
    a = np.asarray(_vals(category, [3, 2 ** 63 + 5, 2 ** 40, 7, 1, 2, 3, 4], [2 ** 64 - 1, 2 ** 64 - 1, 0, 1, 1, 1, 1, 1],
                         [2 ** 63, 2 ** 63, 1, 1, 1, 1, 1, 1]), np.uint64)
    b = np.asarray(_vals(category, [5, 3, 2 ** 30, 2 ** 62, 1, 2, 3, 4], [2 ** 64 - 1, 2, 5, 1, 1, 1, 1, 1],
                         [2, 2 ** 63, 1, 1, 1, 1, 1, 1]), np.uint64)
    body = ("    i = tl.arange(0, N)\n    a = tl.load(a_ptr + i)\n    b = tl.load(b_ptr + i)\n"
            "    tl.store(out + i, tl.umulhi(a, b))\n")
    lo, hi, st = _run("umulhi64", body, {"a_ptr": ("uint64", a), "b_ptr": ("uint64", b), "out": ("uint64", np.zeros(C))})
    exact = [(int(x) * int(y)) >> 64 for x, y in zip(a, b)]
    lo_u = [int(t) % (1 << 64) for t in np.asarray(lo, np.int64)]
    bad = H.check_lanes("boundary", lo_u, lo_u, st, exact, is_int=True)
    assert not bad, bad[:3]


def test_atomic_cas_without_contention_is_determined_and_with_contention_is_a_race():
    body = ("    pid = tl.program_id(0)\n    old = tl.atomic_cas(lock + pid, 0, pid + 1)\n"
            "    tl.store(out + pid, old)\n")
    init = np.asarray([0, 7, 0, 0], np.int32)
    lo, hi, st = _run("cas_own", body, {"lock": ("int32", init), "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    _check("positive", lo, hi, st, [0, 7, 0, 0], is_int=True)               # the old values
    lo, hi, st = _run("cas_own", body, {"lock": ("int32", init), "out": ("int32", np.zeros(4))}, out_name="lock",
                      grid=(4, 1, 1))
    _check("positive", lo, hi, st, [1, 7, 3, 4], is_int=True)               # swapped where old == 0
    body = ("    pid = tl.program_id(0)\n    old = tl.atomic_cas(lock, 0, pid + 1)\n"
            "    tl.store(out + pid, old)\n")
    lo, hi, st = _run("cas_shared", body, {"lock": ("int32", np.zeros(1)), "out": ("int32", np.zeros(4))},
                      grid=(4, 1, 1))
    assert (st != H.ST_OK).all()                                            # interleaving-dependent: a race
