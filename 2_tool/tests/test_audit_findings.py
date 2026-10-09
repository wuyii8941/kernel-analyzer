"""Regressions for the external audit at 1aee15e (3_audits/README.md, findings F01-F07), written before the fixes.

Each test states its evidence level: rule level (one evaluator function on constructed values), full interpreter (a
compiled kernel evaluated on synthetic captures, CPU only), or GPU end to end (``check.run`` / ``measure.run`` on the
sm_86 device).  Expected values come from independent computations: Python integers, exact rationals, or the
enumeration of every serialization.
"""
from __future__ import annotations

import itertools
import json
from fractions import Fraction as Fr
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from kernel_analyzer.reference_eval import ttir_eval as E
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, TV

HERE = Path(__file__).resolve().parent
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ------------------------------------------------------------------------------------------------ F01 integers
# rule level: arith integer ops against Python big integers, every declared width, boundary arguments

def _signed(v, w):
    return ((int(v) + (1 << (w - 1))) % (1 << w)) - (1 << (w - 1))


def _unsigned_values(w):
    vals = {0, 1, 2, 3, 5, 7, (1 << (w - 1)) - 1, 1 << (w - 1), (1 << (w - 1)) + 1, (1 << w) - 2, (1 << w) - 1}
    if w == 64:
        vals |= {(1 << 53) - 1, 1 << 53, (1 << 53) + 1, 1 << 62, (1 << 63) - 2, (1 << 63) + (1 << 53)}
    return sorted(vals)


def _signed_values(w):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    vals = {0, 1, -1, 2, -2, 3, -3, 7, -7, lo, lo + 1, lo + 2, hi, hi - 1}
    if w == 64:
        vals |= {(1 << 53) + 1, -(1 << 53) - 1, 1 << 62, -(1 << 62)}
    return sorted(vals)


def _int_args(pairs, w):
    a = np.array([_signed(p[0], w) for p in pairs], dtype=np.int64)
    b = np.array([_signed(p[1], w) for p in pairs], dtype=np.int64)
    return [TV("i", f"i{w}", a), TV("i", f"i{w}", b)]


def _op(w):
    return NS(result_types=[NS(elem=f"i{w}")], node_id="audit", attrs={})


def _trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


UNSIGNED = {"divui": lambda a, b: a // b, "remui": lambda a, b: a % b, "ceildivui": lambda a, b: -(-a // b),
            "maxui": max, "minui": min}
SIGNED = {"divsi": _trunc_div, "remsi": lambda a, b: a - b * _trunc_div(a, b),
          "ceildivsi": lambda a, b: -((-a) // b), "floordivsi": lambda a, b: a // b,
          "maxsi": max, "minsi": min}
WIDTHS = (8, 16, 32, 64)


def _check_int(name, w, pairs, expect, legal):
    out = E._int_op(name, _op(w), _int_args(pairs, w))
    bad = []
    for j, (a, b) in enumerate(pairs):
        st = int(np.asarray(out.st).reshape(-1)[j])
        got = int(np.asarray(out.lo).reshape(-1)[j]) % (1 << w)
        if legal(a, b):
            want = expect(a, b) % (1 << w)
            if st != ST_OK or got != want:
                bad.append((a, b, want, got, st))
        elif st == ST_OK:   # division by zero, INT_MIN / -1: undefined behaviour, never a legal value
            bad.append((a, b, "not established", got, st))
    assert not bad, bad[:6]


@pytest.mark.parametrize("w", WIDTHS)
@pytest.mark.parametrize("name", sorted(UNSIGNED))
def test_unsigned_integer_ops_match_python_integers(name, w):
    vals = _unsigned_values(w)
    pairs = list(itertools.product(vals, vals))
    legal = (lambda a, b: True) if name in ("maxui", "minui") else (lambda a, b: b != 0)
    _check_int(name, w, pairs, UNSIGNED[name], legal)


@pytest.mark.parametrize("w", WIDTHS)
@pytest.mark.parametrize("name", sorted(SIGNED))
def test_signed_integer_ops_match_python_integers(name, w):
    vals = _signed_values(w)
    pairs = list(itertools.product(vals, vals))
    int_min = -(1 << (w - 1))
    if name in ("maxsi", "minsi"):
        legal = lambda a, b: True  # noqa: E731
    else:
        legal = lambda a, b: b != 0 and not (a == int_min and b == -1)  # noqa: E731
    _check_int(name, w, pairs, SIGNED[name], legal)


@pytest.mark.parametrize("w", WIDTHS)
@pytest.mark.parametrize("name", ["shrui", "shrsi", "shli"])
def test_shifts_match_python_integers(name, w):
    vals = _unsigned_values(w) if name != "shrsi" else _signed_values(w)
    shifts = sorted({0, 1, 2, w // 2, w - 2, w - 1, w, w + 1})
    pairs = list(itertools.product(vals, shifts))
    expect = {"shrui": lambda a, s: (a % (1 << w)) >> s, "shrsi": lambda a, s: _signed(a, w) >> s,
              "shli": lambda a, s: a << s}[name]
    _check_int(name, w, pairs, expect, lambda a, s: s < w)


@pytest.mark.parametrize("w", WIDTHS)
def test_integer_compare_matches_python_integers(w):
    vals = _unsigned_values(w) + [v % (1 << w) for v in _signed_values(w)]
    pairs = list(itertools.product(sorted(set(vals)), repeat=2))
    a, b = _int_args(pairs, w)
    preds = {"ult": lambda x, y: x < y, "ule": lambda x, y: x <= y, "ugt": lambda x, y: x > y,
             "uge": lambda x, y: x >= y, "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}
    spreds = {"slt": lambda x, y: x < y, "sle": lambda x, y: x <= y, "sgt": lambda x, y: x > y,
              "sge": lambda x, y: x >= y}
    bad = []
    for pred, fn in list(preds.items()) + list(spreds.items()):
        out = E._cmpi(NS(attrs={"predicate": pred}), [a, b])
        for j, (x, y) in enumerate(pairs):
            if pred in spreds:
                want = int(fn(_signed(x, w), _signed(y, w)))
            else:
                want = int(fn(x % (1 << w), y % (1 << w)))
            if int(out.lo[j]) != want or int(out.st[j]) != ST_OK:
                bad.append((pred, x, y, want, int(out.lo[j])))
    assert not bad, bad[:6]


def test_small_integer_ops_are_unchanged():
    """ordinary control: small operands of every op, every width (no boundary argument)"""
    small = list(range(-9, 10))
    for w in WIDTHS:
        pairs = [(a, b) for a in small for b in small]
        for name, fn in SIGNED.items():
            _check_int(name, w, pairs, fn, (lambda a, b: True) if name in ("maxsi", "minsi") else (lambda a, b: b != 0))
        upairs = [(a % (1 << w), b % (1 << w)) for a in range(0, 12) for b in range(0, 12)]
        for name, fn in UNSIGNED.items():
            _check_int(name, w, upairs, fn, (lambda a, b: True) if name in ("maxui", "minui") else (lambda a, b: b != 0))


def test_unsigned_64_bit_ops_through_the_interpreter():
    """full interpreter: a compiled kernel on uint64 / int64 storages (divui, remui, maxui, shrui, cmpi ugt; divsi,
    remsi at INT64_MIN), evaluated on synthetic captures"""
    pytest.importorskip("triton")
    from test_signatures_structural import _run

    U = 1 << 64
    x = [1 << 63, U - 1, 1, (1 << 63) + 5, (1 << 53) + 1, 7, U - 2, 1 << 63]
    y = [3, 1 << 63, 1 << 63, 2, 1, 2, U - 1, (1 << 63) + 1]
    body = ("    i = tl.arange(0, N)\n    a = tl.load(x + i)\n    b = tl.load(y + i)\n"
            "    tl.store(q + i, a // b)\n    tl.store(r + i, a % b)\n    tl.store(mx + i, tl.maximum(a, b))\n"
            "    tl.store(sh + i, a >> 3)\n    tl.store(gt + i, (a > b).to(tl.uint64))\n"
            "    sa = tl.load(sx + i)\n    sb = tl.load(sy + i)\n    tl.store(sq + i, sa // sb)\n"
            "    tl.store(sr + i, sa % sb)\n")
    sx = [-(1 << 63), -(1 << 63), -(1 << 63) + 1, -(1 << 63), 1 << 62, -7, 7, -(1 << 63)]
    sy = [3, 1, -3, 2, -5, 2, -2, (1 << 63) - 1]
    z = np.zeros(8)
    bufs = {"x": ("uint64", np.array(x, dtype=np.uint64)), "y": ("uint64", np.array(y, dtype=np.uint64)),
            "q": ("uint64", z), "r": ("uint64", z), "mx": ("uint64", z), "sh": ("uint64", z), "gt": ("uint64", z),
            "sx": ("int64", sx), "sy": ("int64", sy), "sq": ("int64", z), "sr": ("int64", z)}
    ref, ident = _run("audit_u64", body, bufs, full=True)
    want = {"q": [a // b for a, b in zip(x, y)], "r": [a % b for a, b in zip(x, y)],
            "mx": [max(a, b) for a, b in zip(x, y)], "sh": [a >> 3 for a in x],
            "gt": [int(a > b) for a, b in zip(x, y)],
            "sq": [_trunc_div(a, b) for a, b in zip(sx, sy)],
            "sr": [a - b * _trunc_div(a, b) for a, b in zip(sx, sy)]}
    bad = {}
    for k, w in want.items():
        buf = ref.buffers[ident[k]]
        got = [int(v) % U for v in np.asarray(buf.lo)]
        if not (np.asarray(buf.st) == ST_OK).all() or got != [v % U for v in w]:
            bad[k] = (got, [v % U for v in w])
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ F02 scaled dot

def _ftv(lo, hi=None, elem="bf16", cond=False):
    lo = np.asarray(lo, dtype=np.float64)
    hi = lo.copy() if hi is None else np.asarray(hi, dtype=np.float64)
    return TV("f", elem, lo, hi, None, np.zeros(lo.shape, np.int8), np.full(lo.shape, cond, bool))


def _scale(e, shape, cond=False):
    return TV("i", "i8", np.full(shape, e, np.int64), None, None, np.zeros(shape, np.int8), np.full(shape, cond, bool))


def _evaluator():
    return NS(_rules=__import__("collections").Counter())


def test_scaled_dot_keeps_the_upper_end_of_an_interval_operand():
    """rule level (audit probe): a = [1, 2] per element, 1 x 32, times ones, scale 2^0: every real result in [32, 64]
    must be enclosed"""
    a = _ftv(np.ones((1, 32)), np.full((1, 32), 2.0))
    b = _ftv(np.ones((32, 1)))
    c = _ftv([[0.0]], elem="f32")
    s = _scale(127, (1, 1))
    out = E.KernelReferenceEvaluator._scaled_dot(_evaluator(), NS(node_id="audit"), a, s, b, s, c, "bf16", "bf16")
    assert out.st[0, 0] == ST_OK
    assert out.lo[0, 0] <= 32.0 and out.hi[0, 0] >= 64.0


def test_scaled_dot_output_is_conditional_when_only_a_scale_is():
    a = _ftv(np.ones((1, 32)))
    b = _ftv(np.ones((32, 1)))
    c = _ftv([[0.0]], elem="f32")
    out = E.KernelReferenceEvaluator._scaled_dot(_evaluator(), NS(node_id="audit"), a, _scale(127, (1, 1), cond=True),
                                                 b, None, c, "bf16", "bf16")
    assert bool(out.cond[0, 0])
    out = E.KernelReferenceEvaluator._scaled_dot(_evaluator(), NS(node_id="audit"), a, None, b,
                                                 _scale(127, (1, 1), cond=True), c, "bf16", "bf16")
    assert bool(out.cond[0, 0])
    out = E.KernelReferenceEvaluator._scaled_dot(_evaluator(), NS(node_id="audit"), a, _scale(127, (1, 1)), b, None, c,
                                                 "bf16", "bf16")
    assert not bool(out.cond[0, 0])   # ordinary control


def test_scaled_upcast_fp8_keeps_the_interval():
    """rule level: amdg.scaled_upcast_fp8 of an interval operand encloses x * 2^(e - 127) for every x in it"""
    x = _ftv(np.array([1.0, -3.0]), np.array([1.5, -2.0]), elem="f8E4M3FN")
    scale = _scale(129, (2,))
    op = NS(node_id="audit", name="amdg.scaled_upcast_fp8", result_types=[NS(elem="bf16", shape=(2,))], attrs={})
    ev = E.KernelReferenceEvaluator.__new__(E.KernelReferenceEvaluator)
    ev._rules = __import__("collections").Counter()
    ev.mode = E.NumericMode.NUMERICAL_DIFFERENCE
    out = ev._op_scaled_upcast_fp8(op, [x, scale], {}, None)
    assert (out.lo <= np.array([4.0, -12.0])).all() and (out.hi >= np.array([6.0, -8.0])).all()


def test_scaled_dot_of_a_computed_operand_through_the_interpreter():
    """full interpreter: a = where(x / 3 * 3 == x, 2, 1).to(bf16) is 2 in real arithmetic, but the reference cannot
    decide the comparison where x / 3 is not a float: there a is the path union [1, 2].  dot_scaled(a, bf16; b, e4m3
    with scales) must enclose the exact rational result (independent Fraction computation, not a TTIR / TTGIR
    cross-check)"""
    triton = pytest.importorskip("triton")
    import hashlib
    import importlib.util

    import signature_harness as H
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence

    src = '''import triton
import triton.language as tl


@triton.jit
def kernel(a_ptr, b_ptr, bs_ptr, out, N: tl.constexpr):
    rm = tl.arange(0, 32)
    rk = tl.arange(0, 64)
    rs = tl.arange(0, 2)
    x = tl.load(a_ptr + rm[:, None] * 64 + rk[None, :])
    a = tl.where(x / 3.0 * 3.0 == x, 2.0, 1.0).to(tl.bfloat16)
    b = tl.load(b_ptr + rk[:, None] * 32 + rm[None, :])
    bsc = tl.load(bs_ptr + rm[:, None] * 2 + rs[None, :])
    c = tl.dot_scaled(a, None, "bf16", b, bsc, "e4m3")
    tl.store(out + rm[:, None] * 32 + rm[None, :], c)
'''
    H.KDIR.mkdir(parents=True, exist_ok=True)
    path = H.KDIR / f"audit_dscaled_{hashlib.sha256(src.encode()).hexdigest()[:10]}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp8e4nv", "bs_ptr": "*u8", "out": "*fp32", "N": "constexpr"}
    # fp8e4nv is not a 3.6.0 type on sm_86: compiled (not run) for sm_90; the reference uses the TTIR only
    ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": 32}),
                        target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    rng = np.random.default_rng(7)
    xa = rng.integers(1, 64, (32, 64)).astype(np.float32)
    bt = torch.tensor(np.abs(rng.standard_normal((64, 32))) * 4 + 0.5).to(torch.float8_e4m3fn)
    b_raw, b_dec = bt.view(torch.uint8).numpy(), bt.float().numpy().astype(np.float64)
    b_s = rng.integers(124, 131, (32, 2)).astype(np.uint8)
    bufs = [("a_ptr", "float32", xa), ("b_ptr", "float8_e4m3fn", b_raw), ("bs_ptr", "uint8", b_s),
            ("out", "float32", np.zeros((32, 32), np.float32))]
    args, base = [], 1 << 20
    for i, (name, dt, arr) in enumerate(bufs):
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=name, kind="tensor", constexpr=False, signature_type=None, dtype=dt,
                                shape=arr.shape, stride=None, element_size=arr.itemsize, data_ptr=base * (i + 1),
                                storage_ptr=base * (i + 1), storage_nbytes=raw.size, storage_id=i, before=raw,
                                after=raw.copy()))
    args.append(CapturedArg(index=4, name="N", kind="int", constexpr=True, signature_type="constexpr", value=32))
    launch = CapturedLaunch(index=0, kernel_name="audit_dot_scaled", kernel_hash="", grid=(1, 1, 1), args=args,
                            asm={k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}, cubin_sha256=None, metadata={},
                            libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = ref.buffers[base * 4]
    lo, hi, st = out.lo.reshape(32, 32), out.hi.reshape(32, 32), out.st.reshape(32, 32)
    bad = []
    for i in range(0, 32, 5):
        for j in range(0, 32, 5):
            exact = sum(2 * Fr(float(b_dec[k, j])) * Fr(2) ** (int(b_s[j, k // 32]) - 127)
                        for k in range(64))
            if st[i, j] != ST_OK or not (Fr(float(lo[i, j])) <= exact <= Fr(float(hi[i, j]))):
                bad.append((i, j, float(exact), float(lo[i, j]), float(hi[i, j]), int(st[i, j])))
    assert not bad, bad[:4]
    assert any(r.startswith("path_union:") for r in ref.reasons) or (hi > lo).any()   # the operand was an interval


# ------------------------------------------------------------------------------------------------ F03 premises

def test_three_program_non_commutative_lock_is_not_a_point_answer():
    """full interpreter: three lock-protected updates x <- x + 1, 2 x, x + 1 from x = 0; program order and reverse order
    both give 3, the six serializations give {2, 3, 4}: no complete point reference may be reported"""
    pytest.importorskip("triton")
    from test_signatures_structural import _run

    body = ("    pid = tl.program_id(0)\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    v = tl.load(x)\n"
            "    v = tl.where(pid == 1, v * 2, v + 1)\n"
            "    tl.store(x, v)\n"
            "    tl.debug_barrier()\n"
            "    tl.atomic_xchg(lock, 0)\n")
    results = set()
    for order in itertools.permutations(range(3)):
        v = 0
        for p in order:
            v = 2 * v if p == 1 else v + 1
        results.add(v)
    assert results == {2, 3, 4}
    ref, ident = _run("audit_lock3", body, {"x": ("int32", [0]), "lock": ("int32", [0])}, grid=(3, 1, 1), full=True)
    cls = ref.element_classes(ident["x"])
    assert cls[0] not in ("complete_composed", "conditional_local"), cls


def test_cas_lock_certificate_proves_commuting_critical_sections():
    """full interpreter: lock-protected critical sections that commute -- a serialized add over 16 programs, and the
    layer-norm pattern (first holder stores its part, later holders add theirs, a branch on a shared counter) --
    are proved order-independent by the commutativity certificate: unconditional complete, with the proof recorded"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    body = ("    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    ref, ident = _run("audit_lock16", body, {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])}, grid=(16, 1, 1),
                      full=True)
    buf = ref.buffers[ident["data"]]
    assert (np.asarray(buf.st) == ST_OK).all() and (np.asarray(buf.lo) == 16).all()
    assert (ref.element_classes(ident["data"]) == "complete_composed").all()
    assert any(r.startswith("proved:the launch result does not depend") for r in ref.reasons), sorted(ref.reasons)
    ln = ("    pid = tl.program_id(0)\n    i = tl.arange(0, N)\n    part = tl.load(x + pid * N + i)\n"
          "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n    count = tl.load(cnt)\n"
          "    if count == 0:\n        tl.atomic_xchg(cnt, 1)\n    else:\n        part += tl.load(dw + i)\n"
          "    tl.store(dw + i, part)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    x = np.random.default_rng(4).standard_normal((16, 8)).astype(np.float32)
    ref, ident = _run("audit_lock_ln", ln, {"x": ("fp32", x.reshape(-1)), "dw": ("fp32", np.zeros(8)),
                                            "cnt": ("int32", [0]), "lock": ("int32", [0])}, grid=(16, 1, 1), full=True)
    assert (ref.element_classes(ident["dw"]) == "complete_composed").all()
    dw = ref.buffers[ident["dw"]]
    for j in range(8):
        exact = sum(Fr(float(v)) for v in x[:, j])
        assert Fr(float(dw.lo[j])) <= exact <= Fr(float(dw.hi[j]))


def test_cas_order_agreement_without_commutativity_stays_a_premise():
    """full interpreter: six programs, x <- x + 1 or 2x by a palindromic program type, so program order and reverse
    order agree but other orders do not; the sections do not commute, the certificate is refused and the value holds
    only under the premise -- never in the unconditional complete class"""
    pytest.importorskip("triton")
    from test_signatures_structural import _run

    body = ("    pid = tl.program_id(0)\n    t = tl.minimum(pid, 5 - pid) % 2\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    v = tl.load(xs)\n    v = tl.where(t == 1, v * 2.0, v + 1.0)\n    tl.store(xs, v)\n"
            "    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    ref, ident = _run("audit_lock_pal", body, {"xs": ("fp32", [0.0]), "lock": ("int32", [0])}, grid=(6, 1, 1),
                      full=True)
    assert (ref.element_classes(ident["xs"]) == "complete_under_premise").all()
    assert ref.compare()["xs"]["classes"].get("complete_under_premise") == 1
    assert any(r.startswith("assumed:the launch result does not depend") for r in ref.reasons)


def test_scan_premise_unless_a_certificate_proves_associativity():
    """full interpreter: a generic associative_scan whose combine region z3 proves associative (cummax with index
    tie-break) is unconditional, with the proof recorded; a region that is not associative in general (the official
    test's "roll", refuted by z3) keeps the premise even where the two bracketings agree on these inputs; a plain
    cumsum (associativity of real addition) is unconditional (ordinary control)"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    comb = "    take = v2 > v1\n    return tl.where(take, v2, v1), tl.where(take, i2, i1)\n"
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n"
            "    v, k = tl.associative_scan((x, i), 0, _cm)\n    tl.store(out + i, k)\n"
            "\n\n@triton.jit\ndef _cm(v1, i1, v2, i2):\n" + comb)
    x = np.asarray([3, 1, 4, 1, 5, 9, 2, 6], np.int32)
    ref, ident = _run("scan_cummax", body, {"x_ptr": ("int32", x), "out": ("int32", np.zeros(8))}, full=True)
    assert (ref.element_classes(ident["out"]) == "complete_composed").all()
    assert any(r.startswith("proved:scan combine associative") for r in ref.reasons), sorted(ref.reasons)
    roll = ("    i = tl.arange(0, N)\n    f = tl.load(f_ptr + i)\n    x = tl.load(x_ptr + i)\n"
            "    a, l, c = tl.associative_scan((f, x, x), 0, _cm)\n    tl.store(out + i, l)\n"
            "\n\n@triton.jit\ndef _cm(a1, l1, c1, a2, l2, c2):\n"
            "    return a1 + a2, tl.where(a2 == 1, c1, 0) + l2, c2\n")
    ref, ident = _run("audit_scan_roll", roll, {"f_ptr": ("int32", np.zeros(8)), "x_ptr": ("int32", x),
                                                "out": ("int32", np.zeros(8))}, full=True)
    assert (ref.element_classes(ident["out"]) == "complete_under_premise").all()
    assert any(r.startswith("assumed:scan combine associative") for r in ref.reasons), sorted(ref.reasons)
    body = "    i = tl.arange(0, N)\n    tl.store(out + i, tl.cumsum(tl.load(x_ptr + i), 0))\n"
    ref, ident = _run("audit_cumsum", body, {"x_ptr": ("fp32", x.astype(np.float32)), "out": ("fp32", np.zeros(8))},
                      full=True)
    assert (ref.element_classes(ident["out"]) == "complete_composed").all()


def test_scan_certificate_unit_cases():
    """rule level: z3 proves cumprod, the linear recurrence and take-first associative, refutes roll and a + b / 2,
    and gives no certificate for an operation outside the translated set"""
    pytest.importorskip("z3")
    from kernel_analyzer.reference_eval import certificates as C
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    def region(args, body, ret):
        text = ("module {\n  tt.func public @k(%x: tensor<8xf32>) {\n    %c1 = arith.constant 1 : i32\n"
                "    %c0 = arith.constant 0 : i32\n"
                f"    %r = \"tt.scan\"(%x) <{{axis = 0 : i32, reverse = false}}> ({{\n    ^bb0({args}):\n{body}"
                f"      tt.scan.return {ret}\n    }}) : (tensor<8xf32>) -> tensor<8xf32>\n    tt.return\n  }}\n}}\n")
        mod = parse_ttir(text)
        op = next(o for o in mod.entry().walk() if o.name == "tt.scan")
        return op.regions[0]
    const = {"%c1": ("const", "i32", 1), "%c0": ("const", "i32", 0)}
    lit = lambda o: int(o.attrs["value"].split(":")[0])  # noqa: E731
    cases = {
        "prod": (region("%a: f32, %b: f32", "      %p = arith.mulf %a, %b : f32\n", "%p : f32"), True),
        "first": (region("%a: f32, %b: f32", "", "%a : f32"), True),
        "linrec": (region("%a1: f32, %b1: f32, %a2: f32, %b2: f32",
                          "      %m = arith.mulf %a1, %a2 : f32\n      %n = arith.mulf %b1, %a2 : f32\n"
                          "      %s = arith.addf %n, %b2 : f32\n", "%m, %s : f32, f32"), True),
        "half": (region("%a: f32, %b: f32", "      %h = arith.mulf %b, %b : f32\n      %s = arith.subf %a, %h : f32\n",
                        "%s : f32"), False),
        "roll": (region("%a1: i32, %l1: i32, %c1_: i32, %a2: i32, %l2: i32, %c2: i32",
                        "      %s = arith.addi %a1, %a2 : i32\n      %e = arith.cmpi eq, %a2, %c1 : i32\n"
                        "      %w = arith.select %e, %c1_, %c0 : i32\n      %t = arith.addi %w, %l2 : i32\n",
                        "%s, %t, %c2 : i32, i32, i32"), False),
        "div": (region("%a: f32, %b: f32", "      %d = arith.divf %a, %b : f32\n", "%d : f32"), False),
    }
    got = {k: C.scan_associativity(r, {n: v for n, v in const.items() if n in C.outer_names(r)}, lit).proved
           for k, (r, _) in cases.items()}
    assert got == {k: want for k, (_, want) in cases.items()}, got
    assert "not translated" in C.scan_associativity(cases["div"][0], {}, lit).detail


def test_audit_order_probe_on_the_production_function():
    """rule level (audit probe): the production agreement function with the exact serial executor of the three
    programs must not keep 3 as an established point"""
    import collections
    import copy

    class Mini:
        def __init__(self, x, written=True):
            self.lo = np.array([x], np.int64)
            self.hi = None
            self.st = np.array([ST_OK], np.int8)
            self.kind = "i"
            self.written = np.array([written])
            self.cond = np.array([False])

        def copy(self):
            return copy.deepcopy(self)

    def exact(order):
        x = 0
        for i in order:
            x = 2 * x if i == 1 else x + 1
        return x

    def run(programs, grid, bindings, memory, order):
        memory[1] = Mini(exact(order))
        return {}, collections.Counter()

    ctx = NS(_rules=collections.Counter(), _run_programs=run)
    mem = {1: Mini(exact([0, 1, 2]))}
    E.KernelReferenceEvaluator._cas_reverse_order(ctx, [(0,), (1,), (2,)], (3, 1, 1), {}, mem, {1: Mini(0, False)},
                                                  {}, collections.Counter())
    assert int(mem[1].st[0]) != ST_OK


# ------------------------------------------------------------------------------------------------ F07 statistics

def _row(seed, k, reps, reasons=None, atomic=None):
    n = k.size
    return {"seed": seed, "k": k, "k_reps": reps, "r_lo": k, "r_hi": k, "ok": np.ones(n, bool),
            "written": np.ones(n, bool), "repeat_inputs_differ": False, "reasons": reasons or {},
            **({} if atomic is None else {"atomic_written": np.asarray(atomic, bool)})}


UNKNOWN = {"not_established:execution validity: same-program store then load without a barrier, thread 3 reads the "
           "element thread 1 wrote@%5": 1}


def test_unknown_execution_validity_withholds_statistics():
    from kernel_analyzer.check import _execution_status
    k = np.array([1.0, 2.0])
    st = _execution_status([_row(0, k, [k.copy()], UNKNOWN)], [{"float_atomics": False}], 2)
    assert st["statistics"] == "withheld", st
    st = _execution_status([_row(0, k, [k + 1e-7], UNKNOWN, atomic=[True, True])], [{"float_atomics": True}], 8)
    assert st["statistics"] == "withheld", st
    probe = {"not_established:execution validity not established@probe": 1}   # the audit's own probe text
    st = _execution_status([_row(0, k, [k.copy()], probe)], [{"float_atomics": False}], 2)
    assert st["statistics"] == "withheld", st


def test_float_atomic_in_the_call_does_not_admit_differences_elsewhere():
    """within-input mean only where the differing elements are written by float atomics"""
    from kernel_analyzer.check import _execution_status
    k = np.array([1.0, 2.0])
    st = _execution_status([_row(0, k, [k + np.array([0.0, 1e-7])], atomic=[True, False])],
                           [{"float_atomics": True}], 8)
    assert st["statistics"] == "withheld", st
    st = _execution_status([_row(0, k, [k + np.array([1e-7, 0.0])], atomic=[True, False])],
                           [{"float_atomics": True}], 8)
    assert st["statistics"] == "within-input mean", st   # ordinary control: the atomic element differs
    st = _execution_status([_row(0, k, [k + 1e-7])], [{"float_atomics": True}], 8)
    assert st["statistics"] == "withheld", st             # no element-level atomic evidence


@pytest.mark.parametrize("reason", [
    "not_established:execution race@probe",     # the audit's own probe text (no colon)
    "cross-program write race at %7",
    "not_established:cross-program race on load@%9",
    "not_established:execution race: an address one program read is written by another program of the same launch",
    "conflicting lanes in one store at %4",
])
def test_every_race_producer_withholds_statistics(reason):
    from kernel_analyzer.check import _execution_status
    k = np.array([1.0, 2.0])
    st = _execution_status([_row(0, k, [k.copy()], {reason: 1})], [{"float_atomics": False}], 2)
    assert st["statistics"] == "withheld", (reason, st)


def test_unknown_validity_from_the_interpreter_reaches_the_statistics_gate():
    """full interpreter -> admission: the evaluator's own finding for a same-program store then load by another
    element without a barrier.  Without the TTGIR the thread mapping is unknown (execution validity not established);
    with it the reader is another thread (execution race).  Both withhold the statistics.  The kernel is evaluated
    on synthetic captures only (not run on the GPU)."""
    pytest.importorskip("triton")
    from test_signatures_structural import _run

    from kernel_analyzer.check import _execution_status
    body = ("    i = tl.arange(0, N)\n    tl.store(buf + i, tl.load(x + i))\n"
            "    tl.store(out + i, tl.load(buf + (N - 1 - i)))\n")
    bufs = {"x": ("fp32", np.arange(8.0)), "buf": ("fp32", np.zeros(8)), "out": ("fp32", np.zeros(8))}
    k = np.zeros(8)
    for drop, marker in ((True, "execution validity"), (False, "execution race")):
        ref, ident = _run("audit_validity", body, bufs, full=True, drop_ttgir=drop)
        assert [r for r in ref.reasons if marker in r], sorted(ref.reasons)
        st = _execution_status([_row(0, k, [k.copy()], dict(ref.reasons))], [{"float_atomics": False}], 2)
        assert st["statistics"] == "withheld", (marker, st)


# ------------------------------------------------------------------------------------------------ F06 declaration

def _decl(tmp_path, **extra):
    call = tmp_path / "call.py"
    if not call.exists():
        call.write_text("def f(inp):\n    return {'y': inp['x']}\n")
    d = {"call": "call.py:f", "inputs": {"x": {"sampler": {"normal": [0, 1]}, "shape": [8], "dtype": "float32"}},
         "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 60, "gpu_seconds": 60, "case_timeout": 30, "max_units": 12},
         "_base_dir": str(tmp_path)}
    d.update(extra)
    return d


@pytest.mark.parametrize("change", [
    {"alpha": 0.01}, {"units": {"development": 8, "confirmation": 16}}, {"resolution": {"ulp_fraction": 0.5}},
    {"magnitude_bound": {"elementwise": 1e-3, "basis": "declared"}}, {"equivalence": {"rel": 1e-3, "basis": "x"}},
    {"repeats": 4}, {"units": {"seed_offset": 100}},
])
def test_declaration_digest_binds_every_setting(tmp_path, change):
    from kernel_analyzer import measure
    base = measure.expand(_decl(tmp_path))
    assert measure.expand(_decl(tmp_path))["declaration_sha256"] == base["declaration_sha256"]   # reproducible
    assert measure.expand(_decl(tmp_path, **change))["declaration_sha256"] != base["declaration_sha256"]


def test_declaration_digest_binds_the_source_of_the_call(tmp_path):
    from kernel_analyzer import measure
    base = measure.expand(_decl(tmp_path))
    (tmp_path / "call.py").write_text("def f(inp):\n    return {'y': inp['x'] * 1}\n")
    assert measure.expand(_decl(tmp_path))["declaration_sha256"] != base["declaration_sha256"]


def test_sampling_stream_does_not_depend_on_the_statistics_settings(tmp_path):
    """changing alpha must not change the sampled inputs (same units, same inputs, comparable results)"""
    from kernel_analyzer import measure
    a = measure.make_inputs(measure.expand(_decl(tmp_path)), {}, 3, device="cpu")["x"]
    b = measure.make_inputs(measure.expand(_decl(tmp_path, alpha=0.01)), {}, 3, device="cpu")["x"]
    assert torch.equal(a, b)


def test_settings_reach_the_analysis(tmp_path, monkeypatch):
    """alpha, delta, repeats and M are passed to check.run (statistics layer), not only written into the JSON"""
    from kernel_analyzer import check, measure
    seen = {}

    def fake_run(case, **kw):
        seen.update(kw)
        return {"outputs": {}}
    monkeypatch.setattr(check, "run", fake_run)
    exp = measure.expand(_decl(tmp_path, alpha=0.01, equivalence={"rel": 1e-3, "basis": "declared"}, repeats=3,
                               magnitude_bound={"elementwise": 0.5, "basis": "declared"}))
    measure.run_level(exp, {})
    assert seen["alpha"] == 0.01 and seen["equivalence_rel"] == 1e-3 and seen["repeats"] == 3
    assert seen["magnitude_bound"]["elementwise"] == 0.5


def test_bounded_route_text_matches_the_behaviour(tmp_path):
    from kernel_analyzer import measure
    exp = measure.expand(_decl(tmp_path))
    text = exp["bounded_route"]
    assert not text.startswith("recorded only") and "passed to check.run" in text and "magnitude_bound" in text


def test_missing_items_give_a_structured_result(tmp_path):
    from kernel_analyzer import measure
    d = _decl(tmp_path)
    del d["inputs"]["x"]["shape"]
    with pytest.raises(measure.MissingDeclaration) as exc:
        measure.expand(d)
    assert exc.value.items == ["inputs.x.shape"]


# ------------------------------------------------------------------------------------------------ GPU end to end
# F04 (provenance), F05 (Gluon through the unified entry), F07 on a float-atomic call; measure.run -> report

def _e2e(tmp_path, call, x, **extra):
    from kernel_analyzer import measure
    d = {"call": f"audit_calls.py:{call}", "inputs": {"x": x}, "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 6},
         "units": {"development": 2, "confirmation": 4}, "_base_dir": str(HERE)}
    d.update(extra)
    rep = measure.run(d, out=str(tmp_path / "report.json"))
    json.loads((tmp_path / "report.json").read_text())   # the report is plain JSON
    assert len(rep["levels"]) == 1
    return rep, rep["levels"][0]


X1D = {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}


@CUDA
def test_classic_kernel_through_the_unified_entry(tmp_path):
    rep, lv = _e2e(tmp_path, "classic", X1D)
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert y["status"] == "evaluated" and y["reference"]["complete_rate"] == 1.0
    assert y["reference"]["reference_scope"] == "call-level"
    assert rep["expanded_declaration"]["declaration_sha256"]


@CUDA
def test_gluon_kernel_through_the_unified_entry(tmp_path):
    """F05: a TTGIR-only launch goes through measure.run -> check.run -> report, with the IR kind kept"""
    rep, lv = _e2e(tmp_path, "gluon_affine", X1D)
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert y["status"] == "evaluated" and y["reference"]["complete_rate"] == 1.0, y
    assert lv["notes"]["ir_kinds"] == ["ttgir"]
    assert lv["notes"]["ir_coverage_complete"] == [True]


@CUDA
def test_float_atomic_kernel_through_the_unified_entry(tmp_path):
    rep, lv = _e2e(tmp_path, "atomic_row_sum", {"sampler": {"normal": [0, 1]}, "shape": [8, 512], "dtype": "float32"})
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert y["status"] == "evaluated" and y["reference"]["complete_rate"] == 1.0, y


@CUDA
@pytest.mark.parametrize("call", ["value_equal_upstream", "zero_upstream", "compiled_after_copy"])
def test_upstream_values_equal_to_inputs_are_not_promoted_to_call_level(tmp_path, call):
    """F04: neither a value-equal computed buffer nor an all-zero underflowed buffer proves provenance (their producers
    are arithmetic), and a copy read by a compiled kernel has no aligned producer record: the reference stays
    kernel-level"""
    rep, lv = _e2e(tmp_path, call, X1D)
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert y["mixed_non_triton_sources"], y
    assert y["reference"]["reference_scope"].startswith("kernel-level"), y["reference"]
    assert y["reference"]["complete_rate_call_level"] == 0.0


@CUDA
@pytest.mark.parametrize("call,producer", [("honest_copy", "copy: aten.clone"), ("atomic_row_sum", "const: aten.zeros")])
def test_upstream_copies_and_constants_with_producer_records_are_call_level(tmp_path, call, producer):
    """F04 follow-up: a true copy of the input (ATen layout change) and an accumulator zero-filled by ATen have real
    producers in the traced run (copy of the declared input / exact constant): call-level, with the record named"""
    x = {"sampler": {"normal": [0, 1]}, "shape": [8, 512], "dtype": "float32"} if call == "atomic_row_sum" else X1D
    rep, lv = _e2e(tmp_path, call, x)
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert not y["mixed_non_triton_sources"], y
    assert any(producer in r for r in y["upstream_with_producer_record"]), y["upstream_with_producer_record"]
    assert y["reference"]["reference_scope"].startswith("call-level (upstream copies"), y["reference"]
    assert y["reference"]["complete_rate_call_level"] == y["reference"]["complete_rate"] == 1.0


def test_producer_classification_of_aten_ops():
    """rule level (CPU tensors): copies of inputs and exact constants are clean; arithmetic, a constant that rounds in
    the dtype, a dtype change and an in-place update of an input are computed"""
    from types import SimpleNamespace

    from kernel_analyzer import provenance as P
    rec = SimpleNamespace(launches=[])
    x = torch.rand(16) + 1.0
    prov = {x.untyped_storage().data_ptr(): ("input", "declared input")}
    tr = P._Trace(rec)
    with tr.mode:
        a = x.reshape(4, 4).t().contiguous()
        b = (x[:1] + 2.0 ** -25).repeat(16)
        c = torch.full((4,), 0.1)
        d = torch.full((4,), 0.5)
        e = x.to(torch.float16)
        f = torch.cat([x, d])
        x.mul_(0.5)
    for ev in tr.events:
        P._apply(prov, ev)
    status = {k: prov[t.untyped_storage().data_ptr()][0] for k, t in
              dict(a=a, b=b, c=c, d=d, e=e, f=f, x=x).items()}
    assert status == {"a": "copy", "b": "computed", "c": "computed", "d": "const", "e": "computed", "f": "copy",
                      "x": "computed"}, status


def test_float_atomic_mark_is_kept_until_a_plain_store():
    """full interpreter: the element-level float-atomic mark is a buffer field (not the per-launch writer, which every
    launch resets) and a plain store clears it"""
    pytest.importorskip("triton")
    from test_signatures_structural import _run

    body = "    i = tl.arange(0, N)\n    tl.atomic_add(out + i, tl.load(x + i))\n"
    ref, ident = _run("audit_atomic_mark", body, {"x": ("fp32", np.arange(8.0)), "out": ("fp32", np.zeros(8))},
                      full=True)
    assert np.asarray(ref.buffers[ident["out"]].float_atomic).all()
    body = "    i = tl.arange(0, N)\n    tl.atomic_add(out + i, tl.load(x + i))\n    tl.store(out + i, tl.load(x + i))\n"
    ref, ident = _run("audit_atomic_then_store", body, {"x": ("fp32", np.arange(8.0)), "out": ("fp32", np.zeros(8))},
                      full=True)
    assert not np.asarray(ref.buffers[ident["out"]].float_atomic).any()


@CUDA
def test_budget_exhaustion_gives_a_structured_result(tmp_path):
    rep, lv = _e2e(tmp_path, "slow", X1D, budget={"cpu_seconds": 60, "gpu_seconds": 60, "case_timeout": 1,
                                                  "max_units": 6})
    assert lv["status"] == "over budget" and lv["failure_class"] == "over budget", lv
    assert rep["failure_counts"]["over budget"] == 1


@CUDA
def test_output_without_a_reference_gives_a_structured_result(tmp_path):
    rep, lv = _e2e(tmp_path, "aten_output", X1D)
    assert lv["status"] == "ok", lv
    y = lv["outputs"]["y"]
    assert y["status"] == "not established" and y["reason"] == "not written by Triton" and y["failure_class"] == "binding"


@CUDA
def test_heavy_imports_happen_before_the_case_timeout_is_armed(tmp_path):
    """found while fixing (F05 / F06 entry checks): the case timeout (SIGALRM) was armed before the case setup, whose
    first ``torch._dynamo`` import takes about a second here; a one-second budget interrupted that import and left a
    partially initialised module that broke every later level of the process.  In a fresh interpreter the imports
    must already be done when the alarm is armed."""
    import subprocess
    import sys
    import textwrap
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        sys.path[:0] = [{str(HERE.parent / "src")!r}]
        from kernel_analyzer import measure
        seen = []
        orig = measure._alarm
        def spy(seconds):
            seen.append(all(m in sys.modules for m in ("torch._dynamo", "torch._inductor.config")))
            return orig(seconds)
        measure._alarm = spy
        d = {{"call": "audit_calls.py:classic", "inputs": {{"x": {X1D!r}}}, "compare": {{"mode": "A", "measure": ["y"]}},
              "budget": {{"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 3}},
              "units": {{"development": 1, "confirmation": 2}}, "_base_dir": {str(HERE)!r}}}
        rep = measure.run(d)
        print(json.dumps({{"armed_after_imports": seen, "status": rep["levels"][0]["status"]}}))
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"armed_after_imports": [True], "status": "ok"}, (res, out.stderr[-2000:])


# ------------------------------------------------------------------------------------------------ audit section 5.3

def test_bounded_route_does_not_call_an_observed_width_a_pre_data_guarantee():
    """audit section 5.3: the detectable effect of the bounded route used the mean endpoint width observed in this run;
    it is a sensitivity conditional on those widths.  A pre-data value needs a declared width bound W: 2 r_n + W."""
    import math

    from kernel_analyzer.reference_eval.sensitivity import bounded_route, sensitivity_fields
    rng = np.random.default_rng(1)
    l = rng.normal(0.0, 0.1, 64)
    h = l + 0.05
    r = bounded_route(l, h, 1.0, 0.05)
    assert "guaranteed_detectable_effect" not in r
    rn = 1.0 * math.sqrt(2.0 * math.log(2.0 / 0.05) / 64)
    assert abs(r["detectable_effect_given_observed_widths"] - (2 * rn + 0.05)) < 1e-12
    assert "not a pre-data" in r["detectable_effect_given_observed_widths_meaning"]
    assert r["pre_data_detectable_effect"] is None
    r = bounded_route(l, h, 1.0, 0.05, width_bound=0.1)
    assert abs(r["pre_data_detectable_effect"] - (2 * rn + 0.1)) < 1e-12
    s = sensitivity_fields(l, h, 0.05)
    assert "post-data" in s["mde_approximate"]["meaning"]


# ------------------------------------------------------------------------------------------------ W2 / W6 precision

def test_compensated_prefix_sums_are_rigorous_and_tighter():
    """rule level: at working-precision level 2 the prefix sums use a per-prefix compensated (Sum2) bound, intersected
    with the gamma bound: every enclosure contains the exact rational prefix, and cancelling prefixes get much
    narrower enclosures than at level 1"""
    from kernel_analyzer.reference_eval import intervals as iv
    rng = np.random.default_rng(3)
    x = (np.where(np.arange(64) % 2 == 0, 2.0 ** 20, -2.0 ** 20) + rng.uniform(0.5, 1.5, 64)).reshape(1, 64)
    x = x.astype(np.float32).astype(np.float64)
    exact = np.cumsum([Fr(float(v)) for v in x[0]])
    with iv.working_precision(1):
        lo1, hi1 = iv.icumsum(x, x, 1)
    with iv.working_precision(2):
        lo2, hi2 = iv.icumsum(x, x, 1)
        rlo, rhi = iv.icumsum(x[:, ::-1].copy(), x[:, ::-1].copy(), 1, reverse=True)
    assert all(Fr(float(lo2[0, j])) <= exact[j] <= Fr(float(hi2[0, j])) for j in range(64))
    yf = [Fr(float(v)) for v in x[0, ::-1]]                         # the scanned row (x flipped)
    rexact = [sum(yf[j:], Fr(0)) for j in range(64)]                # reverse scan: suffix sums
    assert all(Fr(float(rlo[0, j])) <= rexact[j] <= Fr(float(rhi[0, j])) for j in range(64))
    assert np.max(hi2 - lo2) < 1e-3 * np.max(hi1 - lo1)
    assert iv.precision_level() == 1


@CUDA
def test_refinement_raises_the_working_precision_until_the_resolution_is_met(tmp_path):
    """W2 / W6 (audit task book section 5): a requested resolution that level 1 misses is recomputed at higher working
    precision; the report states each level, the outcome and the level that met the target"""
    rep, lv = _e2e(tmp_path, "cumsum_rows", {"state": "audit_calls.py:cancelling_rows", "shape": [4, 256],
                                              "dtype": "float32"}, resolution={"ulp_fraction": 0.125})
    assert lv["status"] == "ok", lv
    ref = lv["outputs"]["y"]["reference"]
    steps = lv["refinement"]["levels"]
    assert steps[0]["level"] == 1 and steps[0]["resolution_met"] is False, steps
    assert lv["refinement"]["outcome"].startswith("met at level"), lv["refinement"]
    assert ref["resolution_met"] is True and lv["refinement"]["final_level"] >= 2


@CUDA
def test_refinement_reports_a_target_it_cannot_reach(tmp_path):
    """a target no working precision reaches gives an explicit outcome, never a met resolution"""
    rep, lv = _e2e(tmp_path, "cumsum_rows", {"state": "audit_calls.py:cancelling_rows", "shape": [4, 256],
                                              "dtype": "float32"}, resolution={"ulp_fraction": 1e-12, "max_level": 2})
    assert lv["status"] == "ok", lv
    assert lv["outputs"]["y"]["reference"]["resolution_met"] is False
    assert lv["refinement"]["outcome"].startswith(("not met", "no improvement")), lv["refinement"]


def test_cas_lock_certificate_refuses_an_aliased_lock_word():
    """full interpreter: a second pointer argument bound to the lock storage resets the lock word before the acquire
    loop (outside the critical section, so the section footprint check does not see it).  On a device that breaks
    mutual exclusion; the serial evaluation does not show it.  The lock word has a writer other than its acquire and
    release: no certificate."""
    triton = pytest.importorskip("triton")
    pytest.importorskip("z3")

    body = ("    i = tl.arange(0, N)\n    tl.store(other, 0)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n"
            "    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    from test_signatures_structural import C, _kernel
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    mod = _kernel("audit_lock_alias", body, ["data", "lock", "other"])
    sig = {"data": "*fp32", "lock": "*i32", "other": "*i32", "N": "constexpr"}
    ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": C}), target=GPUTarget("cuda", 86, 32),
                        options={"num_warps": 1})
    raw_d = np.zeros(8, np.float32).view(np.uint8)
    raw_l = np.zeros(1, np.int32).view(np.uint8)
    args = [CapturedArg(index=0, name="data", kind="tensor", constexpr=False, signature_type="*fp32", dtype="float32",
                        shape=(8,), stride=None, element_size=4, data_ptr=1 << 20, storage_ptr=1 << 20,
                        storage_nbytes=32, storage_id=0, before=raw_d.copy(), after=raw_d.copy())]
    for k, name in ((1, "lock"), (2, "other")):     # both pointers bound to the same lock storage
        args.append(CapturedArg(index=k, name=name, kind="tensor", constexpr=False, signature_type="*i32",
                                dtype="int32", shape=(1,), stride=None, element_size=4, data_ptr=2 << 20,
                                storage_ptr=2 << 20, storage_nbytes=4, storage_id=1, before=raw_l.copy(),
                                after=raw_l.copy()))
    args.append(CapturedArg(index=3, name="N", kind="int", constexpr=True, signature_type="constexpr", value=C))
    launch = CapturedLaunch(index=0, kernel_name="alias", kernel_hash="", grid=(8, 1, 1), args=args,
                            asm={k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}, cubin_sha256=None, metadata={},
                            libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    assert not any(r.startswith("proved:the launch result") for r in ref.reasons), sorted(ref.reasons)
    assert not (ref.element_classes(1 << 20) == "complete_composed").any()


def test_cas_lock_certificate_with_the_official_lock_values():
    """full interpreter: the official test_atomic_cas builds its lock values with tl.full((1,), v).item() (a one-element
    dense constant and tt.unsplat); the pattern recognizes them and the 128-lane serialized add is certified"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    body = ("    num0 = tl.full((1, ), 0, dtype=tl.int32).item()\n    num1 = tl.full((1, ), 1, dtype=tl.int32).item()\n"
            "    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, num0, num1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, num0)\n")
    ref, ident = _run("audit_lock_item", body, {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                      grid=(12, 1, 1), full=True)
    assert (np.asarray(ref.buffers[ident["data"]].lo) == 12).all()
    assert (ref.element_classes(ident["data"]) == "complete_composed").all(), sorted(ref.reasons)


def test_cas_lock_certificate_for_the_tutorial_layer_norm_shape():
    """full interpreter: tutorial 05's backward pass -- lock words and counters in one tensor (Count = Lock + G, derived
    from the lock pointer), a masked load without other, first holder stores, later holders add: certified"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    body = ("    pid = tl.program_id(0)\n    i = tl.arange(0, N)\n    m = i < 6\n"
            "    part = tl.load(x + pid * N + i, mask=m, other=0.0)\n"
            "    lk = locks + pid % 2\n    cnt = lk + 2\n"
            "    while tl.atomic_cas(lk, 0, 1) == 1:\n        pass\n"
            "    count = tl.load(cnt)\n"
            "    if count == 0:\n        tl.atomic_xchg(cnt, 1)\n"
            "    else:\n        part += tl.load(dw + (pid % 2) * N + i, mask=m)\n"
            "    tl.store(dw + (pid % 2) * N + i, part, mask=m)\n"
            "    tl.debug_barrier()\n    tl.atomic_xchg(lk, 0)\n")
    x = np.random.default_rng(5).standard_normal((10, 8)).astype(np.float32)
    ref, ident = _run("audit_lock_tut05", body, {"x": ("fp32", x.reshape(-1)), "dw": ("fp32", np.zeros(16)),
                                                 "locks": ("int32", np.zeros(4))}, out_name="dw", grid=(10, 1, 1),
                      full=True)
    cls = ref.element_classes(ident["dw"]).reshape(2, 8)
    assert (cls[:, :6] == "complete_composed").all(), (cls, sorted(ref.reasons))
    dw = ref.buffers[ident["dw"]]
    for g in range(2):
        for j in range(6):
            exact = sum(Fr(float(v)) for v in x[g::2, j])
            assert Fr(float(dw.lo[g * 8 + j])) <= exact <= Fr(float(dw.hi[g * 8 + j]))


def test_cas_lock_certificate_with_the_official_main_barrier():
    """full interpreter: official main lowers tl.debug_barrier to ttg.barrier (3.6.0: gpu.barrier); the same lock kernel
    with its barrier spelled ttg.barrier is certified too (found by the broad capture on official main)"""
    triton = pytest.importorskip("triton")
    pytest.importorskip("z3")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    from test_signatures_structural import C, _kernel
    body = ("    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    mod = _kernel("audit_lock_ttgbarrier", body, ["data", "lock"])
    ck = triton.compile(ASTSource(fn=mod.kernel, signature={"data": "*fp32", "lock": "*i32", "N": "constexpr"},
                                  constexprs={"N": C}), target=GPUTarget("cuda", 86, 32), options={"num_warps": 1})
    ttir = ck.asm["ttir"]
    assert "gpu.barrier" in ttir
    ttir = ttir.replace("gpu.barrier", "ttg.barrier")
    raw_d, raw_l = np.zeros(C, np.float32).view(np.uint8), np.zeros(1, np.int32).view(np.uint8)
    args = [CapturedArg(index=0, name="data", kind="tensor", constexpr=False, signature_type="*fp32", dtype="float32",
                        shape=(C,), stride=None, element_size=4, data_ptr=1 << 20, storage_ptr=1 << 20,
                        storage_nbytes=4 * C, storage_id=0, before=raw_d.copy(), after=raw_d.copy()),
            CapturedArg(index=1, name="lock", kind="tensor", constexpr=False, signature_type="*i32", dtype="int32",
                        shape=(1,), stride=None, element_size=4, data_ptr=2 << 20, storage_ptr=2 << 20,
                        storage_nbytes=4, storage_id=1, before=raw_l.copy(), after=raw_l.copy()),
            CapturedArg(index=2, name="N", kind="int", constexpr=True, signature_type="constexpr", value=C)]
    launch = CapturedLaunch(index=0, kernel_name="lock_ttg", kernel_hash="", grid=(9, 1, 1), args=args,
                            asm={"ttir": ttir, "ttgir": ck.asm["ttgir"]}, cubin_sha256=None, metadata={},
                            libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    assert (ref.element_classes(1 << 20) == "complete_composed").all(), sorted(ref.reasons)


def test_cas_lock_certificate_on_amd_buffer_ops():
    """full interpreter (official AMD gfx942 TTGIR fixture of the official test_atomic_cas kernel): the critical
    section uses amdg.buffer_load / amdg.buffer_store; the replay follows the evaluator's address rule and certifies
    the 50-program serialized add (found by the AMD cross-level capture: the TTIR side was certified, this side not)"""
    pytest.importorskip("z3")
    from pathlib import Path

    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    ttgir = (Path(__file__).parent / "data/amd_ttgir/serialized_add_gfx942.ttgir").read_text()
    assert "amdg.buffer_load" in ttgir and "amdg.buffer_store" in ttgir
    raw_d, raw_l = np.zeros(128, np.float32).view(np.uint8), np.zeros(1, np.int32).view(np.uint8)
    args = [CapturedArg(index=0, name="data", kind="tensor", constexpr=False, signature_type="*fp32", dtype="float32",
                        shape=(128,), stride=None, element_size=4, data_ptr=1 << 20, storage_ptr=1 << 20,
                        storage_nbytes=512, storage_id=0, before=raw_d.copy(), after=raw_d.copy()),
            CapturedArg(index=1, name="Lock", kind="tensor", constexpr=False, signature_type="*i32", dtype="int32",
                        shape=(1,), stride=None, element_size=4, data_ptr=2 << 20, storage_ptr=2 << 20,
                        storage_nbytes=4, storage_id=1, before=raw_l.copy(), after=raw_l.copy()),
            CapturedArg(index=2, name="triton_dtype", kind="int", constexpr=True, signature_type="constexpr", value=0),
            CapturedArg(index=3, name="SEM", kind="int", constexpr=True, signature_type="constexpr", value=0)]
    launch = CapturedLaunch(index=0, kernel_name="serialized_add", kernel_hash="", grid=(50, 1, 1), args=args,
                            asm={"ttgir": ttgir}, cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    assert (np.asarray(ref.buffers[1 << 20].lo) == 50).all()
    assert (ref.element_classes(1 << 20) == "complete_composed").all(), sorted(ref.reasons)


def test_cas_lock_certificate_with_float_width_changes():
    """full interpreter: tutorial 05 keeps its partial sums in fp16 and widens them (arith.extf) inside the critical
    section; in the numerical-difference reference extf / truncf are exact, so the section still commutes and is
    certified (found by the tutorial capture)"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    body = ("    pid = tl.program_id(0)\n    i = tl.arange(0, N)\n    part = tl.load(x + pid * N + i)\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    acc = part.to(tl.float32) + tl.load(dw + i)\n    tl.store(dw + i, acc)\n"
            "    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    x = np.random.default_rng(6).standard_normal((9, 8)).astype(np.float16)
    ref, ident = _run("audit_lock_extf", body, {"x": ("fp16", x.reshape(-1)), "dw": ("fp32", np.zeros(8)),
                                                "lock": ("int32", [0])}, out_name="dw", grid=(9, 1, 1), full=True)
    assert (ref.element_classes(ident["dw"]) == "complete_composed").all(), sorted(ref.reasons)


def test_cas_lock_certificate_budget_refusal_keeps_the_premise(monkeypatch):
    """full interpreter: a certificate over its deterministic budget is refused (the premise stays), never assumed"""
    pytest.importorskip("triton")
    pytest.importorskip("z3")
    from test_signatures_structural import _run

    from kernel_analyzer.reference_eval import lock_certificate as LC
    monkeypatch.setattr(LC, "MAX_TERMS", 4)
    body = ("    i = tl.arange(0, N)\n    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n    tl.debug_barrier()\n    tl.atomic_xchg(lock, 0)\n")
    ref, ident = _run("audit_lock_budget", body, {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                      grid=(7, 1, 1), full=True)
    assert (ref.element_classes(ident["data"]) == "complete_under_premise").all(), sorted(ref.reasons)
