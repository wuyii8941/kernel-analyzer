"""Independent review tests for the ATTN path A A2 draft (package B: fused SDPA ATen overloads, dropout 0), 2026-10-21.

The draft pair contracts/draft/ATTN_path_a_A2_20261020.yaml + ATTN_adapter_path_a_A2_20261020.py is copied into
/data1/tzh/kernel-analyzer/.cache/spec_a2_review/B (with copies of the reference modules it imports from kernel_spec,
kept for hash binding) and loaded with kernel_spec.registry.load. Every expected value comes from the reviewer's own
attention code below, written from the PyTorch 2.10 SDPA page (scores = scale q.k, upper-left inclusive causal,
repeat_interleave GQA) and FlashAttention-2 Algorithm 2 (P = exp(S - L), D = rowsum(dO o O), dS = P o (dP - D)):
mpmath at 60 digits and Fraction, no kernel_spec.attn_aten_semantics / attn_aten_indep / adapter helper is used for
the computation. Public profiles are compared with the validated top-level adapter on random inputs.
"""
from __future__ import annotations

import hashlib
import random
import shutil
import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import ends, evaluate  # noqa: E402

DRAFT_YAML = ROOT / "contracts" / "draft" / "ATTN_path_a_A2_20261020.yaml"
DRAFT_PY = ROOT / "contracts" / "draft" / "ATTN_adapter_path_a_A2_20261020.py"
MODULES = [ROOT / "kernel_spec" / "attn_aten_semantics.py", ROOT / "kernel_spec" / "attn_aten_indep.py"]
STAGE = ROOT.parent / ".cache" / "spec_a2_review" / "B"
PROFILE = "torch210_aten_sdpa_fused_dropout0"
FLASH = "aten._scaled_dot_product_flash_attention"
FLASH_BWD = "aten._scaled_dot_product_flash_attention_backward"
EFF_BWD = "aten._scaled_dot_product_efficient_attention_backward"

mp = mpmath.mp
mp.dps = 60
TOL = mpmath.mpf(10) ** -40


@pytest.fixture(autouse=True)
def _precision():
    """Other test modules change mpmath's global precision; every test here runs at 60 digits."""
    with mpmath.workdps(60):
        yield


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def contract():
    (STAGE / "contracts").mkdir(parents=True, exist_ok=True)
    (STAGE / "adapters").mkdir(parents=True, exist_ok=True)
    (STAGE / "kernel_spec_copy").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DRAFT_YAML, STAGE / "contracts" / "ATTN.yaml")
    shutil.copyfile(DRAFT_PY, STAGE / "adapters" / "ATTN.py")
    for m in MODULES:
        shutil.copyfile(m, STAGE / "kernel_spec_copy" / m.name)
    return registry.load("ATTN", contracts=STAGE / "contracts", adapters=STAGE / "adapters")


@pytest.fixture(scope="module")
def validated():
    return registry.load("ATTN")


# ------------------------------------------------------------------ reviewer's own arithmetic

def M(x):
    if isinstance(x, mpmath.mpf):
        return x
    x = F(x)
    return mpmath.mpf(x.numerator) / x.denominator


def dot(a, b):
    return sum((M(x) * M(y) for x, y in zip(a, b)), mpmath.mpf(0))


def shape(v):
    s = []
    while isinstance(v, list):
        s.append(len(v))
        v = v[0]
    return s


def zeros(*dims):
    if len(dims) == 1:
        return [mpmath.mpf(0)] * dims[0]
    return [zeros(*dims[1:]) for _ in range(dims[0])]


def kv_of(h, Hq, Hkv, mapping="contiguous"):
    return h // (Hq // Hkv) if mapping == "contiguous" else h % Hkv


def my_forward(q, k, v, causal, scale, mapping="contiguous"):
    """SDPA page formula: out = softmax(scale q k^T + causal bias) v, with the natural logsumexp of each row."""
    B, Hq, L, _ = shape(q)
    Hkv, S, Ev = len(k[0]), len(k[0][0]), len(v[0][0][0])
    out, lse = zeros(B, Hq, L, Ev), zeros(B, Hq, L)
    for b in range(B):
        for h in range(Hq):
            g = kv_of(h, Hq, Hkv, mapping)
            for i in range(L):
                js = [j for j in range(S) if not causal or j <= i]
                s = [M(scale) * dot(q[b][h][i], k[b][g][j]) for j in js]
                lse[b][h][i] = mpmath.log(sum(mpmath.exp(x) for x in s))
                p = [mpmath.exp(x - lse[b][h][i]) for x in s]
                for d in range(Ev):
                    out[b][h][i][d] = sum(pj * M(v[b][g][j][d]) for pj, j in zip(p, js))
    return out, lse


def my_jacobian_vjp(q, k, v, dout, causal, scale):
    """Exact VJP of the forward through the explicit softmax Jacobian ds = (diag(p) - p p^T) dp (FA2 Sec. 2.2)."""
    B, Hq, L, E = shape(q)
    Hkv, S, Ev = len(k[0]), len(k[0][0]), len(v[0][0][0])
    dq, dk, dv = zeros(B, Hq, L, E), zeros(B, Hkv, S, E), zeros(B, Hkv, S, Ev)
    sc = M(scale)
    for b in range(B):
        for h in range(Hq):
            g = kv_of(h, Hq, Hkv)
            for i in range(L):
                js = [j for j in range(S) if not causal or j <= i]
                s = [sc * dot(q[b][h][i], k[b][g][j]) for j in js]
                z = sum(mpmath.exp(x) for x in s)
                p = [mpmath.exp(x) / z for x in s]
                dp = [dot(dout[b][h][i], v[b][g][j]) for j in js]
                n = len(js)
                ds = [sum(((p[a] if a == c else 0) - p[a] * p[c]) * dp[c] for c in range(n)) for a in range(n)]
                for a, j in enumerate(js):
                    for e in range(E):
                        dq[b][h][i][e] += sc * ds[a] * M(k[b][g][j][e])
                        dk[b][g][j][e] += sc * ds[a] * M(q[b][h][i][e])
                    for d in range(Ev):
                        dv[b][g][j][d] += p[a] * M(dout[b][h][i][d])
    return dq, dk, dv


def my_fa2_backward(q, k, v, dout, o, lse, causal, scale):
    """FlashAttention-2 Algorithm 2 with the supplied O and L, scale inserted by the chain rule (S = scale Q K^T)."""
    B, Hq, L, E = shape(q)
    Hkv, S, Ev = len(k[0]), len(k[0][0]), len(v[0][0][0])
    dq, dk, dv = zeros(B, Hq, L, E), zeros(B, Hkv, S, E), zeros(B, Hkv, S, Ev)
    sc = M(scale)
    for b in range(B):
        for h in range(Hq):
            g = kv_of(h, Hq, Hkv)
            for i in range(L):
                D = dot(dout[b][h][i], o[b][h][i])
                for j in range(S):
                    if causal and j > i:
                        continue
                    P = mpmath.exp(sc * dot(q[b][h][i], k[b][g][j]) - M(lse[b][h][i]))
                    dS = P * (dot(dout[b][h][i], v[b][g][j]) - D)
                    for e in range(E):
                        dq[b][h][i][e] += sc * dS * M(k[b][g][j][e])
                        dk[b][g][j][e] += sc * dS * M(q[b][h][i][e])
                    for d in range(Ev):
                        dv[b][g][j][d] += P * M(dout[b][h][i][d])
    return dq, dk, dv


def flat(x):
    if isinstance(x, list):
        for y in x:
            yield from flat(y)
    else:
        yield x


def encloses(ref, mine, *, width=mpmath.mpf(10) ** -20):
    """Every coordinate of `mine` lies in the adapter's enclosure (to 1e-40) and the enclosure is tight."""
    a, b = list(flat(ref)), list(flat(mine))
    if shape(ref) != shape(mine) or len(a) != len(b):
        return False
    for r, m in zip(a, b):
        lo, hi = ends(r)
        lo, hi = M(lo), M(hi)
        if not (lo - TOL * (1 + abs(m)) <= m <= hi + TOL * (1 + abs(m))) or hi - lo > width * (1 + abs(m)):
            return False
    return True


def exact_in(ref, mine):
    for r, m in zip(flat(ref), flat(mine)):
        lo, hi = ends(r)
        if not lo <= m <= hi:
            return False
    return shape(ref) == shape(mine)


# ------------------------------------------------------------------ call builders

def rnd(rng, *dims, lim=24):
    if len(dims) == 1:
        return [rng.randint(-lim, lim) / 8 for _ in range(dims[0])]
    return [rnd(rng, *dims[1:], lim=lim) for _ in range(dims[0])]


def T(v, t):
    return {"t": t, "v": v, "shape": shape(v)}


def to_dyadic(x, bits=10):
    """Round an mpf to a multiple of 2^-bits (a value exactly representable in f32 and, for small values, bf16)."""
    if isinstance(x, list):
        return [to_dyadic(y, bits) for y in x]
    return float(F(int(mpmath.nint(x * 2 ** bits)), 2 ** bits))


SEED2 = {"t": "i64", "v": [0, 0], "shape": [2]}
OFF0 = {"t": "i64", "v": 0, "shape": []}


def flash(q, k, v, causal, scale, dt="bf16", dropout=0.0, extra=()):
    call = {"op": FLASH, "args": [T(q, dt), T(k, dt), T(v, dt), dropout, causal, *extra], "kwargs": {}}
    if scale is not None:
        call["kwargs"]["scale"] = scale
    return call


def flash_bwd(g, q, k, v, o, lse, causal, scale, dt="bf16", dropout=0.0, cum=(None, None), maxes=None):
    L, S = len(q[0][0]), len(k[0][0])
    mq, mk = maxes or (L, S)
    return {"op": FLASH_BWD, "args": [T(g, dt), T(q, dt), T(k, dt), T(v, dt), T(o, dt), T(lse, "f32"), cum[0], cum[1],
                                      mq, mk, dropout, causal, SEED2, OFF0], "kwargs": {"scale": scale}}


def eff_bwd(g, q, k, v, o, lse, causal, scale, dt="f32", dropout=0.0, mask=(True, True, True, False), bias=None,
            pass_causal=None):
    args = [T(g, dt), T(q, dt), T(k, dt), T(v, dt), bias, T(o, dt), T(lse, "f32"), OFF0, OFF0, dropout, list(mask)]
    if causal or pass_causal:
        args.append(causal)
    return {"op": EFF_BWD, "args": args, "kwargs": {"scale": scale}}


def run(contract, call, fn=None):
    return evaluate(fn or contract.adapter.reference, PROFILE, call)


# ------------------------------------------------------------------ staging

def test_stage_is_the_reviewed_files(contract):
    assert sha(STAGE / "contracts" / "ATTN.yaml") == sha(DRAFT_YAML)
    assert sha(STAGE / "adapters" / "ATTN.py") == sha(DRAFT_PY)
    assert Path(contract.adapter.__file__).resolve().is_relative_to(STAGE.resolve())
    import kernel_spec.attn_aten_indep as AI
    import kernel_spec.attn_aten_semantics as AS
    assert sha(AS.__file__) == sha(STAGE / "kernel_spec_copy" / "attn_aten_semantics.py")
    assert sha(AI.__file__) == sha(STAGE / "kernel_spec_copy" / "attn_aten_indep.py")
    prof = contract.data["profiles"][PROFILE]
    assert prof["task_choice_required"] is False
    assert "make_reference" in Path(contract.adapter.__file__).read_text()
    assert {FLASH, FLASH_BWD, EFF_BWD} <= set(contract.data["covers"])


# ------------------------------------------------------------------ AT.ATEN_SAVED: algebra (no adapter)

def test_fa2_equals_exact_vjp_when_supplied_values_are_exact():
    """With O and L the exact forward values, FA2 Algorithm 2 equals the Jacobian VJP coordinate by coordinate."""
    rng = random.Random(11)
    for causal in (True, False):
        q, k, v, g = rnd(rng, 1, 4, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 4, 5, 3)
        o, lse = my_forward(q, k, v, causal, F(3, 4))
        fa2 = my_fa2_backward(q, k, v, g, o, lse, causal, F(3, 4))
        jac = my_jacobian_vjp(q, k, v, g, causal, F(3, 4))
        for a, b in zip(fa2, jac):
            assert all(abs(x - y) < mpmath.mpf(10) ** -50 for x, y in zip(flat(a), flat(b)))


def test_fa2_identity_symbolic():
    sympy = pytest.importorskip("sympy")
    s = sympy.symbols("s0:3", real=True)
    dp = sympy.symbols("d0:3", real=True)
    Z = sum(sympy.exp(x) for x in s)
    p = [sympy.exp(x) / Z for x in s]
    Lse = sympy.log(Z)
    D = sum(pi * di for pi, di in zip(p, dp))           # D = dO.O = sum_j P_j dO.v_j
    for a in range(3):
        fa2 = sympy.exp(s[a] - Lse) * (dp[a] - D)
        jac = sum(((p[a] if a == c else 0) - p[a] * p[c]) * dp[c] for c in range(3))
        assert sympy.simplify(fa2 - jac) == 0


# ------------------------------------------------------------------ forward

@pytest.mark.parametrize("causal,scale", [(True, None), (True, F(1, 4)), (False, F(3, 4)), (False, None)])
def test_flash_forward_small_gqa(contract, causal, scale):
    rng = random.Random(1)
    q, k, v = rnd(rng, 1, 4, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 2, 5, 3)
    r = run(contract, flash(q, k, v, causal, None if scale is None else str(scale)))
    assert r.outcome == "value", r.note
    out = r.fields["out"]
    assert len(out) == 9 and [out[i] for i in (2, 3, 6, 7, 8)] == [None] * 5 and out[4] == 5 and out[5] == 5
    sc = scale if scale is not None else 1 / mpmath.sqrt(3)     # default 1/sqrt(E), E = 3
    o, lse = _forward_real_scale(q, k, v, causal, sc)
    assert encloses(out[0], o) and encloses(out[1], lse)
    # the contiguous mapping is discriminated: round-robin gives a different value
    o_rr, _ = _forward_real_scale(q, k, v, causal, sc, mapping="rr")
    assert not encloses(out[0], o_rr)


def _forward_real_scale(q, k, v, causal, sc, mapping="contiguous"):
    B, Hq, L, _ = shape(q)
    Hkv, S, Ev = len(k[0]), len(k[0][0]), len(v[0][0][0])
    sc = sc if isinstance(sc, mpmath.mpf) else M(sc)
    out, lse = zeros(B, Hq, L, Ev), zeros(B, Hq, L)
    for h in range(Hq):
        g = kv_of(h, Hq, Hkv, mapping)
        for i in range(L):
            js = [j for j in range(S) if not causal or j <= i]
            s = [sc * dot(q[0][h][i], k[0][g][j]) for j in js]
            lse[0][h][i] = mpmath.log(sum(mpmath.exp(x) for x in s))
            for d in range(Ev):
                out[0][h][i][d] = sum(mpmath.exp(x - lse[0][h][i]) * M(v[0][g][j][d]) for x, j in zip(s, js))
    return out, lse


def test_flash_forward_noncausal_nonsquare(contract):
    rng = random.Random(2)
    q, k, v = rnd(rng, 1, 4, 3, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 2, 5, 3)
    r = run(contract, flash(q, k, v, False, "1/2"))
    assert r.outcome == "value"
    o, lse = my_forward(q, k, v, False, F(1, 2))
    assert encloses(r.fields["out"][0], o) and encloses(r.fields["out"][1], lse)
    assert r.fields["out"][4] == 3 and r.fields["out"][5] == 5


# ------------------------------------------------------------------ backward

@pytest.mark.parametrize("causal", [True, False])
def test_flash_backward_supplied_values(contract, causal):
    rng = random.Random(3)
    q, k, v, g = rnd(rng, 1, 4, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 4, 5, 3)
    o_exact, lse_exact = my_forward(q, k, v, causal, F(1, 2))
    o = to_dyadic(o_exact, 6)                      # what a low-precision forward might return
    lse = to_dyadic(lse_exact, 10)
    r = run(contract, flash_bwd(g, q, k, v, o, lse, causal, "1/2"))
    assert r.outcome == "value", r.note
    want = my_fa2_backward(q, k, v, g, o, lse, causal, F(1, 2))
    assert len(r.fields["out"]) == 3
    for got, w in zip(r.fields["out"], want):
        assert encloses(got, w)
    # the diagnostic reading is the exact VJP of the forward at (q, k, v)
    rr = run(contract, flash_bwd(g, q, k, v, o, lse, causal, "1/2"), contract.adapter.READINGS["recompute_saved"])
    for got, w in zip(rr.fields["out"], my_jacobian_vjp(q, k, v, g, causal, F(1, 2))):
        assert encloses(got, w)
    # and the profile is not the recomputed VJP here (the supplied values differ from the exact ones)
    assert not all(encloses(a, b) for a, b in zip(r.fields["out"], my_jacobian_vjp(q, k, v, g, causal, F(1, 2))))


def test_lse_shift_scales_every_gradient(contract):
    """P = exp(S - L): adding 1/2 to the supplied L multiplies dq, dk and dv by exp(-1/2) (O unchanged)."""
    rng = random.Random(4)
    q, k, v, g = rnd(rng, 1, 4, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 2, 5, 3), rnd(rng, 1, 4, 5, 3)
    o_exact, lse_exact = my_forward(q, k, v, True, F(1, 2))
    o, lse = to_dyadic(o_exact, 6), to_dyadic(lse_exact, 10)
    lse2 = [[[x + 0.5 for x in row] for row in hh] for hh in lse]
    base = my_fa2_backward(q, k, v, g, o, lse, True, F(1, 2))
    r = run(contract, flash_bwd(g, q, k, v, o, lse2, True, "1/2"))
    f = mpmath.exp(mpmath.mpf(-1) / 2)
    for got, b in zip(r.fields["out"], base):
        assert encloses(got, [[[[f * x for x in row] for row in hh] for hh in bb] for bb in b])


def test_exact_fraction_backward_q_zero_and_k_zero(contract):
    """Scores 0 and supplied L = 0 give P = 1 exactly: all gradients are rationals computed with Fraction."""
    rng = random.Random(5)
    for which in ("q", "k"):
        Hq, Hkv, L, E = 4, 2, 4, 2
        q = [[[[0.0] * E for _ in range(L)] for _ in range(Hq)]] if which == "q" else rnd(rng, 1, Hq, L, E)
        k = [[[[0.0] * E for _ in range(L)] for _ in range(Hkv)]] if which == "k" else rnd(rng, 1, Hkv, L, E)
        v, g, o = rnd(rng, 1, Hkv, L, E), rnd(rng, 1, Hq, L, E), rnd(rng, 1, Hq, L, E)
        lse = [[[0.0] * L for _ in range(Hq)]]
        sc = F(1, 2)
        dq = [[[[F(0)] * E for _ in range(L)] for _ in range(Hq)]]
        dk = [[[[F(0)] * E for _ in range(L)] for _ in range(Hkv)]]
        dv = [[[[F(0)] * E for _ in range(L)] for _ in range(Hkv)]]
        for h in range(Hq):
            gk = h // (Hq // Hkv)
            for i in range(L):
                D = sum(F(a) * F(b) for a, b in zip(g[0][h][i], o[0][h][i]))
                for j in range(i + 1):
                    dS = sum(F(a) * F(b) for a, b in zip(g[0][h][i], v[0][gk][j])) - D
                    for e in range(E):
                        dq[0][h][i][e] += sc * dS * F(k[0][gk][j][e])
                        dk[0][gk][j][e] += sc * dS * F(q[0][h][i][e])
                        dv[0][gk][j][e] += F(g[0][h][i][e])
        r = run(contract, flash_bwd(g, q, k, v, o, lse, True, "1/2"))
        assert r.outcome == "value", r.note
        for got, w in zip(r.fields["out"], (dq, dk, dv)):
            assert exact_in(got, w)
        # complete-output check accepts the exact values and rejects one wrong coordinate
        obs = [T([[[[float(x) for x in row] for row in hh] for hh in t[0]]], "bf16") for t in (dq, dk, dv)]
        assert contract.adapter.check_output(PROFILE, flash_bwd(g, q, k, v, o, lse, True, "1/2"), {"out": obs})
        bad = [dict(x) for x in obs]
        vv = [[[[float(y) for y in row] for row in hh] for hh in dv[0]]]
        vv[0][1][0][0] += 1.0
        bad[2] = T(vv, "bf16")
        assert not contract.adapter.check_output(PROFILE, flash_bwd(g, q, k, v, o, lse, True, "1/2"), {"out": bad})
        wrong_dtype = list(obs)
        wrong_dtype[0] = dict(obs[0], t="f32")
        assert not contract.adapter.check_output(PROFILE, flash_bwd(g, q, k, v, o, lse, True, "1/2"),
                                                 {"out": wrong_dtype})
        assert not contract.adapter.check_output(PROFILE, flash_bwd(g, q, k, v, o, lse, True, "1/2"),
                                                 {"out": obs[:2]})


@pytest.mark.parametrize("width", [5, 32])
def test_efficient_backward_padded_and_unpadded_lse(contract, width):
    rng = random.Random(6)
    q, k, v, g = (rnd(rng, 1, 2, 5, 3) for _ in range(4))
    o_exact, lse_exact = my_forward(q, k, v, True, F(1, 8))
    o, lse = to_dyadic(o_exact, 12), to_dyadic(lse_exact, 12)
    padded = [[row + [float("inf")] * (width - 5) for row in hh] for hh in lse]
    r = run(contract, eff_bwd(g, q, k, v, o, padded, True, "1/8"))
    assert r.outcome == "value", r.note
    assert len(r.fields["out"]) == 4 and r.fields["out"][3] is None
    for got, w in zip(r.fields["out"][:3], my_fa2_backward(q, k, v, g, o, lse, True, F(1, 8))):
        assert encloses(got, w)


# ------------------------------------------------------------------ recorded argument structures, reduced

def test_recorded_qwen_flash_forward_and_backward_reduced(contract):
    rng = random.Random(7)
    L, E = 4, 2
    q, k, v = rnd(rng, 1, 14, L, E), rnd(rng, 1, 2, L, E), rnd(rng, 1, 2, L, E)
    r = run(contract, flash(q, k, v, True, 0.125))
    assert r.outcome == "value", r.note
    o_exact, lse_exact = my_forward(q, k, v, True, F(1, 8))
    assert encloses(r.fields["out"][0], o_exact) and encloses(r.fields["out"][1], lse_exact)
    g = rnd(rng, 1, 14, L, E)
    o, lse = to_dyadic(o_exact, 6), to_dyadic(lse_exact, 14)
    rb = run(contract, flash_bwd(g, q, k, v, o, lse, True, 0.125))
    assert rb.outcome == "value", rb.note
    want = my_fa2_backward(q, k, v, g, o, lse, True, F(1, 8))
    assert [shape(x) for x in rb.fields["out"]] == [[1, 14, L, E], [1, 2, L, E], [1, 2, L, E]]
    for got, w in zip(rb.fields["out"], want):
        assert encloses(got, w)


@pytest.mark.parametrize("model,H,causal", [("bert", 2, False), ("gpt2", 3, True)])
def test_recorded_efficient_backward_reduced(contract, model, H, causal):
    rng = random.Random(8)
    L, E = 4, 2
    q, k, v, g = (rnd(rng, 1, H, L, E) for _ in range(4))
    o_exact, lse_exact = my_forward(q, k, v, causal, F(1, 8))
    o = to_dyadic(o_exact, 12)
    lse = [[row + [float("inf")] * 28 for row in hh] for hh in to_dyadic(lse_exact, 12)]
    recorded = eff_bwd(g, q, k, v, o, lse, causal, 0.125, dropout=0.1)
    r = run(contract, recorded)
    assert r.outcome == "unsupported" and "AT.ATEN_RNG" in r.note
    r0 = run(contract, eff_bwd(g, q, k, v, o, lse, causal, 0.125, dropout=0.0))
    assert r0.outcome == "value", r0.note
    for got, w in zip(r0.fields["out"][:3], my_fa2_backward(q, k, v, g, o, to_dyadic(lse_exact, 12), causal, F(1, 8))):
        assert encloses(got, w)


# ------------------------------------------------------------------ refusal classes

def _base(rng):
    q, k, v = rnd(rng, 1, 4, 3, 2), rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 2, 3, 2)
    g, o = rnd(rng, 1, 4, 3, 2), rnd(rng, 1, 4, 3, 2)
    lse = rnd(rng, 1, 4, 3)
    return q, k, v, g, o, lse


def _refusals():
    rng = random.Random(9)
    q, k, v, g, o, lse = _base(rng)
    q2, k2, v2 = rnd(rng, 1, 2, 2, 2), rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 2, 3, 2)
    qe, ke, ve, ge, oe = (rnd(rng, 1, 2, 3, 2) for _ in range(5))
    lsee = [[row + [float("inf")] * 29 for row in rnd(rng, 1, 2, 3)[0]]]
    bad_val = rnd(rng, 1, 4, 3, 2)
    bad_val[0][0][0][0] = 0.1
    inf_q = rnd(rng, 1, 4, 3, 2)
    inf_q[0][0][0][0] = float("inf")
    nan_pad = [[row[:3] + [float("nan")] + row[4:] for row in lsee[0]]]
    zero_pad = [[row[:3] + [0.0] + row[4:] for row in lsee[0]]]
    return [
        ("flash dropout", flash(q, k, v, True, 0.5, dropout=0.1), "unsupported"),
        ("flash bwd dropout", flash_bwd(g, q, k, v, o, lse, True, 0.5, dropout=0.1), "unsupported"),
        ("eff bwd dropout", eff_bwd(ge, qe, ke, ve, oe, lsee, False, 0.5, dropout=0.1), "unsupported"),
        ("dropout above 1", flash(q, k, v, True, 0.5, dropout=1.5), "invalid_input"),
        ("flash causal L!=S", flash(q2, k2, v2, True, 0.5), "unsupported"),
        ("eff causal L!=S", eff_bwd(rnd(rng, 1, 2, 2, 2), q2, k2, v2, rnd(rng, 1, 2, 2, 2),
                                    [[r + [float("inf")] * 30 for r in rnd(rng, 1, 2, 2)[0]]], True, 0.5),
         "unsupported"),
        ("flash f32", flash(q, k, v, False, 0.5, dt="f32"), "unsupported"),
        ("flash f64", flash(q, k, v, False, 0.5, dt="f64"), "unsupported"),
        ("flash Hq not divisible", flash(rnd(rng, 1, 3, 3, 2), k, v, False, 0.5), "invalid_input"),
        ("eff unequal heads", eff_bwd(g, q, k, v, o, [[r + [float("inf")] * 29 for r in lse[0]]], False, 0.5),
         "unsupported"),
        ("debug mask", flash(q, k, v, False, 0.5, extra=(True,)), "unsupported"),
        ("cum_seq given", flash_bwd(g, q, k, v, o, lse, True, 0.5, cum=({"t": "i32", "v": [0, 3]}, None)),
         "unsupported"),
        ("max_q != L", flash_bwd(g, q, k, v, o, lse, True, 0.5, maxes=(4, 3)), "unsupported"),
        ("attn_bias", eff_bwd(ge, qe, ke, ve, oe, lsee, False, 0.5, bias=T(rnd(rng, 1, 2, 3, 3), "f32")),
         "unsupported"),
        ("bias grad", eff_bwd(ge, qe, ke, ve, oe, lsee, False, 0.5, mask=(True, True, True, True)), "unsupported"),
        ("unrequested dq", eff_bwd(ge, qe, ke, ve, oe, lsee, False, 0.5, mask=(False, True, True, False)),
         "unsupported"),
        ("mask not 4", eff_bwd(ge, qe, ke, ve, oe, lsee, False, 0.5, mask=(True, True, True)), "invalid_input"),
        ("lse NaN padding", eff_bwd(ge, qe, ke, ve, oe, nan_pad, False, 0.5), "unsupported"),
        ("lse 0 padding", eff_bwd(ge, qe, ke, ve, oe, zero_pad, False, 0.5), "unsupported"),
        ("eff lse width 16", eff_bwd(ge, qe, ke, ve, oe, [[r[:16] for r in lsee[0]]], False, 0.5), "unsupported"),
        ("flash bwd padded lse", flash_bwd(g, q, k, v, o, [[r + [float("inf")] * 29 for r in lse[0]]], True, 0.5),
         "unsupported"),
        ("lse bf16", dict(flash_bwd(g, q, k, v, o, lse, True, 0.5)), "unsupported"),
        ("inf in q", flash(inf_q, k, v, False, 0.5), "unsupported"),
        ("value not bf16", flash(bad_val, k, v, False, 0.5), "invalid_input"),
        ("grad_out shape", flash_bwd(rnd(rng, 1, 4, 2, 2), q, k, v, o, lse, True, 0.5), "invalid_input"),
        ("rank 3", {"op": FLASH, "args": [T(q[0], "bf16"), T(k[0], "bf16"), T(v[0], "bf16"), 0.0, False],
                    "kwargs": {}}, "invalid_input"),
        ("dtype mix", {"op": FLASH, "args": [T(q, "bf16"), T(k, "f16"), T(v, "bf16"), 0.0, False], "kwargs": {}},
         "unsupported"),
        ("head dim differs", flash(q, rnd(rng, 1, 2, 3, 3), v, False, 0.5), "invalid_input"),
    ]


@pytest.mark.parametrize("name,call,outcome", _refusals(), ids=[r[0] for r in _refusals()])
def test_refusal_classes(contract, name, call, outcome):
    if name == "lse bf16":
        call["args"][5] = dict(call["args"][5], t="bf16")
    r = run(contract, call)
    assert r.outcome == outcome, (name, r.outcome, r.note)
    if name.endswith("dropout"):
        assert "AT.ATEN_RNG" in r.note


def test_flash_value_head_dim_must_equal_query_head_dim(contract):
    """Review issue 1 (fixed): TS_META_SDPA.FLASH_FWD attention = empty_like(query); PyTorch CUDA refuses Ev != E."""
    rng = random.Random(10)
    q, k, v = rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 2, 3, 1)
    assert run(contract, flash(q, k, v, False, 0.5)).outcome == "invalid_input"
    g, o, lse = rnd(rng, 1, 2, 3, 1), rnd(rng, 1, 2, 3, 1), rnd(rng, 1, 2, 3)
    assert run(contract, flash_bwd(g, q, k, v, o, lse, True, 0.5)).outcome == "invalid_input"
    # the misreading mutant accepts it (so the refusal is witnessed)
    assert run(contract, flash(q, k, v, False, 0.5), contract.adapter.MUTANTS["AT.M19"]).outcome == "value"


@pytest.mark.parametrize("width", [3, 32])
def test_efficient_backward_value_head_dim_differs(contract, width):
    """The efficient backward keeps Ev != E (PyTorch's efficient kernel accepts it): checked against FA2."""
    rng = random.Random(14)
    q, k, g_unused = rnd(rng, 1, 2, 3, 3), rnd(rng, 1, 2, 4, 3), None
    v = rnd(rng, 1, 2, 4, 2)
    g = rnd(rng, 1, 2, 3, 2)
    o_exact, lse_exact = my_forward(q, k, v, False, F(1, 4))
    o, lse = to_dyadic(o_exact, 12), to_dyadic(lse_exact, 12)
    padded = [[row + [float("inf")] * (width - 3) for row in hh] for hh in lse]
    r = run(contract, eff_bwd(g, q, k, v, o, padded, False, "1/4"))
    assert r.outcome == "value", r.note
    assert [shape(x) for x in r.fields["out"][:3]] == [[1, 2, 3, 3], [1, 2, 4, 3], [1, 2, 4, 2]]
    for got, w in zip(r.fields["out"][:3], my_fa2_backward(q, k, v, g, o, lse, False, F(1, 4))):
        assert encloses(got, w)


# ------------------------------------------------------------------ public profiles unchanged

def _public_inputs():
    rng = random.Random(12)
    out = []
    for n in range(40):
        rank = rng.choice((3, 4))
        B = rng.choice((1, 2))
        hkv = rng.choice((1, 2))
        gqa = rng.random() < 0.5
        hq = hkv * rng.choice((1, 2)) if gqa else rng.choice((hkv, hkv)) if hkv > 1 else rng.choice((1, 2))
        L, S, E, Ev = rng.randint(1, 3), rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 2)
        lead = (B,) if rank == 4 else ()
        mk = lambda h, n_, e: rnd(rng, *lead, h, n_, e)  # noqa: E731
        inp = {"q": mk(hq, L, E), "k": mk(hkv, S, E), "v": mk(hkv, S, Ev), "dtype": rng.choice(("f32", "bf16", "f64")),
               "attn_mask": None, "attn_mask_dtype": None, "is_causal": False,
               "scale": rng.choice((None, "1/2", 1)), "enable_gqa": gqa, "dropout_p": 0}
        kind = rng.choice(("none", "causal", "bool", "add"))
        if kind == "causal":
            inp["is_causal"] = True
        elif kind == "bool":
            inp["attn_mask"] = [[rng.random() < 0.7 for _ in range(S)] for _ in range(L)]
            inp["attn_mask_dtype"] = "b8"
        elif kind == "add":
            inp["attn_mask"] = [[rng.choice((0, 1, -1, "-inf")) for _ in range(S)] for _ in range(L)]
            inp["attn_mask_dtype"] = inp["dtype"]
        if n % 2:
            hq_eff = len(inp["q"][0]) if rank == 4 else len(inp["q"])
            inp["upstream"] = mk(hq_eff, L, Ev)
            out.append(("torch210_sdpa_defined_real_vjp", inp))
        else:
            out.append(("torch210_sdpa_defined_real", inp))
    out.append(("torch210_sdpa_defined_real", dict(out[0][1], dropout_p="1/10")))
    out.append(("torch210_sdpa_defined_real", dict(out[0][1], window=[1, 1])))
    return out


def test_public_profiles_agree_with_validated(contract, validated):
    outcomes = set()
    for profile, inp in _public_inputs():
        a = evaluate(contract.adapter.reference, profile, inp)
        b = evaluate(validated.adapter.reference, profile, inp)
        assert a.outcome == b.outcome, (profile, inp, a.note, b.note)
        outcomes.add(a.outcome)
        if a.outcome == "value":
            assert set(a.fields) == set(b.fields)
            for name in a.fields:
                assert [ends(x) for x in flat(a.fields[name])] == [ends(x) for x in flat(b.fields[name])]
    assert "value" in outcomes and "unsupported" in outcomes


def test_public_vjp_matches_reviewer_jacobian_gqa(contract):
    rng = random.Random(13)
    q, k, v, g = rnd(rng, 1, 4, 3, 2), rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 2, 3, 2), rnd(rng, 1, 4, 3, 2)
    inp = {"q": q, "k": k, "v": v, "dtype": "f32", "attn_mask": None, "attn_mask_dtype": None, "is_causal": True,
           "scale": "1/2", "enable_gqa": True, "dropout_p": 0, "upstream": g}
    r = evaluate(contract.adapter.reference, "torch210_sdpa_defined_real_vjp", inp)
    assert r.outcome == "value"
    for name, w in zip(("dq", "dk", "dv"), my_jacobian_vjp(q, k, v, g, True, F(1, 2))):
        assert encloses(r.fields[name], w)


def test_validated_files_untouched():
    assert sha(ROOT / "contracts" / "ATTN.yaml") == sha(ROOT / "validated" / "contracts" / "ATTN.yaml")
    assert sha(ROOT / "adapters" / "ATTN.py") == sha(ROOT / "validated" / "adapters" / "ATTN.py")
