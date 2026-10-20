"""Independent review of path A closure package A (NORM aten_native, EMB aten_dense_backward, NLL aten_real).

Reviewer tests, 2026-10-20. The draft pairs contracts/draft/<ID>_path_a_A2_20261020.yaml / <ID>_adapter_path_a_A2_20261020.py
are staged under .cache/spec_a2_review/A/{contracts,adapters} (and copied to tmp_path for isolation) and loaded with
kernel_spec.registry.load. Expected values are computed here independently, from the documented formulas, with plain
Python Fractions and an integer-sqrt enclosure of 1/sqrt(q); nothing from kernel_spec.norm_emb_nll_aten or the adapters is
used for the expected side. kernel_spec.num.ends is used only to read the reference's interval endpoints.

Comparison: exact equality where both sides are rationals; otherwise the reference enclosure must intersect the
reviewer's enclosure and be narrow (width < 2^-100 relative), so a vacuous enclosure cannot pass.
"""
import random
import shutil
import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1]
REPO = SPEC.parent
sys.path[:0] = [str(SPEC)]
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import ends, evaluate  # noqa: E402

DRAFT = SPEC / "contracts" / "draft"
REVIEW_STAGE = REPO / ".cache" / "spec_a2_review" / "A"
VALIDATED = SPEC / "validated"
IDS = ("NORM", "EMB", "NLL")
F = Fraction


# ------------------------------------------------------------------ staging / loading
@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root = tmp_path_factory.mktemp("a2_review_A")
    for sub in ("contracts", "adapters"):
        (root / sub).mkdir()
    for cid in IDS:
        shutil.copy(DRAFT / f"{cid}_path_a_A2_20261020.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_path_a_A2_20261020.py", root / "adapters" / f"{cid}.py")
    return root


@pytest.fixture(scope="module")
def C(stage):
    return {cid: registry.load(cid, contracts=stage / "contracts", adapters=stage / "adapters") for cid in IDS}


@pytest.fixture(scope="module")
def V(stage):
    return {cid: registry.load(cid, contracts=VALIDATED / "contracts", adapters=VALIDATED / "adapters")
            for cid in ("NORM", "EMB")}


def test_review_stage_matches_drafts():
    """The persistent review stage (used for check_contracts) holds byte-identical copies of the drafts."""
    for cid in IDS:
        for src, dst in ((DRAFT / f"{cid}_path_a_A2_20261020.yaml", REVIEW_STAGE / "contracts" / f"{cid}.yaml"),
                         (DRAFT / f"{cid}_adapter_path_a_A2_20261020.py", REVIEW_STAGE / "adapters" / f"{cid}.py")):
            if dst.exists():
                assert dst.read_bytes() == src.read_bytes(), dst


def run(c, profile, call):
    return evaluate(c.adapter.reference, profile, call)


# ------------------------------------------------------------------ independent numerics
P = 400  # bits for the integer-sqrt enclosure


def rsqrt_enc(q):
    """Enclosure [lo, hi] of 1/sqrt(q) for rational q > 0, from integer square roots."""
    q = F(q)
    assert q > 0
    a = q.denominator * 4 ** P
    b = q.numerator
    s = isqrt(a // b)          # s^2 <= floor(a/b) <= a/b < floor(a/b)+1 <= (s+1)^2
    return F(s, 2 ** P), F(s + 1, 2 ** P)


def is_square_rational(q):
    q = F(q)
    n, d = q.numerator, q.denominator
    return n >= 0 and isqrt(n) ** 2 == n and isqrt(d) ** 2 == d


def rsqrt(q):
    """Exact 1/sqrt(q) when rational, else an enclosure tuple."""
    q = F(q)
    if is_square_rational(q):
        return F(isqrt(q.denominator), isqrt(q.numerator))
    return rsqrt_enc(q)


def iv_affine(r, a, b=F(0)):
    """a * r + b with r exact or an enclosure."""
    if isinstance(r, F):
        return a * r + b
    lo, hi = a * r[0], a * r[1]
    return (min(lo, hi) + b, max(lo, hi) + b)


def ref_ends(v):
    if isinstance(v, F):
        return v, v
    if isinstance(v, int):
        return F(v), F(v)
    lo, hi = ends(v)
    return F(lo), F(hi)


def agree(ref, mine, where=""):
    """Coordinate-wise exact equality / narrow enclosure intersection over nested lists."""
    if mine is None:
        assert ref is None, where
        return
    if isinstance(mine, list):
        assert isinstance(ref, list) and len(ref) == len(mine), (where, ref, mine)
        for i, (r, m) in enumerate(zip(ref, mine)):
            agree(r, m, f"{where}[{i}]")
        return
    lo, hi = ref_ends(ref)
    if isinstance(mine, F):
        if lo == hi:
            assert lo == mine, (where, lo, mine)
        else:
            assert lo <= mine <= hi, (where, lo, hi, mine)
    else:
        assert max(lo, mine[0]) <= min(hi, mine[1]), (where, (lo, hi), mine)
    scale = max(abs(lo), abs(hi), F(1))
    assert hi - lo <= scale / 2 ** 100, (where, "reference enclosure too wide", hi - lo)


def nest(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0] if shape[0] else 0
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def tens(dtype, flat, shape):
    """Recorded tensor dict; flat values are Fractions exactly representable as Python floats in the dtype."""
    return {"t": dtype, "v": nest([float(v) for v in flat], list(shape)), "shape": list(shape)}


def itens(flat, shape, dtype="i64"):
    return {"t": dtype, "v": nest([int(v) for v in flat], list(shape)), "shape": list(shape)}


def dyadic(rng, n, k=96, den=16):
    """Values k'/den with |k'| <= k: exactly representable in bf16 (<= 8 significant bits), f16, f32, f64."""
    return [F(rng.randint(-k, k), den) for _ in range(n)]


def shape_of(x):
    s = []
    while isinstance(x, list):
        s.append(len(x))
        x = x[0] if x else None
    return tuple(s)


# ------------------------------------------------------------------ NORM: reviewer's own formulas
def my_layer_norm(x, M, N, w, b, eps):
    out, mean, rstd = [], [], []
    for m in range(M):
        row = x[m * N:(m + 1) * N]
        mu = sum(row, F(0)) / N
        var = sum(((v - mu) ** 2 for v in row), F(0)) / N           # biased (T_LAYERNORM.VAR)
        r = rsqrt(var + eps)                                          # eps inside the root (T_LAYERNORM.DEF)
        for j, v in enumerate(row):
            wj = w[j] if w else F(1)
            bj = b[j] if b else F(0)
            out.append(iv_affine(r, (v - mu) * wj, bj))
        mean.append(mu)
        rstd.append(r)
    return out, mean, rstd


def my_ln_backward(go, x, M, N, mean, rstd, w):
    """Exact VJP of y = (x - mu) r w + b at supplied (mu, r), with mu, r treated as functions of x through the
    standard layer-norm formulas evaluated at the supplied values (decomposition form)."""
    dx, dw, db = [], [F(0)] * N, [F(0)] * N
    for m in range(M):
        mu, r = mean[m], rstd[m]
        xh = [(x[m * N + j] - mu) * r for j in range(N)]
        h = [go[m * N + j] * (w[j] if w else 1) for j in range(N)]
        a = sum(h, F(0)) / N
        c = sum((h[j] * xh[j] for j in range(N)), F(0)) / N
        dx += [r * (h[j] - a - xh[j] * c) for j in range(N)]
        for j in range(N):
            dw[j] += go[m * N + j] * xh[j]
            db[j] += go[m * N + j]
    return dx, dw, db


def my_rms(x, M, N, w, eps):
    out = []
    for m in range(M):
        row = x[m * N:(m + 1) * N]
        r = rsqrt(sum((v * v for v in row), F(0)) / N + eps)
        out += [iv_affine(r, v * (w[j] if w else 1)) for j, v in enumerate(row)]
    return out


@pytest.mark.parametrize("dtype,shape,eps", [("f32", (1, 6, 16), 1e-12), ("bf16", (1, 6, 16), 1e-12),
                                             ("f32", (1, 4, 24), 1e-05)])
def test_native_layer_norm_recorded_structure(C, dtype, shape, eps):
    rng = random.Random(hash((dtype, shape)) & 0xffff)
    M, N = shape[0] * shape[1], shape[2]
    x, w, b = dyadic(rng, M * N), dyadic(rng, N), dyadic(rng, N)
    x[0:N] = [F(3, 2)] * N                     # a constant row: variance 0, radicand = eps (exactly the float eps)
    call = {"op": "aten.native_layer_norm", "args": [tens(dtype, x, shape), [N], tens(dtype, w, (N,)),
                                                     tens(dtype, b, (N,)), eps]}
    r = run(C["NORM"], "aten_native", call)
    assert r.outcome == "value", r.note
    out, mean, rstd = my_layer_norm(x, M, N, w, b, F(eps))
    o = r.fields["out"]
    assert len(o) == 3
    agree(o[0], nest(out, list(shape)), "out")
    # canonical statistics shape fixed by the reference: outer dims followed by 1 (NM.ATEN_STATS)
    assert shape_of(o[1]) == shape[:2] + (1,) and shape_of(o[2]) == shape[:2] + (1,)
    agree(o[1], nest(mean, [shape[0], shape[1], 1]), "mean")
    agree(o[2], nest(rstd, [shape[0], shape[1], 1]), "rstd")


def test_native_layer_norm_backward_recorded_structure_flat_bf16_stats(C):
    """bf16, flat [M] statistics (recorded FlagGems structure), statistics NOT those of the input."""
    rng = random.Random(7)
    M, N = 6, 16
    x, go, w, b = dyadic(rng, M * N), dyadic(rng, M * N), dyadic(rng, N), dyadic(rng, N)
    mean = dyadic(rng, M)
    rstd = [F(rng.randint(1, 64), 8) for _ in range(M)]
    call = {"op": "aten.native_layer_norm_backward",
            "args": [tens("bf16", go, (1, M, N)), tens("bf16", x, (1, M, N)), [N], tens("bf16", mean, (M,)),
                     tens("bf16", rstd, (M,)), tens("bf16", w, (N,)), tens("bf16", b, (N,)), [True, True, True]]}
    r = run(C["NORM"], "aten_native", call)
    assert r.outcome == "value", r.note
    dx, dw, db = my_ln_backward(go, x, M, N, mean, rstd, w)
    agree(r.fields["out"], [nest(dx, [1, M, N]), dw, db])
    # every value is rational here: the reference must be exact, not an enclosure
    for v in r.fields["out"][1] + r.fields["out"][2]:
        assert isinstance(v, F)
    # the same statistics in canonical shape [1, M, 1] give the same result
    canon = dict(call, args=call["args"][:3] + [tens("bf16", mean, (1, M, 1)), tens("bf16", rstd, (1, M, 1))]
                 + call["args"][5:])
    assert run(C["NORM"], "aten_native", canon).fields == r.fields
    # output_mask selects positions; null where false
    for mask in ([True, False, False], [False, True, False], [False, False, True], [False, False, False]):
        rm = run(C["NORM"], "aten_native", dict(call, args=call["args"][:7] + [mask]))
        assert rm.outcome == "value"
        exp = [nest(dx, [1, M, N]), dw, db]
        agree(rm.fields["out"], [e if k else None for e, k in zip(exp, mask)])


def test_native_layer_norm_backward_supplied_vs_recomputed(C):
    """With the exact forward statistics the backward equals the VJP of the forward (finite-difference-free check
    by exact derivative of y w.r.t. x at rational rstd)."""
    # rows chosen so that var + eps is a dyadic square: [0, 2] var 1, eps 3 -> rstd 1/2; [1, 5] var 4, eps 12 -> 1/4
    x = [F(0), F(2), F(1), F(5)]
    eps = 0  # supply eps through the statistics only
    mean, rstd = [F(1), F(3)], [F(1, 2), F(1, 4)]
    go = [F(1), F(-2), F(3), F(1, 2)]
    call = {"op": "aten.native_layer_norm_backward",
            "args": [tens("f32", go, (2, 2)), tens("f32", x, (2, 2)), [2], tens("f32", mean, (2, 1)),
                     tens("f32", rstd, (2, 1)), None, None, [True, False, False]]}
    r = run(C["NORM"], "aten_native", call)
    # independent: N = 2, x_hat = +-1 r d; for N = 2 the layer-norm Jacobian at a row (a, b), var + e = s:
    # y0 = (a - b) / 2 * s^-1/2, with s = (a - b)^2 / 4 + e ; dy0/da = s^-1/2 / 2 - (a-b)^2/8 s^-3/2 = e s^-3/2 / 2
    # here the statistics embed e = s - var: row 1 s = 4 (e = 3), row 2 s = 16 (e = 12)
    out = []
    for (a, bb), g, s, e in (((x[0], x[1]), go[0:2], F(4), F(3)), ((x[2], x[3]), go[2:4], F(16), F(12))):
        rt = rsqrt(s)                          # exact
        d = e * rt ** 3 / 2                    # dy0/da = -dy0/db = dy1/db = -dy1/da
        ga = g[0] * d - g[1] * d
        out += [ga, -ga]
    assert eps == 0
    agree(r.fields["out"], [nest(out, [2, 2]), None, None])


def test_native_layer_norm_backward_refusals(C):
    x = tens("f32", [F(0), F(2), F(1), F(3)], (2, 2))
    st = tens("f32", [F(1), F(1)], (2,))
    base = [x, x, [2], st, st, None, None, [True, False, False]]
    ok = run(C["NORM"], "aten_native", {"op": "aten.native_layer_norm_backward", "args": base})
    assert ok.outcome == "value"
    bad_count = base[:3] + [tens("f32", [F(1)] * 3, (3,)), st] + base[5:]
    assert run(C["NORM"], "aten_native", {"op": "aten.native_layer_norm_backward", "args": bad_count}).outcome \
        == "invalid_input"
    no_weight = base[:7] + [[True, True, False]]
    assert run(C["NORM"], "aten_native", {"op": "aten.native_layer_norm_backward", "args": no_weight}).outcome \
        == "unsupported"
    inf_stats = base[:3] + [{"t": "f32", "v": [1.0, float("inf")], "shape": [2]}, st] + base[5:]
    assert run(C["NORM"], "aten_native", {"op": "aten.native_layer_norm_backward", "args": inf_stats}).outcome \
        == "unsupported"


def test_native_layer_norm_refusals(C):
    ln = lambda x, ns, eps: {"op": "aten.native_layer_norm", "args": [x, ns, None, None, eps]}  # noqa: E731
    two = tens("f32", [F(2), F(2)], (1, 2))
    assert run(C["NORM"], "aten_native", ln(two, [2], 0)).outcome == "domain_undefined"     # radicand 0
    assert run(C["NORM"], "aten_native", ln(two, [2], -1)).outcome == "domain_undefined"    # radicand < 0
    assert run(C["NORM"], "aten_native", ln(two, [2], F(1, 4))).outcome == "value"
    assert run(C["NORM"], "aten_native", ln(two, [3], 1)).outcome == "invalid_input"
    assert run(C["NORM"], "aten_native", ln(two, [], 1)).outcome == "invalid_input"
    inf = {"t": "f32", "v": [[1.0, float("-inf")]], "shape": [1, 2]}
    assert run(C["NORM"], "aten_native", ln(inf, [2], 1)).outcome == "unsupported"
    i64 = itens([1, 2], (1, 2))
    assert run(C["NORM"], "aten_native", ln(i64, [2], 1)).outcome == "invalid_input"


def test_rms_norm_d5_and_eps_none(C):
    rng = random.Random(11)
    M, N = 4, 64
    x, w = dyadic(rng, M * N), dyadic(rng, N)
    call = {"op": "aten.rms_norm", "args": [tens("f32", x, (M, N)), [N], tens("f32", w, (N,))]}
    r = run(C["NORM"], "aten_native", call)
    assert r.outcome == "value"
    agree(r.fields["out"], nest(my_rms(x, M, N, w, F(1, 2 ** 23)), [M, N]))   # torch.finfo(float32).eps
    # the FlagGems D5 reading (eps 1e-5) is disjoint from the reference on some coordinate
    d5 = my_rms(x, M, N, w, F(1e-5))
    flat = [v for row in r.fields["out"] for v in row]
    assert any(ref_ends(a)[1] < m[0] or ref_ends(a)[0] > m[1] for a, m in zip(flat, d5) if not isinstance(m, F))
    # f64: eps None is 2^-52
    r64 = run(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [tens("f64", x[:N], (1, N)), [N]]})
    agree(r64.fields["out"], nest(my_rms(x[:N], 1, N, None, F(1, 2 ** 52)), [1, N]))
    # f16 / bf16 eps None: unsupported (NM.ATEN_RMS_EPS_NONE); explicit eps answered
    for dt in ("f16", "bf16"):
        assert run(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [tens(dt, x[:N], (1, N)), [N]]}
                   ).outcome == "unsupported"
        re_ = run(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [tens(dt, x[:N], (1, N)), [N], None,
                                                                             F(1, 4)]})
        agree(re_.fields["out"], nest(my_rms(x[:N], 1, N, None, F(1, 4)), [1, N]))
    # kwargs eps and exact rational result
    r3 = run(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [tens("f32", [F(3), F(-3)], (1, 2)), [2]],
                                        "kwargs": {"eps": 7}})
    agree(r3.fields["out"], [[F(3, 4), F(-3, 4)]])
    assert run(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [tens("f32", [F(0), F(0)], (1, 2)), [2],
                                                                         None, 0]}).outcome == "domain_undefined"


def test_layer_norm_check_output_stats_shape_and_dtype(C):
    """Forward stats canonical shape [outer..., 1] is enforced by check_output; a flat [M] (the shape FlagGems was
    recorded to return) is visible as a mismatch. Dtype of stats: input dtype or f32 accepted (NM.ATEN_DTYPES)."""
    # rows [0, 2] and [1, 3], eps 3: mean 1, 2; var 1; rstd 1/2; out -1/2, 1/2
    x = [F(0), F(2), F(1), F(3)]
    call = {"op": "aten.native_layer_norm", "args": [tens("bf16", x, (1, 2, 2)), [2], None, None, 3]}
    out = tens("bf16", [F(-1, 2), F(1, 2)] * 2, (1, 2, 2))
    chk = C["NORM"].adapter.check_output

    def obs(mean_shape, dt):
        return {"out": [out, tens(dt, [F(1), F(2)], mean_shape), tens(dt, [F(1, 2), F(1, 2)], mean_shape)]}
    assert chk("aten_native", call, obs((1, 2, 1), "bf16"))
    assert chk("aten_native", call, obs((1, 2, 1), "f32"))
    assert not chk("aten_native", call, obs((2,), "bf16"))          # flat [M] stats: shape mismatch is visible
    assert not chk("aten_native", call, obs((2,), "f32"))
    assert not chk("aten_native", call, obs((1, 2, 1), "f16"))
    wrong = {"out": [out, tens("f32", [F(1), F(2)], (1, 2, 1)), tens("f32", [F(1, 2), F(1, 4)], (1, 2, 1))]}
    assert not chk("aten_native", call, wrong)


def test_layer_norm_check_output_f32_stats_for_f64_input_is_accepted(C):
    """Documents the breadth of NM.ATEN_DTYPES: float32 statistics are accepted even for a float64 input, although
    neither PyTorch backend produces that (the stated rationale covers reduced-precision inputs only)."""
    x = [F(0), F(2)]
    call = {"op": "aten.native_layer_norm", "args": [tens("f64", x, (1, 2)), [2], None, None, 3]}
    o = {"out": [tens("f64", [F(-1, 2), F(1, 2)], (1, 2)), tens("f32", [F(1)], (1, 1)),
                 tens("f32", [F(1, 2)], (1, 1))]}
    assert C["NORM"].adapter.check_output("aten_native", call, o)


def test_norm_only_aten_profile_answers_and_old_profiles_refuse(C):
    calls = [{"op": "aten.native_layer_norm", "args": [tens("f32", [F(0), F(2)], (1, 2)), [2], None, None, 3]},
             {"op": "aten.rms_norm", "args": [tens("f32", [F(1), F(1)], (1, 2)), [2], None, 3]}]
    for call in calls:
        res = {p: run(C["NORM"], p, call).outcome for p in C["NORM"].data["profiles"]}
        assert res["aten_native"] == "value"
        assert all(v == "unsupported" for p, v in res.items() if p != "aten_native"), res


# ------------------------------------------------------------------ NORM: existing profiles unchanged
def _rq(rng, n, k=40, den=8):
    return [str(F(rng.randint(-k, k), den)) for _ in range(n)]


def _n2(flat, rows):
    step = len(flat) // rows
    return [flat[i * step:(i + 1) * step] for i in range(rows)]


def _same(a, b):
    assert a.outcome == b.outcome, (a, b)
    if a.fields is None:
        return

    def walk(x, y):
        if isinstance(x, dict):
            assert isinstance(y, dict) and set(x) == set(y)
            for k in x:
                walk(x[k], y[k])
        elif isinstance(x, list):
            assert isinstance(y, list) and len(x) == len(y)
            for p, q in zip(x, y):
                walk(p, q)
        elif x is None:
            assert y is None
        else:
            assert ref_ends(x) == ref_ends(y)
    walk(a.fields, b.fields)


def test_norm_existing_profiles_agree_with_validated(C, V):
    rng = random.Random(23)
    n = 0
    for trial in range(30):
        rows, N = rng.randint(1, 3), rng.randint(1, 4)
        x = _n2(_rq(rng, rows * N), rows)
        up = _n2(_rq(rng, rows * N), rows)
        eps = str(F(rng.randint(0, 8), 4))
        w = _rq(rng, N) if rng.random() < 0.6 else None
        b = _rq(rng, N) if rng.random() < 0.6 else None
        cands = [("torch", {"op": "layer_norm", "x": x, "normalized_shape": [N], "weight": w, "bias": b, "eps": eps,
                            "upstream": up}),
                 ("torch_autograd", {"op": "layer_norm", "x": x, "normalized_shape": [N], "weight": w, "bias": b,
                                     "eps": eps, "upstream": up}),
                 ("torch", {"op": "rms_norm", "x": x, "normalized_shape": [N], "weight": w, "eps": eps, "upstream": up}),
                 ("rms_offset", {"op": "rms_norm", "x": x, "normalized_shape": [N], "weight": w, "eps": eps,
                                 "upstream": up, "offset": rng.choice([0, 1])})]
        G = rng.choice([1, 2])
        Cc = G * rng.randint(1, 2)
        L = rng.randint(1, 2)
        gx = [[_rq(rng, L) for _ in range(Cc)] for _ in range(rng.randint(1, 2))]
        gup = [[_rq(rng, L) for _ in range(Cc)] for _ in range(len(gx))]
        cands.append(("torch", {"op": "group_norm", "x": gx, "num_groups": G, "weight": _rq(rng, Cc), "bias": None,
                                "eps": eps, "upstream": gup}))
        for prof, inp in cands:
            a = run(C["NORM"], prof, inp)
            bref = run(V["NORM"], prof, inp)
            _same(a, bref)
            n += a.outcome == "value"
    assert n > 60


# ------------------------------------------------------------------ EMB
def my_emb_bwd(g, idx, Vn, D, pad, freq):
    out = [[F(0)] * D for _ in range(Vn)]
    cnt = {}
    for k in idx:
        cnt[k] = cnt.get(k, 0) + 1
    for p, k in enumerate(idx):
        if k == pad:
            continue
        for d in range(D):
            out[k][d] += g[p * D + d] / (cnt[k] if freq else 1)
    return out


@pytest.mark.parametrize("dtype,Vn,pad,freq", [("f32", 40, -1, False), ("bf16", 2, -1, False), ("f32", 50, 0, False),
                                               ("bf16", 50, 0, False), ("f32", 9, 3, True), ("f64", 7, -1, True)])
def test_embedding_backward_recorded_structure(C, dtype, Vn, pad, freq):
    rng = random.Random(Vn * 31 + pad)
    L, D = 24, 6
    g = dyadic(rng, L * D)
    idx = [rng.randrange(Vn) for _ in range(L)]
    idx[:3] = [0, 0, Vn - 1]
    call = {"op": "aten.embedding_backward", "args": [tens(dtype, g, (1, L, D)), itens(idx, (1, L)), Vn, pad, freq,
                                                      False]}
    r = run(C["EMB"], "aten_dense_backward", call)
    assert r.outcome == "value", r.note
    mine = my_emb_bwd(g, idx, Vn, D, pad, freq)
    agree(r.fields["out"], mine)
    if pad == -1:
        assert any(v != 0 for v in mine[Vn - 1])   # -1 pads no row; the last row receives gradient
    dense = {"op": "aten.embedding_dense_backward", "args": call["args"][:5]}
    assert run(C["EMB"], "aten_dense_backward", dense).fields == r.fields
    i32 = dict(call, args=[call["args"][0], itens(idx, (1, L), "i32")] + call["args"][2:])
    agree(run(C["EMB"], "aten_dense_backward", i32).fields["out"], mine)


def test_embedding_backward_refusals(C):
    g = tens("f32", [F(1), F(2)], (1, 2))
    mk = lambda *a: {"op": "aten.embedding_backward", "args": list(a)}  # noqa: E731
    assert run(C["EMB"], "aten_dense_backward", mk(g, itens([0], (1,)), 2, -1, False, True)).outcome == "unsupported"
    assert run(C["EMB"], "aten_dense_backward", mk(g, itens([2], (1,)), 2, -1, False, False)).outcome \
        == "invalid_input"
    assert run(C["EMB"], "aten_dense_backward", mk(g, itens([-1], (1,)), 2, -1, False, False)).outcome \
        == "invalid_input"
    assert run(C["EMB"], "aten_dense_backward", mk(g, itens([0, 1], (2,)), 2, -1, False, False)).outcome \
        == "invalid_input"
    # padding_idx outside [0, num_weights) pads nothing (literal comparison)
    r = run(C["EMB"], "aten_dense_backward", mk(g, itens([1], (1,)), 2, 5, False, False))
    agree(r.fields["out"], [[F(0), F(0)], [F(1), F(2)]])
    # empty indices: all zeros (EM.ATEN_BACKWARD)
    r0 = run(C["EMB"], "aten_dense_backward", mk({"t": "f32", "v": [], "shape": [0, 2]}, itens([], (0,)), 2, -1,
                                                 False, False))
    assert r0.outcome == "value"
    agree(r0.fields["out"], [[F(0), F(0)], [F(0), F(0)]])
    for p in C["EMB"].data["profiles"]:
        if p != "aten_dense_backward":
            assert run(C["EMB"], p, mk(g, itens([0], (1,)), 2, -1, False, False)).outcome == "unsupported"


def test_emb_existing_profile_agrees_with_validated(C, V):
    rng = random.Random(5)
    n = 0
    for _ in range(40):
        Vn, D, L = rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 5)
        weight = [_rq(rng, D) for _ in range(Vn)]
        idx = [rng.randrange(Vn) for _ in range(L)]
        inp = {"weight": weight, "indices": idx, "upstream": [_rq(rng, D) for _ in range(L)],
               "dtype": rng.choice(["f32", "f64", "bf16"]), "index_dtype": "i64",
               "padding_idx": rng.choice([None, 0, -1, Vn - 1]), "max_norm": None, "norm_type": 2,
               "scale_grad_by_freq": rng.random() < 0.5, "sparse": False}
        a = run(C["EMB"], "torch210_dense_real", inp)
        b = run(V["EMB"], "torch210_dense_real", inp)
        _same(a, b)
        n += a.outcome == "value"
    assert n > 20


# ------------------------------------------------------------------ NLL
def my_nll_fwd(x, rows, Cn, t, w, red, ign):
    loss, tw = [], F(0)
    for n, c in enumerate(t):
        if c == ign:
            loss.append(F(0))
            continue
        wc = w[c] if w else F(1)
        loss.append(-wc * x[n * Cn + c])
        tw += wc
    if red == 0 and rows is not None:
        return [loss, F(0)]
    s = sum(loss, F(0))
    return [s / tw if red == 1 else s, tw]


def my_nll_bwd(go, rows, Cn, t, w, red, ign, tw):
    out = [F(0)] * ((rows or 1) * Cn)
    for n, c in enumerate(t):
        if c == ign:
            continue
        gv = go[n] if (red == 0 and rows is not None) else go[0]
        if red == 1:
            gv = gv / tw
        out[n * Cn + c] = -(w[c] if w else 1) * gv
    return out


@pytest.mark.parametrize("red,weighted", [(1, False), (2, False), (0, False), (1, True), (2, True), (0, True)])
def test_nll_recorded_structure(C, red, weighted):
    rng = random.Random(red * 10 + weighted)
    Nn, Cn = 32, 40
    x = [F(-rng.randint(0, 400), 32) for _ in range(Nn * Cn)]   # log-probability-like (<= 0)
    t = [rng.randrange(Cn) for _ in range(Nn)]
    for i in range(0, Nn, 5):
        t[i] = -100
    w = [F(rng.randint(1, 16), 8) for _ in range(Cn)] if weighted else None
    fwd = {"op": "aten.nll_loss_forward", "args": [tens("f32", x, (Nn, Cn)), itens(t, (Nn,)),
                                                   tens("f32", w, (Cn,)) if w else None, red, -100]}
    r = run(C["NLL"], "aten_real", fwd)
    assert r.outcome == "value", r.note
    mine = my_nll_fwd(x, Nn, Cn, t, w, red, -100)
    agree(r.fields["out"], mine)
    tw = mine[1]
    go = [F(3, 4)] if red else [F(rng.randint(-8, 8), 4) for _ in range(Nn)]
    gshape = () if red else (Nn,)
    supplied = tw if red != 0 else F(0)
    bwd = {"op": "aten.nll_loss_backward", "args": [tens("f32", go, gshape), tens("f32", x, (Nn, Cn)), itens(t, (Nn,)),
                                                    tens("f32", w, (Cn,)) if w else None, red, -100,
                                                    tens("f32", [supplied], ())]}
    rb = run(C["NLL"], "aten_real", bwd)
    assert rb.outcome == "value", rb.note
    agree(rb.fields["out"], nest(my_nll_bwd(go, Nn, Cn, t, w, red, -100, supplied), [Nn, Cn]))


def test_nll_forward_edges(C):
    fwd = lambda x, xs, t, ts, w, red, ign=-100: {"op": "aten.nll_loss_forward", "args": [  # noqa: E731
        tens("f32", x, xs), itens(t, ts), tens("f32", w, (xs[-1],)) if w else None, red, ign]}
    x = [F(-1), F(-2), F(-3), F(-4)]
    # all ignored: mean undefined, sum 0 with total_weight 0, none zeros with total_weight 0
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [-100, -100], (2,), None, 1)).outcome == "domain_undefined"
    agree(run(C["NLL"], "aten_real", fwd(x, (2, 2), [-100, -100], (2,), None, 2)).fields["out"], [F(0), F(0)])
    agree(run(C["NLL"], "aten_real", fwd(x, (2, 2), [-100, -100], (2,), None, 0)).fields["out"],
          [[F(0), F(0)], F(0)])
    # non-ignored targets whose class weights sum to 0: mean undefined (nonzero / 0)
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, 1], (2,), [F(1), F(-1)], 1)).outcome == "domain_undefined"
    agree(run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, 1], (2,), [F(1), F(-1)], 2)).fields["out"],
          [F(1) - F(4), F(0)])
    # empty batch
    e = {"op": "aten.nll_loss_forward", "args": [{"t": "f32", "v": [], "shape": [0, 3]}, itens([], (0,)), None, 1, -100]}
    assert run(C["NLL"], "aten_real", e).outcome == "domain_undefined"
    e2 = dict(e, args=e["args"][:3] + [2, -100])
    agree(run(C["NLL"], "aten_real", e2).fields["out"], [F(0), F(0)])
    # 1D self: reduction none returns the single loss and total_weight = w[t] (native frame takes the scalar path)
    one = {"op": "aten.nll_loss_forward", "args": [tens("f64", [F(-1), F(-5, 2)], (2,)), itens([1], ()),
                                                   tens("f64", [F(2), F(3)], (2,)), 0, -100]}
    agree(run(C["NLL"], "aten_real", one).fields["out"], [F(15, 2), F(3)])
    # ignore_index equal to a class index
    agree(run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, 1], (2,), None, 1, ign=0)).fields["out"], [F(4), F(1)])
    # refusals
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, 2], (2,), None, 1)).outcome == "invalid_input"
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, -5], (2,), None, 1)).outcome == "invalid_input"
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [0, 1], (2,), None, 3)).outcome == "invalid_input"
    assert run(C["NLL"], "aten_real", fwd(x, (2, 2), [0], (1,), None, 1)).outcome == "invalid_input"
    assert run(C["NLL"], "aten_real", fwd(x, (1, 2, 2), [0, 1], (1, 2), None, 1)).outcome == "invalid_input"
    i32 = {"op": "aten.nll_loss_forward", "args": [tens("f32", x, (2, 2)), itens([0, 1], (2,), "i32"), None, 1, -100]}
    assert run(C["NLL"], "aten_real", i32).outcome == "unsupported"
    nan = {"op": "aten.nll_loss_forward", "args": [{"t": "f32", "v": [[float("nan"), 0.0]], "shape": [1, 2]},
                                                   itens([1], (1,)), None, 1, -100]}
    assert run(C["NLL"], "aten_real", nan).outcome in ("invalid_input", "unsupported")


def test_nll_backward_edges(C):
    x = tens("f32", [F(-1), F(-2), F(-3), F(-4)], (2, 2))
    bwd = lambda go, gs, t, red, tw, w=None: {"op": "aten.nll_loss_backward", "args": [  # noqa: E731
        tens("f32", go, gs), x, itens(t, (2,)), tens("f32", w, (2,)) if w else None, red, -100, tens("f32", [tw], ())]}
    # supplied total_weight is used, not the recomputed one (NL.BWD_SUPPLIED)
    agree(run(C["NLL"], "aten_real", bwd([F(1)], (), [0, 1], 1, F(5))).fields["out"],
          [[F(-1, 5), F(0)], [F(0), F(-1, 5)]])
    # all ignored, total_weight 0: zeros (NL.BWD_EMPTY); one non-ignored with total_weight 0: undefined
    agree(run(C["NLL"], "aten_real", bwd([F(1)], (), [-100, -100], 1, F(0))).fields["out"],
          [[F(0), F(0)], [F(0), F(0)]])
    assert run(C["NLL"], "aten_real", bwd([F(1)], (), [1, -100], 1, F(0))).outcome == "domain_undefined"
    # sum / none ignore total_weight's value
    agree(run(C["NLL"], "aten_real", bwd([F(2)], (), [1, 0], 2, F(0), [F(3), F(5)])).fields["out"],
          [[F(0), F(-10)], [F(-6), F(0)]])
    agree(run(C["NLL"], "aten_real", bwd([F(2), F(-1)], (2,), [1, -100], 0, F(7))).fields["out"],
          [[F(0), F(-2)], [F(0), F(0)]])
    # shape refusals
    assert run(C["NLL"], "aten_real", bwd([F(1)], (), [0, 1], 0, F(0))).outcome == "invalid_input"
    assert run(C["NLL"], "aten_real", bwd([F(1), F(1)], (2,), [0, 1], 1, F(2))).outcome == "invalid_input"
    two_tw = {"op": "aten.nll_loss_backward", "args": [tens("f32", [F(1)], ()), x, itens([0, 1], (2,)), None, 1, -100,
                                                       tens("f32", [F(1), F(1)], (2,))]}
    assert run(C["NLL"], "aten_real", two_tw).outcome == "invalid_input"
    # values of self are not read: NaN storage allowed in the backward
    nan_self = {"op": "aten.nll_loss_backward", "args": [tens("f32", [F(1)], ()),
                                                         {"t": "f32", "v": [[float("nan"), 0.0]], "shape": [1, 2]},
                                                         itens([1], (1,)), None, 2, -100, tens("f32", [F(1)], ())]}
    agree(run(C["NLL"], "aten_real", nan_self).fields["out"], [[F(0), F(-1)]])
    # the diagnostic reading NL.R1 recomputes and so differs on the supplied-total_weight call
    call = bwd([F(1)], (), [0, 1], 1, F(5))
    rr = evaluate(C["NLL"].adapter.READINGS["NL.R1"], "aten_real", call)
    agree(rr.fields["out"], [[F(-1, 2), F(0)], [F(0), F(-1, 2)]])


def test_nll_harness_shape(C):
    c = C["NLL"]
    assert set(c.data["covers"]) == {"aten.nll_loss_forward", "aten.nll_loss_backward"}
    prof = c.data["profiles"]
    assert list(prof) == ["aten_real"] and prof["aten_real"]["task_choice_required"] is False
    assert "make_reference" in Path(c.adapter_path).read_text()
    assert c.adapter.ATEN_RETURNS == {"aten.nll_loss_forward": {"positions": 2, "not_compared": {}}}
