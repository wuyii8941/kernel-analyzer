"""Path A closure package B (2026-10-20): ATen-call interface of the ATTN draft for the fused SDPA overloads.

The draft pair contracts/draft/ATTN_path_a_A2_20261020.yaml + ATTN_adapter_path_a_A2_20261020.py is staged into a
temporary <stage>/contracts, <stage>/adapters layout and loaded through kernel_spec.registry.load. Expected values are
computed independently here (float64 NumPy formulas and PyTorch 2.10 float64 autograd) and compared with the
reference's rigorous enclosures; optional GPU checks compare the PyTorch 2.10 CUDA kernels with the reference.
"""
from __future__ import annotations

import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import ends, evaluate  # noqa: E402

DRAFT = SPEC / "contracts" / "draft"
PROFILE = "torch210_aten_sdpa_fused_dropout0"
FLASH = "aten._scaled_dot_product_flash_attention"
FLASH_BWD = "aten._scaled_dot_product_flash_attention_backward"
EFF_BWD = "aten._scaled_dot_product_efficient_attention_backward"


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root = tmp_path_factory.mktemp("attn_aten_stage")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    shutil.copy(DRAFT / "ATTN_path_a_A2_20261020.yaml", root / "contracts" / "ATTN.yaml")
    shutil.copy(DRAFT / "ATTN_adapter_path_a_A2_20261020.py", root / "adapters" / "ATTN.py")
    return root


@pytest.fixture(scope="module")
def contract(stage):
    return registry.load("ATTN", contracts=stage / "contracts", adapters=stage / "adapters")


def run(contract, inp):
    return evaluate(contract.adapter.reference, PROFILE, inp)


def mid(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [mid(e) for e in x]
    if isinstance(x, (int, Fraction)):
        return float(x)
    lo, hi = ends(x)
    return (float(lo) + float(hi)) / 2


def rec(arr, t):
    arr = np.asarray(arr)
    return {"t": t, "v": arr.tolist(), "shape": list(arr.shape)}


def bf16_values(rng, shape, scale=4):
    """Random values exactly representable in bf16 (and f32): small dyadic rationals."""
    return rng.integers(-scale * 8, scale * 8 + 1, size=shape) / 8.0


def np_forward(q, k, v, causal, scale):
    B, Hq, L, E = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    kk = np.repeat(k, Hq // Hkv, axis=1)
    vv = np.repeat(v, Hq // Hkv, axis=1)
    s = np.einsum("bhie,bhje->bhij", q, kk) * scale
    if causal:
        s = np.where(np.tril(np.ones((L, S), dtype=bool)), s, -np.inf)
    m = s.max(-1, keepdims=True)
    w = np.exp(s - m)
    lse = (m + np.log(w.sum(-1, keepdims=True)))[..., 0]
    return np.einsum("bhij,bhjd->bhid", w / w.sum(-1, keepdims=True), vv), lse


def np_saved_backward(g, q, k, v, o, lse, causal, scale):
    """FlashAttention-2 Algorithm 2 with the supplied O and L, written independently in float64."""
    B, Hq, L, E = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    grp = Hq // Hkv
    kk = np.repeat(k, grp, axis=1)
    vv = np.repeat(v, grp, axis=1)
    s = np.einsum("bhie,bhje->bhij", q, kk) * scale
    p = np.exp(s - lse[..., None])
    if causal:
        p = np.where(np.tril(np.ones((L, S), dtype=bool)), p, 0.0)
    D = (g * o).sum(-1)
    dP = np.einsum("bhid,bhjd->bhij", g, vv)
    dS = p * (dP - D[..., None]) * scale
    dq = np.einsum("bhij,bhje->bhie", dS, kk)
    dk = np.einsum("bhij,bhie->bhje", dS, q).reshape(B, Hkv, grp, S, E).sum(2)
    dv = np.einsum("bhij,bhid->bhjd", p, g).reshape(B, Hkv, grp, S, v.shape[3]).sum(2)
    return dq, dk, dv


def flash_call(q, k, v, causal, scale, dtype="bf16"):
    return {"op": FLASH, "args": [rec(q, dtype), rec(k, dtype), rec(v, dtype), 0.0, causal],
            "kwargs": {"scale": scale}}


def flash_bwd_call(g, q, k, v, o, lse, causal, scale, dtype="bf16"):
    L, S = q.shape[2], k.shape[2]
    return {"op": FLASH_BWD, "args": [rec(g, dtype), rec(q, dtype), rec(k, dtype), rec(v, dtype), rec(o, dtype),
                                      rec(lse, "f32"), None, None, L, S, 0.0, causal,
                                      {"t": "i64", "v": [0, 0], "shape": [2]}, {"t": "i64", "v": 0, "shape": []}],
            "kwargs": {"scale": scale}}


def eff_bwd_call(g, q, k, v, o, lse, causal, scale, dropout=0.0):
    args = [rec(g, "f32"), rec(q, "f32"), rec(k, "f32"), rec(v, "f32"), None, rec(o, "f32"), rec(lse, "f32"),
            {"t": "i64", "v": 0, "shape": []}, {"t": "i64", "v": 0, "shape": []}, dropout, [True, True, True, False]]
    if causal:
        args.append(True)
    return {"op": EFF_BWD, "args": args, "kwargs": {"scale": scale}}


def to_bf16(x):
    """Round float64 values to bf16 (round to nearest even), returned as float64."""
    import torch
    return torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).double().numpy()


# ------------------------------------------------------------------ interface


def test_checker_consistent(stage):
    sys.path.insert(0, str(SPEC / "tools"))
    import json
    import check_contracts as CC
    rep = CC.check_contract(stage / "contracts" / "ATTN.yaml", json.loads((SPEC / "SOURCES.json").read_text()),
                            stage / "adapters")
    assert rep["verdict"] == "consistent", rep
    assert all(m["rejected"] for m in rep["mutations"])


def test_adapter_exposes_aten_interface(contract):
    text = contract.adapter_path.read_text() if hasattr(contract, "adapter_path") else \
        (Path(contract.adapter.__file__)).read_text()
    assert "make_reference" in text
    for op in (FLASH, FLASH_BWD, EFF_BWD):
        assert op in contract.data["covers"]
    returns = contract.adapter.ATEN_RETURNS
    assert returns[FLASH]["positions"] == 9 and set(returns[FLASH]["not_compared"]) == {"2", "3", "6", "7", "8"}
    assert returns[FLASH_BWD] == {"positions": 3, "not_compared": {}}
    assert returns[EFF_BWD] == {"positions": 4, "not_compared": {}}
    assert contract.data["profiles"][PROFILE]["task_choice_required"] is False


def test_not_compared_positions_are_null(contract):
    rng = np.random.default_rng(0)
    q, k, v = (bf16_values(rng, s) for s in ((1, 2, 3, 2), (1, 1, 3, 2), (1, 1, 3, 2)))
    r = run(contract, flash_call(q, k, v, True, 0.5))
    assert r.outcome == "value"
    out = r.fields["out"]
    assert len(out) == 9
    for i in (2, 3, 6, 7, 8):
        assert out[i] is None
    assert out[4] == 3 and out[5] == 3


# ------------------------------------------------------------------ independent float64 cross-checks


@pytest.mark.parametrize("causal,scale,heads", [(True, 0.125, (4, 2)), (False, 0.5, (2, 2)), (True, None, (3, 1))])
def test_flash_forward_matches_float64(contract, causal, scale, heads):
    rng = np.random.default_rng(1)
    Hq, Hkv = heads
    q = bf16_values(rng, (1, Hq, 4, 4))
    k = bf16_values(rng, (1, Hkv, 4, 4))
    v = bf16_values(rng, (1, Hkv, 4, 4))
    r = run(contract, flash_call(q, k, v, causal, scale))
    assert r.outcome == "value", r.note
    o_ref, lse_ref = np_forward(q, k, v, causal, 0.5 if scale is None else scale)
    np.testing.assert_allclose(np.array(mid(r.fields["out"][0])), o_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(mid(r.fields["out"][1])), lse_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("causal", [True, False])
def test_flash_backward_uses_supplied_values(contract, causal):
    rng = np.random.default_rng(2)
    q = bf16_values(rng, (1, 4, 4, 2))
    k = bf16_values(rng, (1, 2, 4, 2))
    v = bf16_values(rng, (1, 2, 4, 2))
    g = bf16_values(rng, (1, 4, 4, 2))
    o, lse = np_forward(q, k, v, causal, 0.25)
    o = to_bf16(o)                                  # what a bf16 forward returns
    # float32 logsumexp, deliberately shifted by 1/2 so that only the supplied value can match
    lse = (lse + 0.5).astype(np.float32).astype(np.float64)
    r = run(contract, flash_bwd_call(g, q, k, v, o, lse, causal, 0.25))
    assert r.outcome == "value", r.note
    exp = np_saved_backward(g, q, k, v, o, lse, causal, 0.25)
    for got, want in zip(r.fields["out"], exp):
        np.testing.assert_allclose(np.array(mid(got)), want, rtol=1e-11, atol=1e-11)


def test_recompute_reading_matches_torch_autograd(contract):
    import torch
    rng = np.random.default_rng(3)
    q = bf16_values(rng, (1, 4, 3, 2))
    k = bf16_values(rng, (1, 2, 3, 2))
    v = bf16_values(rng, (1, 2, 3, 2))
    g = bf16_values(rng, (1, 4, 3, 2))
    o, lse = np_forward(q, k, v, True, 0.5)
    inp = flash_bwd_call(g, q, k, v, to_bf16(o), lse.astype(np.float32).astype(np.float64), True, 0.5)
    r = evaluate(contract.adapter.READINGS["recompute_saved"], PROFILE, inp)
    qt, kt, vt = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=0.5, enable_gqa=True)
    out.backward(torch.tensor(g, dtype=torch.float64))
    for got, t in zip(r.fields["out"], (qt, kt, vt)):
        np.testing.assert_allclose(np.array(mid(got)), t.grad.numpy(), rtol=1e-10, atol=1e-10)


def test_efficient_backward_padded_lse(contract):
    rng = np.random.default_rng(4)
    q, k, v, g = (bf16_values(rng, (1, 2, 5, 3)) for _ in range(4))
    o, lse = np_forward(q, k, v, False, 0.125)
    o = o.astype(np.float32).astype(np.float64)
    lse = lse.astype(np.float32).astype(np.float64)
    padded = np.concatenate([lse, np.full((1, 2, 27), np.inf)], axis=-1)
    r = run(contract, eff_bwd_call(g, q, k, v, o, padded, False, 0.125))
    assert r.outcome == "value", r.note
    assert r.fields["out"][3] is None
    for got, want in zip(r.fields["out"][:3], np_saved_backward(g, q, k, v, o, lse, False, 0.125)):
        np.testing.assert_allclose(np.array(mid(got)), want, rtol=1e-11, atol=1e-11)
    bad = padded.copy()
    bad[..., 7] = 0.0
    assert run(contract, eff_bwd_call(g, q, k, v, o, bad, False, 0.125)).outcome == "unsupported"


# ------------------------------------------------------------------ refusals and recorded argument structures


def test_dropout_refused_with_reason(contract):
    rng = np.random.default_rng(5)
    q, k, v, g, o = (bf16_values(rng, (1, 2, 2, 2)) for _ in range(5))
    lse = np.zeros((1, 2, 2))
    r = run(contract, eff_bwd_call(g, q, k, v, o, lse, True, 0.125, dropout=0.1))
    assert r.outcome == "unsupported" and "RNG" in r.note
    call = flash_call(q, k, v, True, 0.125)
    call["args"][3] = 0.1
    r = run(contract, call)
    assert r.outcome == "unsupported" and "RNG" in r.note


def test_refusal_reasons(contract):
    rng = np.random.default_rng(6)
    q = bf16_values(rng, (1, 2, 1, 2))
    k = bf16_values(rng, (1, 1, 1, 2))
    r = run(contract, eff_bwd_call(q, q, k, k, q, np.zeros((1, 2, 1)), False, 0.5))
    assert r.outcome == "unsupported" and "head counts" in r.note
    q2 = bf16_values(rng, (1, 1, 2, 2))
    k2 = bf16_values(rng, (1, 1, 3, 2))
    r = run(contract, flash_call(q2, k2, k2, True, 0.5))
    assert r.outcome == "unsupported" and "AT.ATEN_CAUSAL" in r.note
    r = run(contract, flash_call(q2, k2, k2, False, 0.5, dtype="f32"))
    assert r.outcome == "unsupported"
    call = flash_call(q2, k2, k2, False, 0.5)
    call["args"][0]["v"][0][0][0][0] = 0.1          # not a bf16 value
    assert run(contract, call).outcome == "invalid_input"


def test_value_head_dim(contract):
    """Flash requires Ev == E (output shaped like query); the efficient backward allows Ev != E."""
    rng = np.random.default_rng(9)
    q, k = bf16_values(rng, (1, 1, 2, 2)), bf16_values(rng, (1, 1, 2, 2))
    v = bf16_values(rng, (1, 1, 2, 1))
    r = run(contract, flash_call(q, k, v, False, 0.5))
    assert r.outcome == "invalid_input" and "value head dimension" in r.note
    g = bf16_values(rng, (1, 1, 2, 1))
    o, lse = np_forward(q, k, v, False, 0.5)
    o = o.astype(np.float32).astype(np.float64)
    lse = lse.astype(np.float32).astype(np.float64)
    padded = np.concatenate([lse, np.full((1, 1, 30), np.inf)], axis=-1)
    r = run(contract, eff_bwd_call(g, q, k, v, o, padded, False, 0.5))
    assert r.outcome == "value", r.note
    for got, want in zip(r.fields["out"][:3], np_saved_backward(g, q, k, v, o, lse, False, 0.5)):
        np.testing.assert_allclose(np.array(mid(got)), want, rtol=1e-11, atol=1e-11)


def test_recorded_qwen_flash_structure_reduced(contract):
    """The recorded qwen bf16 argument layout (Hq=14, Hkv=2, causal, scale 0.125) with a reduced length."""
    rng = np.random.default_rng(7)
    L, E = 3, 2
    q = bf16_values(rng, (1, 14, L, E))
    k = bf16_values(rng, (1, 2, L, E))
    v = bf16_values(rng, (1, 2, L, E))
    r = run(contract, {"op": FLASH, "args": [rec(q, "bf16"), rec(k, "bf16"), rec(v, "bf16"), 0.0, True],
                       "kwargs": {"scale": 0.125}})
    assert r.outcome == "value", r.note
    o, lse = np_forward(q, k, v, True, 0.125)
    np.testing.assert_allclose(np.array(mid(r.fields["out"][0])), o, rtol=1e-12, atol=1e-12)
    g = bf16_values(rng, (1, 14, L, E))
    rb = run(contract, flash_bwd_call(g, q, k, v, to_bf16(o), lse.astype(np.float32).astype(np.float64), True, 0.125))
    assert rb.outcome == "value", rb.note
    assert [np.array(mid(x)).shape for x in rb.fields["out"]] == [(1, 14, L, E), (1, 2, L, E), (1, 2, L, E)]


def test_check_output(contract):
    inp = next(e["input"] for e in contract.data["examples"] if e["id"] == "AT.E30")
    t = lambda x: {"t": "bf16", "v": x, "shape": [1, 1, 1, 1]}  # noqa: E731
    adapter = contract.adapter
    assert adapter.check_output(PROFILE, inp, {"out": [t([[[[0.0]]]]), t([[[[0.0]]]]), t([[[[5.0]]]])]}) is True
    assert adapter.check_output(PROFILE, inp, {"out": [t([[[[0.0]]]]), t([[[[0.0]]]]), t([[[[4.0]]]])]}) is False
    fwd = next(e["input"] for e in contract.data["examples"] if e["id"] == "AT.E23")
    o = {"t": "bf16", "v": [[[[1.0]], [[1.0]], [[2.0]], [[2.0]]]], "shape": [1, 4, 1, 1]}
    lse = {"t": "f32", "v": [[[0.0], [0.0], [0.0], [0.0]]], "shape": [1, 4, 1]}
    junk = {"t": "i64", "v": [123, 4], "shape": [2]}
    assert adapter.check_output(PROFILE, fwd, {"out": [o, lse, None, None, 1, 1, junk, junk, None]}) is True
    assert adapter.check_output(PROFILE, fwd, {"out": [o, lse, None, None, 2, 1, junk, junk, None]}) is False


# ------------------------------------------------------------------ PyTorch 2.10 CUDA kernels (supporting evidence)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda(), reason="CUDA not available")
def test_pytorch_cuda_flash_agrees(contract):
    import torch
    rng = np.random.default_rng(8)
    q = bf16_values(rng, (1, 4, 8, 8), 1)
    k = bf16_values(rng, (1, 2, 8, 8), 1)
    v = bf16_values(rng, (1, 2, 8, 8), 1)
    dev = lambda x: torch.tensor(x, dtype=torch.bfloat16, device="cuda")  # noqa: E731
    res = torch.ops.aten._scaled_dot_product_flash_attention(dev(q), dev(k), dev(v), 0.0, True, scale=0.125)
    r = run(contract, flash_call(q, k, v, True, 0.125))
    np.testing.assert_allclose(res[0].double().cpu().numpy(), np.array(mid(r.fields["out"][0])), atol=2e-2)
    np.testing.assert_allclose(res[1].double().cpu().numpy(), np.array(mid(r.fields["out"][1])), atol=1e-5)
    assert res[4] == 8 and res[5] == 8 and res[2] is None and res[3] is None
    g = bf16_values(rng, (1, 4, 8, 8), 1)
    o = res[0].double().cpu().numpy()
    lse = res[1].double().cpu().numpy()
    grads = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        dev(g), dev(q), dev(k), dev(v), res[0], res[1], None, None, 8, 8, 0.0, True, res[6], res[7], scale=0.125)
    rb = run(contract, flash_bwd_call(g, q, k, v, o, lse, True, 0.125))
    for got, want in zip(grads, rb.fields["out"]):
        np.testing.assert_allclose(got.double().cpu().numpy(), np.array(mid(want)), atol=6e-2)
