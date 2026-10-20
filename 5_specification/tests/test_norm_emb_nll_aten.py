"""Path A closure package A (norm_emb_nll): ATen-call interfaces of the staged NORM, EMB and NLL drafts.

The drafts in contracts/draft/<ID>_path_a_A2_20261020.yaml / <ID>_adapter_path_a_A2_20261020.py are staged into
tmp_path (contracts/<ID>.yaml, adapters/<ID>.py) and loaded with kernel_spec.registry.load. Checks:
- harness conditions: covers, make_reference, exactly one non-task-choice profile answers each recorded call
  structure, ATEN_RETURNS and return lengths;
- PyTorch 2.10 eager (CPU, float64 computation of the same call where possible) agrees with the exact reference on
  the recorded argument structures (supporting evidence, not the definition);
- the decisions that differ from tempting readings (rms_norm eps=None, supplied statistics / total_weight, ATen
  padding_idx -1, all-ignored mean).
"""
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC)]
from kernel_spec import registry, schema  # noqa: E402
from kernel_spec.num import IV, ends, evaluate  # noqa: E402

DRAFT = SPEC / "contracts" / "draft"
IDS = ("NORM", "EMB", "NLL")
TORCH = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}
ATEN_PROFILE = {"NORM": "aten_native", "EMB": "aten_dense_backward", "NLL": "aten_real"}


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage_A")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid in IDS:
        shutil.copy(DRAFT / f"{cid}_path_a_A2_20261020.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_path_a_A2_20261020.py", root / "adapters" / f"{cid}.py")
    return root


@pytest.fixture(scope="module")
def C(stage):
    return {cid: registry.load(cid, contracts=stage / "contracts", adapters=stage / "adapters") for cid in IDS}


def rec(t):
    t = t.detach().cpu()
    return {"t": {v: k for k, v in TORCH.items()}.get(t.dtype, "i64"),
            "v": (t.double() if t.is_floating_point() else t).tolist(), "shape": list(t.shape)}


def mid(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [mid(x) for x in v]
    if isinstance(v, IV):
        lo, hi = ends(v)
        return float((lo + hi) / 2)
    return float(v)


def close(ref, eager, rtol=1e-9, atol=1e-12):
    if ref is None or eager is None:
        assert ref is None and eager is None
        return
    a = np.asarray(mid(ref), dtype=np.float64)
    b = eager.detach().cpu().double().numpy()
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def answer(c, profile, call):
    return evaluate(c.adapter.reference, profile, call)


def only_aten_profile_answers(c, call):
    applies = [p for p, d in c.data["profiles"].items() if d["task_choice_required"] is False
               and answer(c, p, call).outcome in ("value", "domain_undefined")]
    assert applies == [ATEN_PROFILE[c.id]], applies


# ------------------------------------------------------------------ harness conditions
def test_harness_conditions(C):
    expected = {"NORM": {"aten.native_layer_norm", "aten.native_layer_norm_backward", "aten.rms_norm"},
                "EMB": {"aten.embedding_backward", "aten.embedding_dense_backward"},
                "NLL": {"aten.nll_loss_forward", "aten.nll_loss_backward"}}
    for cid, c in C.items():
        assert expected[cid] <= set(c.data["covers"])
        prof = c.data["profiles"][ATEN_PROFILE[cid]]
        assert prof["task_choice_required"] is False and list(schema.profile_outputs(c.data, ATEN_PROFILE[cid])) == ["out"]
    assert C["NORM"].adapter.ATEN_RETURNS == {"aten.native_layer_norm": {"positions": 3, "not_compared": {}},
                                              "aten.native_layer_norm_backward": {"positions": 3, "not_compared": {}}}
    assert C["NLL"].adapter.ATEN_RETURNS == {"aten.nll_loss_forward": {"positions": 2, "not_compared": {}}}
    assert not hasattr(C["EMB"].adapter, "ATEN_RETURNS")


def test_adapters_read_their_staged_contract(stage):
    for cid in IDS:
        text = (stage / "adapters" / f"{cid}.py").read_text()
        assert "make_reference" in text
        assert "from kernel_spec.registry import CONTRACTS" not in text


def test_check_contracts_on_stage(stage):
    for cid in IDS:
        r = subprocess.run([sys.executable, "tools/check_contracts.py", "--contracts", str(stage / "contracts"),
                            "--adapters", str(stage / "adapters"), "--only", cid], cwd=SPEC, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


# ------------------------------------------------------------------ NORM, recorded structures
@pytest.mark.parametrize("dtype,shape,eps", [("f32", (1, 128, 128), 1e-12), ("bf16", (1, 128, 128), 1e-12),
                                             ("f32", (1, 32, 768), 1e-05)])
def test_native_layer_norm_recorded_structure(C, dtype, shape, eps):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(shape, generator=g).to(TORCH[dtype])
    w = torch.randn(shape[-1], generator=g).to(TORCH[dtype])
    b = torch.randn(shape[-1], generator=g).to(TORCH[dtype])
    call = {"op": "aten.native_layer_norm", "args": [rec(x), [shape[-1]], rec(w), rec(b), eps]}
    only_aten_profile_answers(C["NORM"], call)
    r = answer(C["NORM"], "aten_native", call)
    assert r.outcome == "value" and len(r.fields["out"]) == 3
    o, m, s = torch.ops.aten.native_layer_norm(x.double(), [shape[-1]], w.double(), b.double(), eps)
    for ref, ev in zip(r.fields["out"], (o, m, s)):
        close(ref, ev)
    assert list(np.asarray(mid(r.fields["out"][1])).shape) == [shape[0], shape[1], 1]


def test_native_layer_norm_backward_recorded_structure_supplied_flat_stats(C):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 128, 128, generator=g).bfloat16()
    w, b = torch.randn(128, generator=g).bfloat16(), torch.randn(128, generator=g).bfloat16()
    go = torch.randn(1, 128, 128, generator=g).bfloat16()
    # Statistics as a bf16 producer returns them, flat [M] (the recorded FlagGems structure) and rounded.
    _, m, s = torch.ops.aten.native_layer_norm(x.float(), [128], w.float(), b.float(), 1e-12)
    m, s = m.reshape(128).bfloat16(), s.reshape(128).bfloat16()
    call = {"op": "aten.native_layer_norm_backward",
            "args": [rec(go), rec(x), [128], rec(m), rec(s), rec(w), rec(b), [True, True, True]]}
    only_aten_profile_answers(C["NORM"], call)
    r = answer(C["NORM"], "aten_native", call)
    assert r.outcome == "value"
    eager = torch.ops.aten.native_layer_norm_backward(go.double(), x.double(), [128], m.double(), s.double(),
                                                      w.double(), b.double(), [True, True, True])
    for ref, ev in zip(r.fields["out"], eager):
        close(ref, ev, rtol=1e-8, atol=1e-10)
    masked = dict(call, args=call["args"][:7] + [[True, False, False]])
    out = answer(C["NORM"], "aten_native", masked).fields["out"]
    assert out[1] is None and out[2] is None


def test_rms_norm_d5_structure_eps_none(C):
    g = torch.Generator().manual_seed(3)
    x, w = torch.randn(4, 64, generator=g), torch.randn(64, generator=g)
    call = {"op": "aten.rms_norm", "args": [rec(x), [64], rec(w)]}
    only_aten_profile_answers(C["NORM"], call)
    r = answer(C["NORM"], "aten_native", call)
    explicit = answer(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [rec(x), [64], rec(w), 2.0 ** -23]})
    assert mid(r.fields["out"]) == mid(explicit.fields["out"])
    close(r.fields["out"], torch.ops.aten.rms_norm(x.double(), [64], w.double(), 2.0 ** -23))
    far = answer(C["NORM"], "aten_native", {"op": "aten.rms_norm", "args": [rec(x), [64], rec(w), 1e-5]})
    assert mid(far.fields["out"]) != mid(r.fields["out"])
    half = {"op": "aten.rms_norm", "args": [rec(x.bfloat16()), [64], rec(w.bfloat16())]}
    assert answer(C["NORM"], "aten_native", half).outcome == "unsupported"


def test_layer_norm_check_output_stats_dtype_is_metadata(C):
    x = torch.tensor([[0.0, 2.0]]).bfloat16()
    call = {"op": "aten.native_layer_norm", "args": [rec(x), [2], None, None, 3]}
    out = rec(torch.tensor([[-0.5, 0.5]]).bfloat16())
    for sd in (torch.bfloat16, torch.float32):
        obs = {"out": [out, rec(torch.tensor([[1.0]], dtype=sd)), rec(torch.tensor([[0.5]], dtype=sd))]}
        assert C["NORM"].adapter.check_output("aten_native", call, obs)
    wrong = {"out": [out, rec(torch.tensor([[1.0]])), rec(torch.tensor([[0.25]]))]}
    assert not C["NORM"].adapter.check_output("aten_native", call, wrong)
    f16 = {"out": [out, rec(torch.tensor([[1.0]]).half()), rec(torch.tensor([[0.5]]).half())]}
    assert not C["NORM"].adapter.check_output("aten_native", call, f16)


# ------------------------------------------------------------------ EMB, recorded structures
@pytest.mark.parametrize("dtype,V,pad", [("f32", 512, -1), ("bf16", 2, -1), ("f32", 300, 0), ("bf16", 300, 0)])
def test_embedding_backward_recorded_structure(C, dtype, V, pad):
    g = torch.Generator().manual_seed(4)
    grad = torch.randn(1, 128, 16, generator=g).to(TORCH[dtype])
    idx = torch.randint(0, V, (1, 128), generator=g)
    idx[0, :5] = 0
    idx[0, 5:8] = V - 1
    call = {"op": "aten.embedding_backward", "args": [rec(grad), rec(idx), V, pad, False, False]}
    only_aten_profile_answers(C["EMB"], call)
    r = answer(C["EMB"], "aten_dense_backward", call)
    close(r.fields["out"], torch.ops.aten.embedding_backward(grad.double(), idx, V, pad, False, False))
    out = np.asarray(mid(r.fields["out"]))
    assert out.shape == (V, 16)
    if pad == -1:
        assert np.any(out[V - 1] != 0)          # -1 pads no row at the ATen level (EM.ATEN_PADDING)
    else:
        assert np.all(out[0] == 0)
    sparse = dict(call, args=call["args"][:5] + [True])
    assert answer(C["EMB"], "aten_dense_backward", sparse).outcome == "unsupported"


# ------------------------------------------------------------------ NLL, recorded structures
def test_nll_loss_recorded_structure(C):
    g = torch.Generator().manual_seed(5)
    lp = torch.log_softmax(torch.randn(128, 1000, generator=g), 1)
    t = torch.randint(0, 1000, (128,), generator=g)
    t[::7] = -100
    fwd = {"op": "aten.nll_loss_forward", "args": [rec(lp), rec(t), None, 1, -100]}
    only_aten_profile_answers(C["NLL"], fwd)
    r = answer(C["NLL"], "aten_real", fwd)
    o, tw = torch.ops.aten.nll_loss_forward(lp.double(), t, None, 1, -100)
    close(r.fields["out"][0], o)
    assert r.fields["out"][1] == int((t != -100).sum()) == float(tw)
    go = torch.tensor(1.0)
    bwd = {"op": "aten.nll_loss_backward", "args": [rec(go), rec(lp), rec(t), None, 1, -100, rec(tw.float())]}
    only_aten_profile_answers(C["NLL"], bwd)
    rb = answer(C["NLL"], "aten_real", bwd)
    close(rb.fields["out"], torch.ops.aten.nll_loss_backward(go.double(), lp.double(), t, None, 1, -100, tw))


def test_nll_supplied_total_weight_and_empty_mean(C):
    lp = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    t = torch.tensor([0, -100])
    call = {"op": "aten.nll_loss_backward",
            "args": [rec(torch.tensor(1.0)), rec(lp), rec(t), None, 1, -100, rec(torch.tensor(4.0))]}
    ref = answer(C["NLL"], "aten_real", call).fields["out"]
    reading = evaluate(C["NLL"].adapter.READINGS["NL.R1"], "aten_real", call).fields["out"]
    eager = torch.ops.aten.nll_loss_backward(torch.tensor(1.0), lp, t, None, 1, -100, torch.tensor(4.0))
    assert ref[0][0] == Fraction(-1, 4) and reading[0][0] == -1
    close(ref, eager)
    ign = {"op": "aten.nll_loss_forward", "args": [rec(lp), rec(torch.tensor([-100, -100])), None, 1, -100]}
    assert answer(C["NLL"], "aten_real", ign).outcome == "domain_undefined"
    assert math.isnan(torch.ops.aten.nll_loss_forward(lp, torch.tensor([-100, -100]), None, 1, -100)[0].item())
