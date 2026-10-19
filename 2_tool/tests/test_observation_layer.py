"""The increment observation layer (``reference_eval.observation``; rc3 02 1.2 "formula update"): the written
increment of a state or parameter against the reference increment on the same old value, through mode A (Triton,
automatic reference) and the black-box mode (ATen, the specification), with independent exact answers."""
from __future__ import annotations

from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest
import torch

from kernel_analyzer.reference_eval.observation import floor_of, increment_unit, summarize_increment

HERE = Path(__file__).resolve().parent
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_increment_unit_matches_a_per_coordinate_count():
    rng = np.random.default_rng(4)
    old = rng.uniform(0.5, 1.5, 4000)
    ref_inc = rng.normal(0, 1e-3, 4000)
    ref_inc[:40] = 0.0                                  # reference increment exactly zero
    k = old + ref_inc
    k[100:400] = old[100:400]                           # dropped
    k[400:450] = old[400:450] - ref_inc[400:450]        # flipped
    ok = np.ones(4000, bool)
    ok[3990:] = False
    r = old + ref_inc
    u = increment_unit(k, r, r, old, ok)
    lo, hi = np.nextafter(r - old, -np.inf), np.nextafter(r - old, np.inf)
    above = ok & ((lo > 0) | (hi < 0))
    assert u["above_floor"] == int(above.sum())
    assert u["dropped"] == int((above & (k == old)).sum()) == 300
    assert u["sign_flips"] == int((above & (k != old) & (np.sign(k - old) != np.sign(ref_inc))).sum()) == 50
    m = 0.5 * (lo + hi)
    assert u["retention"] == pytest.approx(np.dot((k - old)[above], m[above]) / np.dot(m[above], m[above]))
    # a floor moves coordinates out of the ratios, never into "dropped"
    f = floor_of({"absolute": 5e-4}, old)
    uf = increment_unit(k, r, r, old, ok, f)
    assert uf["above_floor"] < u["above_floor"] and uf["near_zero_or_below_floor"] > u["near_zero_or_below_floor"]
    s = summarize_increment([u, uf])
    assert s["units"] == 2 and s["dropped"] == u["dropped"] + uf["dropped"]


def _expected_bf16_drops(v_bf16, g_bf16):
    """Independent answer: coordinates where the exact update beta v + (1 - beta) g^2 differs from v but rounding it
    to bf16 (nearest even, from the float32 value the kernel computes) gives v back."""
    beta = Fr(np.float32(0.999).item())         # the kernel's literal is a float32 constant
    one_minus = Fr(np.float32(1.0 - np.float32(0.999)).item())
    v = v_bf16.float().cpu().numpy().astype(np.float64)
    g = g_bf16.float().cpu().numpy().astype(np.float64)
    exact_new = [Fr(float(a)) * beta + one_minus * Fr(float(b)) * Fr(float(b)) for a, b in zip(v, g)]
    f32 = torch.tensor([float(x) for x in exact_new], dtype=torch.float32)
    written = f32.to(torch.bfloat16).double().numpy()
    nz = np.array([x != Fr(float(a)) for x, a in zip(exact_new, v)])
    return int((nz & (written == v)).sum()), int(nz.sum())


def _decl(call, mode="A", dtype="bfloat16", **extra):
    d = {"call": f"audit_calls.py:{call}",
         "inputs": {"v": {"state": "audit_calls.py:unit_state", "shape": [2048], "dtype": dtype},
                    "g": {"state": "audit_calls.py:small_grads", "shape": [2048], "dtype": dtype}},
         "compare": {"mode": mode, "measure": ["v"],
                     "observe": {"v": {"layer": "increment", "old": "v"}}},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 18},
         "units": {"development": 2, "confirmation": 16}, "_base_dir": str(HERE)}
    d.update(extra)
    return d


@CUDA
def test_dropped_state_increments_through_mode_a_match_the_exact_count():
    import audit_calls as ac
    from kernel_analyzer import measure
    lv = measure.run(_decl("ema_state_inplace"))["levels"][0]
    assert lv["status"] == "ok", lv
    ob = lv["outputs"]["v"]["observation"]
    unit0 = ob["per_unit"][0]
    want, nz = _expected_bf16_drops(ac.unit_state(unit0["seed"], (2048,), "bfloat16"),
                                    ac.small_grads(unit0["seed"], (2048,), "bfloat16"))
    assert unit0["dropped"] == want and unit0["reference_increment_nonzero"] == nz, (unit0, want, nz)
    assert ob["summary"]["dropped_fraction"]["mean"] > 0.5            # decay of 1e-3 is below half a bf16 ulp
    assert ob["summary"]["retention"]["mean"] < 0.5
    # the decision layer sees the deficit along the reference increment
    r3 = lv["outputs"]["v"]["statistics"]["aligned"]["rules"]["R3"]["judgment"]
    assert r3.startswith("nonzero (positive)"), r3


@CUDA
def test_float32_state_keeps_its_increments():
    from kernel_analyzer import measure
    lv = measure.run(_decl("ema_state_inplace", dtype="float32"))["levels"][0]
    ob = lv["outputs"]["v"]["observation"]["summary"]
    assert ob["dropped"] == 0 and ob["retention"]["min"] > 0.99, ob


@CUDA
def test_black_box_state_update_reports_the_same_layer():
    from kernel_analyzer import measure
    d = _decl("ema_state_aten", mode="black-box")
    d["compare"]["spec"] = "audit_calls.py:ema_state_spec"
    rep = measure.run(d)
    lv = rep["levels"][0]
    assert lv["status"] == "ok" and lv["mode"] == "black-box", lv
    out = lv["outputs"]["v"]
    assert out["status"] == "evaluated (black box)"
    assert out["observation"]["summary"]["dropped_fraction"]["mean"] > 0.5
    assert lv["refinement"]["category"] == "not applicable"
    assert rep["expanded_declaration"]["query"]["comparison_target"]["quantity"].startswith(
        "K - supplied reference only")


def test_observe_declaration_is_validated():
    from kernel_analyzer import measure
    d = _decl("ema_state_inplace")
    d["compare"]["observe"] = {"v": {"layer": "increment", "old": "nope"}}
    assert any(m.startswith("compare.observe.v.old") for m in measure.missing_items(d))
    d["compare"]["observe"] = {"w": {"layer": "increment", "old": "v"}}
    assert any("not a measured output" in m for m in measure.missing_items(d))
    d = _decl("ema_state_aten", mode="black-box")
    assert any(m.startswith("compare.spec") for m in measure.missing_items(d))

    d = _decl("ema_state_inplace")
    d["compare"] = {"mode": "contract", "measure": ["v"]}
    missing = measure.missing_items(d)
    assert any(m.startswith("compare.binding_plan") for m in missing)
    assert any(m.startswith("compare.specification_root") for m in missing)


def test_contract_mode_rejects_a_draft_binding_before_running_the_call():
    from kernel_analyzer import measure

    d = _decl("ema_state_inplace")
    root = HERE.parents[1] / "5_specification"
    d["compare"] = {
        "mode": "contract", "measure": ["v"], "specification_root": str(root),
        "binding_plan": str(root / "corpus/call_bindings/liger_cross_entropy.json"),
    }
    with pytest.raises(measure.ReferenceNotAdmitted, match="reviewed binding plan"):
        measure.run(d)


def test_black_box_spec_receives_the_pre_call_inputs():
    """An in-place implementation must not silently turn its post-state into the task reference's input."""
    from kernel_analyzer import check

    seen = []

    class Case(check.Case):
        name = "pre-state-for-spec"
        implementation = "in-place test call"
        specification = "point reference on the received pre-state"

        def inputs(self, seed):
            return {"x": torch.tensor([float(seed + 1)])}

        def launch(self, inp):
            inp["x"].add_(10)
            return {"x": inp["x"]}

        def spec(self, inp):
            value = inp["x"].detach().double().numpy()
            seen.append(float(value[0]))
            return {"x": (value, value)}

    report = check.run_black_box(Case(), dev=[0, 1], conf=[2, 3])
    assert report["outputs"]["x"]
    assert seen == [1.0, 2.0, 3.0, 4.0]


@CUDA
def test_mode_b_decomposes_the_increment_frame():
    """Mode B keeps the numerical decomposition but does not certify an arbitrary callback as task truth."""
    from kernel_analyzer import measure
    d = _decl("ema_state_inplace", mode="B")
    d["compare"]["spec"] = "audit_calls.py:ema_state_spec"
    lv = measure.run(d)["levels"][0]
    assert lv["reference_kind"] == "implementation_reference"
    assert lv["semantic_verdict"] == "not_established"
    assert lv["outputs"]["v"]["semantic"]["verdict_scope"] == \
        "reference_difference; not a semantic verdict"
    dec = lv["outputs"]["v"]["observation"]["decomposition"]
    num = lv["outputs"]["v"]["observation"]["summary"]
    # the kernel's float32 constant 0.999 differs from the spec's exact 0.999: a small semantic part; the dropped
    # decrements are numerical (rounding to bf16) and appear in the total too
    assert num["dropped_fraction"]["mean"] > 0.5 and dec["total"]["dropped_fraction"]["mean"] > 0.5
    assert dec["semantic"]["dropped"] == 0 and abs(dec["semantic"]["retention"]["mean"] - 1) < 1e-3, dec["semantic"]


def test_output_unit_separates_uniform_scaling_from_the_rest_of_the_error():
    from kernel_analyzer.reference_eval.observation import output_unit
    rng = np.random.default_rng(7)
    r = rng.normal(size=(64, 32))
    s = output_unit((1 - 3.5e-4) * r, r)                      # a pure shrink
    assert s["alignment"] == pytest.approx(-3.5e-4, rel=1e-9)
    assert s["scale_ratio"] == pytest.approx(1 - 3.5e-4, rel=1e-12)
    assert s["scaling_share"] == pytest.approx(1.0, rel=1e-9) and s["cosine"] == pytest.approx(1.0)
    noise = rng.normal(size=r.shape)
    noise -= (noise * r).sum() / (r * r).sum() * r            # orthogonal to r
    o = output_unit(r + 1e-3 * noise, r)
    assert abs(o["alignment"]) < 1e-15 and o["scaling_share"] < 1e-20
    assert o["relative_rms"] == pytest.approx(1e-3 * np.sqrt((noise ** 2).mean() / (r ** 2).mean()), rel=1e-9)
    assert output_unit(r, r)["scaling_share"] is None          # no error: no share
