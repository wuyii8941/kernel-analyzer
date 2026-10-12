"""Round 2, measurement fixes that the tolerance study depends on (brief section 6):

1. carry-over and producer evidence agree: a write that leaves bytes and version counters unchanged but that the
   aligned producer trace sees (``tensor.data``) stops the carry-over of the reference memory (storage_effects M4);
2. precision refinement re-evaluates the reference on the same capture (same inputs, initial state, actual outputs):
   a higher level never re-executes the call, so no curve mixes different executions;
3. the analysis does not change the candidate's precision switches (TF32, float32 matmul precision): it records them;
4. the unmeasured producer trace runs from the measured run's RNG state and leaves the RNG stream as the measured run
   left it;
5. no resolution is reported as met without a numerical enclosure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
X12 = {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}


def _run(tmp_path, call, inputs, **extra):
    from kernel_analyzer import measure
    d = {"call": f"audit_calls.py:{call}", "inputs": inputs, "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 6},
         "units": {"development": 2, "confirmation": 4}, "_base_dir": str(HERE)}
    d.update(extra)
    rep = measure.run(d, out=str(tmp_path / "report.json"))
    lv = rep["levels"][0]
    assert lv["status"] == "ok", lv
    return lv


@CUDA
def test_data_write_between_launches_stops_the_carry_over(tmp_path):
    lv = _run(tmp_path, "inplace_between_launches_data", {"x": X12})
    assert lv["outputs"]["y"]["reference"]["reference_scope"].startswith("kernel-level"), lv["outputs"]["y"]["reference"]
    ext = lv["provenance_seed0"]["external_reentries"]
    assert any(e["launch"] == 1 and "producer trace" in e.get("why", "") for e in ext), lv["provenance_seed0"]


@CUDA
def test_refinement_reevaluates_the_same_capture(tmp_path):
    import audit_state as st
    rows = {"state": "audit_calls.py:three_level_rows", "shape": [4, 128], "dtype": "float32"}
    st.CALLS["row_sum_counted"] = 0
    lv1 = _run(tmp_path, "row_sum_counted", {"x": rows}, resolution={"ulp_fraction": 0.125, "max_level": 1})
    one_level = st.CALLS["row_sum_counted"]
    assert one_level >= 6, one_level          # setup, the measured run and the repeats of each of the 6 units
    st.CALLS["row_sum_counted"] = 0
    lv3 = _run(tmp_path, "row_sum_counted", {"x": rows}, resolution={"ulp_fraction": 0.125, "max_level": 3})
    assert [s["level"] for s in lv3["refinement"]["levels"]] == [1, 2, 3]
    assert st.CALLS["row_sum_counted"] == one_level, (st.CALLS["row_sum_counted"], one_level)
    assert [s["capture"] for s in lv3["refinement"]["levels"]] == ["captured", "reused", "reused"]
    assert lv1["refinement"]["category"] == "backend limit"


@CUDA
def test_analysis_leaves_the_precision_switches_alone(tmp_path):
    import audit_state as st
    old = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("high")          # what PyTorch's own TF32 hint recommends
        st.SWITCHES.clear()
        lv = _run(tmp_path, "switches_seen", {"x": X12})
        assert st.SWITCHES and set(st.SWITCHES) == {(True, "high", old[2])}, set(st.SWITCHES)
        env = lv["precision_environment"]
        assert env["torch.backends.cuda.matmul.allow_tf32"] is True and env["float32_matmul_precision"] == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old[0]
        torch.set_float32_matmul_precision(old[1])


@CUDA
def test_producer_trace_does_not_shift_the_rng_stream():
    import audit_calls as ac
    from kernel_analyzer import check

    class Case(check.Case):
        name, implementation, specification, spec_bound = "rng", "dropout_then_affine", "none", None

        def setup(self):
            pass

        def inputs(self, seed):
            g = torch.Generator(device="cpu").manual_seed(seed)
            return {"x": (torch.rand(1024, generator=g) + 1).cuda()}

        def launch(self, inp):
            return ac.dropout_then_affine(inp)

        def spec(self, inp):
            return None

    ks = {}
    for trace in (False, True):
        torch.manual_seed(1234)
        keep = {}
        check.run(Case(), dev=[0, 1], conf=[2, 3], keep=keep, repeats=1, producer_records=trace)
        ks[trace] = [r["k"] for r in keep["y"]]
    for a, b in zip(ks[False], ks[True]):
        assert np.array_equal(a, b)


def test_set_only_output_is_not_reported_as_resolution_met(monkeypatch):
    from kernel_analyzer import measure

    def fake(exp, level, prec, captures=None):
        return {"status": "ok", "precision_dependent_calls": 3, "seconds": 0.1, "capture": "captured",
                "outputs": {"y": {"status": "evaluated", "dtype": "int32",
                                  "reference": {"resolved_fraction": None, "resolution_met": None,
                                                "unresolved_elements": 0, "width_over_ulp": {},
                                                "set_target_elements": 16}}}}
    monkeypatch.setattr(measure, "_measure_pass", fake)
    r = measure.run_level({"resolution": {"ulp_fraction": 0.125, "max_level": 3}, "budget": {"gpu_seconds": 600}},
                          {})["refinement"]
    assert r["category"] == "intrinsic set width"
    assert r["levels"][0]["resolution_met"] is None
    assert "numerical enclosure is met" not in r["outcome"] and "no numerical enclosure" in r["outcome"]


def test_refinement_is_refused_rather_than_re_executed_when_the_capture_store_overflows(monkeypatch):
    from kernel_analyzer import measure
    calls = []

    def fake(exp, level, prec, captures=None):
        calls.append(prec)
        if captures is not None:
            captures["overflow"] = True        # the store was dropped at level 1
        return {"status": "ok", "precision_dependent_calls": 3, "seconds": 0.1, "capture": "captured",
                "outputs": {"y": {"status": "evaluated", "dtype": "float32",
                                  "reference": {"resolved_fraction": 0.5, "resolution_met": False,
                                                "unresolved_elements": 4, "width_over_ulp": {},
                                                "set_target_elements": 0}}}}
    monkeypatch.setattr(measure, "_measure_pass", fake)
    r = measure.run_level({"resolution": {"ulp_fraction": 0.125, "max_level": 3}, "budget": {"gpu_seconds": 600}},
                          {})["refinement"]
    assert calls == [1] and r["category"] == "budget exhausted" and "capture store" in r["outcome"], (calls, r)
