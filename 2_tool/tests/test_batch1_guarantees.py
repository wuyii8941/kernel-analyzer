"""Batch 1 guarantees through the unified entry (``kernel_analyzer.measure``):

1. memory-effect rules (``reference_eval.storage_effects`` M1-M5): the three audit classes -- a cross-input digest
   match, in-place arithmetic that leaves every byte unchanged (on an input, through ``.data``, between recorded
   launches), a partial write through an overlapping full-size view -- keep the reference kernel-level; normal
   copies, exact constants, legal aliases and announced compiled writes keep it call-level;
2. the precision controller: the three-level summation counterexample reaches level 3 from the unified entry, with
   the enclosures checked against exact rational row sums, and the stop categories are distinct;
3. packaging: the package runs from a copy outside the repository without touching a repository path.
"""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
X12 = {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}


def _run(tmp_path, call, inputs, **extra):
    from kernel_analyzer import measure
    d = {"call": f"audit_calls.py:{call}", "inputs": inputs, "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 6},
         "units": {"development": 2, "confirmation": 4}, "_base_dir": str(HERE)}
    d.update(extra)
    rep = measure.run(d, out=str(tmp_path / "report.json"))
    json.loads((tmp_path / "report.json").read_text())
    assert len(rep["levels"]) == 1
    lv = rep["levels"][0]
    assert lv["status"] == "ok", lv
    return lv


def _scope(lv):
    return lv["outputs"]["y"]["reference"]["reference_scope"]


def _reads(lv):
    return (lv.get("provenance_seed0") or {}).get("reads") or []


# ------------------------------------------------------------------------------------------------ 1. memory effects

@CUDA
def test_cross_input_digest_match_is_not_a_declared_input(tmp_path):
    lv = _run(tmp_path, "cross_input_rounded", {"x": X12, "w": X12})
    assert _scope(lv).startswith("kernel-level"), _scope(lv)
    why = [r for r in _reads(lv) if r["outcome"] == "upstream"]
    assert why and any("own bytes" in r["reason"] for r in why), _reads(lv)


@CUDA
def test_inplace_arithmetic_with_unchanged_bytes_on_an_input(tmp_path):
    lv = _run(tmp_path, "inplace_unchanged", {"x": X12})
    assert _scope(lv).startswith("kernel-level"), _scope(lv)
    assert any("version increments" in (r.get("reason") or "") for r in _reads(lv)), _reads(lv)


@CUDA
def test_inplace_arithmetic_through_data_is_seen_by_the_producer_trace(tmp_path):
    lv = _run(tmp_path, "inplace_unchanged_data", {"x": X12})
    assert _scope(lv).startswith("kernel-level"), _scope(lv)
    assert lv["producer_trace_per_seed"].get("aligned", 0) >= 1, lv["producer_trace_per_seed"]
    assert any("producer trace" in (r.get("reason") or "") for r in _reads(lv)), _reads(lv)


@CUDA
def test_inplace_arithmetic_between_recorded_launches(tmp_path):
    lv = _run(tmp_path, "inplace_between_launches", {"x": X12})
    assert _scope(lv).startswith("kernel-level"), _scope(lv)
    ext = lv["provenance_seed0"]["external_reentries"]
    assert any(e["launch"] == 1 and "ATen-visible write" in e.get("why", "") for e in ext), ext


@CUDA
def test_partial_write_through_an_overlapping_full_size_view(tmp_path):
    lv = _run(tmp_path, "overlap_fill", {"x": X12})
    assert _scope(lv).startswith("kernel-level"), _scope(lv)


@CUDA
@pytest.mark.parametrize("call", ["partial_init_read", "zeros_then_partial_copy", "honest_copy"])
def test_copies_and_exact_constants_keep_the_call_level_reference(tmp_path, call):
    lv = _run(tmp_path, call, {"x": X12})
    assert _scope(lv).startswith("call-level"), _scope(lv)


@CUDA
@pytest.mark.parametrize("call", ["inplace_after_read", "input_offset_view", "classic"])
def test_declared_inputs_and_legal_aliases_stay_declared(tmp_path, call):
    lv = _run(tmp_path, call, {"x": X12})
    assert _scope(lv) == "call-level", _scope(lv)
    assert all(r["outcome"] == "declared input" for r in _reads(lv)), _reads(lv)


@CUDA
def test_announced_compiled_inplace_write_keeps_the_composition(tmp_path):
    """AOTAutograd bumps the input's version before the compiled kernel writes it: an announced raw-pointer write,
    not an ATen write (the producer trace cannot be aligned for compiled code; the version evidence decides)"""
    lv = _run(tmp_path, "compiled_inplace_then_kernel", {"x": X12})
    assert _scope(lv) == "call-level", (_scope(lv), lv.get("provenance_seed0"))
    # finding D12: the traced run must not switch later calls of the compiled function to eager execution -- every
    # seed records the compiled launch, and the trace of compiled code cannot be aligned (it runs eagerly)
    assert lv["notes"]["ir_kinds"] == ["ttir", "ttir"]
    assert lv["producer_trace_per_seed"] == {"not aligned": 6}, lv["producer_trace_per_seed"]
    carried = lv["provenance_seed0"]["carry_overs"]
    assert any(c["launch"] == 1 and c["evidence"] == "bytes and version counters" for c in carried), carried


# ------------------------------------------------------------------------------------------------ 2. precision

ROWS = {"state": "audit_calls.py:three_level_rows", "shape": [4, 128], "dtype": "float32"}


@CUDA
def test_three_level_summation_reaches_level_3_from_the_unified_entry(tmp_path):
    lv = _run(tmp_path, "row_sum", {"x": ROWS}, resolution={"ulp_fraction": 0.125, "max_level": 3})
    ref = lv["refinement"]
    steps = ref["levels"]
    assert [s["level"] for s in steps] == [1, 2, 3], ref
    assert steps[0]["resolved_fraction"]["y"] == 0.0 and steps[1]["resolved_fraction"]["y"] == 0.0, steps
    assert steps[2]["resolved_fraction"]["y"] == 1.0, steps
    # the widths shrink by orders of magnitude while the pass fraction stays 0
    assert steps[1]["width_over_ulp"]["y"]["median"] < 1e-20 * steps[0]["width_over_ulp"]["y"]["median"], steps
    assert ref["category"] == "met" and ref["outcome"] == "met at level 3", ref
    assert lv["outputs"]["y"]["reference"]["resolution_met"] is True


@CUDA
def test_three_level_summation_enclosures_against_exact_row_sums():
    """independent answer: the exact rational row sums lie in every level's enclosure; only level 3 is within 1/8
    ulp"""
    import audit_calls as ac
    from kernel_analyzer import check, measure

    x = ac.three_level_rows(5, (4, 128), "float32")
    exact = [sum((Fr(float(v)) for v in row), Fr(0)) for row in x.double().cpu().numpy()]

    class Case(check.Case):
        name, implementation, specification, spec_bound = "three_level", "row_sum", "none", None

        def setup(self):
            pass

        def inputs(self, seed):
            return {"x": x.clone()}

        def launch(self, inp):
            return ac.row_sum(inp)

        def spec(self, inp):
            return None

    widths = {}
    for level in (1, 2, 3):
        keep = {}
        check.run(Case(), dev=[0, 1], conf=[2], keep=keep, precision_level=level, repeats=1)
        row = keep["y"][0]
        lo, hi = row["r_lo"], row["r_hi"]
        assert all(Fr(float(lo[i])) <= exact[i] <= Fr(float(hi[i])) for i in range(4)), level
        ulp = measure.ulp_of(0.5 * (lo + hi), "float32")
        widths[level] = (hi - lo) / ulp
    assert (widths[1] > 0.125).all() and (widths[2] > 0.125).all() and (widths[3] <= 0.125).all(), widths


def _fake_pass(results):
    """A _measure_pass stand-in returning one synthetic pass per level."""
    it = iter(results)

    def fake(exp, level, prec):
        r = next(it)
        if "status" in r:
            return r
        outs = {}
        for name, (frac, dtype, nset) in r["outputs"].items():
            met = None if frac is None else frac == 1.0
            outs[name] = {"status": "evaluated", "dtype": dtype,
                          "reference": {"resolved_fraction": frac, "resolution_met": met, "unresolved_elements": 0,
                                        "width_over_ulp": {}, "set_target_elements": nset}}
        return {"status": "ok", "outputs": outs, "precision_dependent_calls": r.get("calls", 5), "seconds": 0.1}
    return fake


def _exp(max_level=3, gpu_seconds=600):
    return {"resolution": {"ulp_fraction": 0.125, "max_level": max_level}, "budget": {"gpu_seconds": gpu_seconds}}


def test_controller_does_not_stop_on_an_unchanged_pass_fraction(monkeypatch):
    from kernel_analyzer import measure
    monkeypatch.setattr(measure, "_measure_pass", _fake_pass([
        {"outputs": {"y": (0.0, "float32", 0)}}, {"outputs": {"y": (0.0, "float32", 0)}},
        {"outputs": {"y": (1.0, "float32", 0)}}]))
    r = measure.run_level(_exp(), {})["refinement"]
    assert r["category"] == "met" and r["final_level"] == 3, r


@pytest.mark.parametrize("passes, max_level, category, prefix", [
    ([{"outputs": {"y": (0.5, "float32", 0)}}] * 3, 3, "backend limit", "backend limit: highest"),
    ([{"outputs": {"y": (0.5, "float32", 0)}, "calls": 0}], 3, "backend limit", "backend limit: no rule"),
    ([{"outputs": {"y": (0.5, "float64", 0)}}], 3, "backend limit", "backend limit: float64"),
    ([{"outputs": {"y": (None, "int32", 12)}}], 3, "intrinsic set width", "intrinsic set width"),
    ([{"outputs": {"y": (1.0, "int32", 12)}}], 3, "intrinsic set width", "intrinsic set width"),
    ([{"outputs": {"y": (1.0, "float32", 0), "z": (None, "float32", 0)}}], 3, "no numerical enclosure",
     "no numerical enclosure: z"),
    ([{"outputs": {"y": (1.0, "float32", 0), "z": (1.0, "float32", 0)}}], 3, "met", "met at level 1"),
    ([{"outputs": {"y": (0.5, "float32", 0)}}, {"status": "over budget", "reason": "case timeout"}], 3,
     "budget exhausted", "budget exhausted: the pass at level 2"),
    ([{"outputs": {"y": (0.5, "float32", 0)}}, {"status": "error", "reason": "boom"}], 3, "pass failed",
     "pass failed"),
])
def test_controller_stop_categories(monkeypatch, passes, max_level, category, prefix):
    from kernel_analyzer import measure
    monkeypatch.setattr(measure, "_measure_pass", _fake_pass(passes))
    r = measure.run_level(_exp(max_level), {})["refinement"]
    assert r["category"] == category and r["outcome"].startswith(prefix), r
    assert r["category_meaning"] == measure.REFINEMENT_CATEGORIES[category]


def test_controller_budget_between_levels(monkeypatch):
    from kernel_analyzer import measure
    monkeypatch.setattr(measure, "_measure_pass", _fake_pass([{"outputs": {"y": (0.5, "float32", 0)}}]))
    r = measure.run_level(_exp(gpu_seconds=-1), {})["refinement"]
    assert r["category"] == "budget exhausted" and r["final_level"] == 1, r


# ------------------------------------------------------------------------------------------------ 3. packaging

AUDIT = r"""
import sys
REPO = sys.argv[1]
seen = []
def hook(event, args):
    if event in ("open", "import") and args and isinstance(args[0], str) and args[0].startswith(REPO):
        seen.append((event, args[0]))
sys.addaudithook(hook)
import kernel_analyzer, kernel_analyzer.measure as m, kernel_analyzer.check, kernel_analyzer.provenance
rec = {"rules": [{"rule": "R1", "verdict": "NOT_CONFIRMED", "n": 40, "p_value_two_sided_conservative": 0.5,
                  "per_unit_bounds": [0.0, 1.0]}]}
out = m.class_statistics(rec, {"fixed_mean": ["R1"]}, 0.05)
assert out["fixed_mean"]["rules"]["R1"]["judgment"] == "not confirmed", out
import kernel_analyzer.cli
print(kernel_analyzer.__file__)
print("SEEN", seen)
"""


def test_package_runs_from_a_copy_outside_the_repository(tmp_path):
    """the package does not import or open a repository path at run time (contract_v3 used to come from
    scripts/essential through a path relative to the source tree)"""
    import shutil
    site = tmp_path / "site"
    shutil.copytree(SRC / "kernel_analyzer", site / "kernel_analyzer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    repo_tool = str(HERE.parent)                       # 2_tool: sources, scripts, tests
    script = tmp_path / "audit.py"
    script.write_text(AUDIT)
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONPATH",)}
    out = subprocess.run([sys.executable, "-I", "-c",
                          f"import sys; sys.path.insert(0, {str(site)!r}); exec(open({str(script)!r}).read())",
                          repo_tool], capture_output=True, text=True, cwd=tmp_path, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.strip().splitlines()
    assert lines[-2].startswith(str(site)), lines
    assert lines[-1] == "SEEN []", lines[-1]
