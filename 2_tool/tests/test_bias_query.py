"""The bias query of the unified entry (composition rules Q1-Q3): the query is fixed before the data (comparison
target, input distribution, observable, sampling unit), the bounded route carries a family-wise guarantee per
declared rule class, and the nonzero and equivalence axes are reported apart, with delta required only when the
equivalence axis is requested."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from kernel_analyzer import measure
from kernel_analyzer.reference_eval.sensitivity import bounded_route

BASE = {"call": "calls.py:f", "inputs": {"x": {"sampler": {"normal": [0, 1]}, "shape": [64], "dtype": "float32"}},
        "compare": {"mode": "A", "measure": ["y"]},
        "budget": {"cpu_seconds": 60, "gpu_seconds": 60, "case_timeout": 30, "max_units": 48}}


def _exp(**over):
    d = copy.deepcopy(BASE)
    d.update(over)
    return measure.expand(d)


# ------------------------------------------------------------------------------------------------ Q1

def test_query_fixes_target_distribution_observable_and_unit():
    e = _exp()
    q = e["query"]
    assert set(q) >= {"comparison_target", "input_distribution", "observable", "sampling_unit", "axes", "families"}
    assert q["comparison_target"]["quantity"].startswith("e_num = K - G")
    assert q["input_distribution"]["inputs"] == BASE["inputs"]
    assert q["input_distribution"]["sampling_seed_sha256"] == e["sampling_seed_sha256"]
    assert q["observable"]["outputs"] == ["y"] and set(q["observable"]["projections"]) == {"R1", "R2", "R3", "R5"}
    assert "seed" in q["sampling_unit"]["unit"]
    assert q["axes"] == ["nonzero"]
    # each of the four items is bound by the declaration digest
    d0 = e["declaration_sha256"]
    assert _exp(compare={"mode": "B", "measure": ["y"], "spec": "calls.py:spec"})["declaration_sha256"] != d0
    assert _exp(inputs={"x": {"sampler": {"uniform": [0, 1]}, "shape": [64], "dtype": "float32"}})[
        "declaration_sha256"] != d0
    assert _exp(compare={"mode": "A", "measure": ["z"]})["declaration_sha256"] != d0
    assert _exp(units={"development": 8, "confirmation": 40})["declaration_sha256"] != d0
    assert "composition_rules" in e and e["composition_rules"]["registry_sha256"]


# ------------------------------------------------------------------------------------------------ Q2

def test_bounded_p_value_matches_the_hoeffding_interval():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(4, 200))
        M = float(rng.uniform(0.5, 3))
        mu = float(rng.uniform(-0.5, 0.5))
        a = np.clip(rng.normal(mu, 0.5, n), -M, M)
        w = rng.uniform(0, 0.05, n)
        l, h = a - w, a + w
        for alpha in (0.01, 0.05, 0.2):
            r = bounded_route(l, h, M, alpha)
            if r["verdict"] in ("ERROR", "NOT_ESTABLISHED"):
                continue
            detected = r["verdict"].startswith("DETECTED")
            # p < alpha exactly when the (1 - alpha) Hoeffding interval excludes 0 (ties have probability zero)
            assert (r["p_value_two_sided"] < alpha) == detected, (r, alpha)
            # independent evaluation of Hoeffding's tail at the observed distance
            lt, ht = np.maximum(l, -M), np.minimum(h, M)
            d = max(float(lt.mean()), -float(ht.mean()), 0.0)
            assert math.isclose(r["p_value_two_sided"], min(1.0, 2 * math.exp(-n * d * d / (2 * M * M))),
                                rel_tol=1e-12)


def test_bounded_route_family_wise_error_under_the_null():
    """null: every rule's projection has mean 0 and |a| <= M; Holm over the bounded p-values within one class keeps
    the family-wise error at or below alpha (simulated, 4000 families of 4 rules)"""
    rng = np.random.default_rng(17)
    alpha, n, M, fams = 0.1, 24, 1.0, 4000
    errors = 0
    for _ in range(fams):
        ps = []
        for k in range(4):
            a = rng.uniform(-M, M, n) if k % 2 else rng.choice([-M, M], n)   # the extreme two-point law too
            ps.append(bounded_route(a, a, M, alpha)["p_value_two_sided"])
        adj = measure._holm(ps)
        errors += any(p <= alpha for p in adj)
    rate = errors / fams
    # Hoeffding is conservative: the rate is far below alpha; the bound is what matters
    assert rate <= alpha, rate


def _rule(name, p_approx, bounded_p=None, verdict="NOT_CONFIRMED", eq=None):
    r = {"rule": name, "verdict": verdict, "n": 40, "p_value_two_sided_conservative": p_approx,
         "per_unit_bounds": [[0.0, 1.0], [0.5, 1.5]]}
    if bounded_p is not None:
        r["bounded"] = {"verdict": "DETECTED_POSITIVE" if bounded_p < 0.05 else "NOT_CONFIRMED",
                        "p_value_two_sided": bounded_p, "interval": [0.1, 0.2], "M": 1.0, "M_basis": "declared"}
    if eq is not None:
        r["equivalence"] = eq
    return r


def test_bounded_route_holm_within_class():
    rec = {"rules": [_rule("R1", 0.5, 0.03), _rule("R5", 0.5, 0.04)]}
    out = measure.class_statistics(rec, {"fixed_mean": ["R1", "R5"]}, 0.05)
    b = out["fixed_mean"]["axes"]["nonzero"]["bounded"]
    # Holm: 0.03 * 2 = 0.06 > 0.05 -> neither survives within the class, although each is below alpha alone
    assert b["rules"]["R1"]["holm_adjusted_p"] == pytest.approx(0.06)
    assert b["rules"]["R1"]["family_verdict"] == "not confirmed" and b["rules"]["R5"]["family_verdict"] == \
        "not confirmed"
    assert "FWER" in b["family"]
    rec = {"rules": [_rule("R1", 0.5, 0.001), _rule("R5", 0.5, 0.04)]}
    b = measure.class_statistics(rec, {"fixed_mean": ["R1", "R5"]}, 0.05)["fixed_mean"]["axes"]["nonzero"]["bounded"]
    assert b["rules"]["R1"]["family_verdict"] == "nonzero (positive)"
    assert b["rules"]["R5"]["family_verdict"] == "nonzero (positive)"       # 0.04 * 1 after the first rejection
    # without M the bounded route says so instead of being silently absent
    b = measure.class_statistics({"rules": [_rule("R1", 0.5)]}, {"fixed_mean": ["R1"]}, 0.05)
    assert b["fixed_mean"]["axes"]["nonzero"]["bounded"]["available"] is False


# ------------------------------------------------------------------------------------------------ Q3

def test_equivalence_axis_requires_delta_only_when_requested():
    assert measure.missing_items(BASE) == []                                  # nonzero only: no delta needed
    d = copy.deepcopy(BASE)
    d["query"] = {"axes": ["nonzero", "equivalence"]}
    assert any(m.startswith("equivalence.rel") for m in measure.missing_items(d))
    d["equivalence"] = {"rel": 0.01, "basis": "declared tolerance"}
    assert measure.missing_items(d) == []
    assert measure.expand(d)["query"]["axes"] == ["nonzero", "equivalence"]
    d2 = copy.deepcopy(BASE)
    d2["equivalence"] = {"rel": 0.01, "basis": "declared"}                    # delta given: the axis is requested
    assert measure.expand(d2)["query"]["axes"] == ["nonzero", "equivalence"]
    d3 = copy.deepcopy(d2)
    d3["query"] = {"axes": ["nonzero"]}                                       # delta given but the axis refused
    assert any("equivalence" in m for m in measure.missing_items(d3))
    d4 = copy.deepcopy(BASE)
    d4["query"] = {"axes": ["equivalence"]}                                   # the nonzero axis is always reported
    assert any("query.axes" in m for m in measure.missing_items(d4))


def test_axes_are_reported_apart():
    eq = {"verdict": "WITHIN_DELTA", "p_value_tost": 0.01, "delta_projection": 0.3}
    rec = {"rules": [_rule("R1", 0.001, verdict="DETECTED_POSITIVE", eq=eq), _rule("R5", 0.7, eq=eq)]}
    out = measure.class_statistics(rec, {"fixed_mean": ["R1", "R5"]}, 0.05, equivalence_rel=0.01)
    axes = out["fixed_mean"]["axes"]
    assert set(axes) == {"nonzero", "equivalence"}
    assert axes["nonzero"]["approximate"]["rules"]["R1"].startswith("nonzero")
    assert axes["equivalence"]["requested"] is True and axes["equivalence"]["delta_rel"] == 0.01
    assert axes["equivalence"]["rules"]["R1"] == "within delta" and axes["equivalence"]["rules"]["R5"] == \
        "within delta"
    # a small certain effect: nonzero and within delta at once, on separate axes
    assert axes["equivalence"]["joint"].startswith("every rule of the class within delta")
    out = measure.class_statistics(rec, {"fixed_mean": ["R1", "R5"]}, 0.05)
    assert out["fixed_mean"]["axes"]["equivalence"] == {"requested": False,
                                                        "statement": "not requested: no equivalence statement"}
