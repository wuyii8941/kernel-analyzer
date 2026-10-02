"""Direction rules of the unified entry on synthetic unit residuals (CPU)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from kernel_analyzer.reference_eval.analysis import _summarize, apply_direction_rules  # noqa: E402


def _intervals(rng, n, d, mu=0.0, v=None, width=0.0):
    v = np.eye(d)[0] if v is None else v
    mid = rng.standard_normal((n, d)) + mu * v
    return mid - width, mid + width


def _old_rules(name, lows, highs, ref, n_cal, n_conf, alpha):
    """The loop implementation the rules replaced (fixed direction and aligned)."""

    out = []
    direction = (0.5 * (lows[:n_cal] + highs[:n_cal])).mean(axis=0)
    w = direction / np.linalg.norm(direction)
    proj = np.array([(np.minimum(lows[i] * w, highs[i] * w).sum(), np.maximum(lows[i] * w, highs[i] * w).sum())
                     for i in range(n_cal, n_cal + n_conf)])
    out.append(_summarize(name, "fixed_direction", proj[:, 0], proj[:, 1], alpha))
    proj = []
    for i in range(n_cal, n_cal + n_conf):
        r = ref[i]
        w = r / np.linalg.norm(r)
        proj.append((np.minimum(lows[i] * w, highs[i] * w).sum(), np.maximum(lows[i] * w, highs[i] * w).sum()))
    proj = np.array(proj)
    out.append(_summarize(name, "aligned_reference_update", proj[:, 0], proj[:, 1], alpha))
    return out


def test_fixed_and_aligned_rules_match_previous_implementation():
    rng = np.random.default_rng(0)
    lows, highs = _intervals(rng, 96, 40, mu=0.3, width=0.01)
    ref = rng.standard_normal((96, 40))
    decl = {"direction_rules": ["fixed_direction", "aligned_reference_update"]}
    new = apply_direction_rules("x", lows, highs, ref, decl, 32, 64, 0.05)
    old = _old_rules("x", lows, highs, ref, 32, 64, 0.05)
    for a, b in zip(new, old):
        assert a["verdict"] == b["verdict"]
        assert a["lower_bound_of_E_l"] == pytest.approx(b["lower_bound_of_E_l"], rel=1e-12, abs=1e-15)
        assert a["p_value_two_sided_conservative"] == pytest.approx(b["p_value_two_sided_conservative"], rel=1e-9)


def test_cross_fit_uses_all_units_and_adjusts_for_folds():
    rng = np.random.default_rng(1)
    lows, highs = _intervals(rng, 96, 8, mu=1.0)
    decl = {"direction_rules": ["cross_fit"], "cross_fit_folds": 3}
    (r,) = apply_direction_rules("x", lows, highs, np.zeros((96, 8)), decl, 32, 64, 0.05)
    assert r["n"] == 96 and r["folds"] == 3 and len(r["fold_results"]) == 3
    assert sum(f["n"] for f in r["fold_results"]) == 96
    best = min(f["p_value_two_sided_conservative"] for f in r["fold_results"])
    assert r["p_value_two_sided_conservative"] == pytest.approx(min(1.0, 3 * best))
    assert r["verdict"] == "DETECTED_POSITIVE"
    zero = np.zeros((96, 8))
    (z,) = apply_direction_rules("x", zero, zero, zero, decl, 32, 64, 0.05)
    assert z["verdict"] == "UNRESOLVED_MEASUREMENT"


def test_grouped_rule_sums_declared_groups():
    rng = np.random.default_rng(2)
    d, size = 1024, 64
    v = np.full(d, 1 / np.sqrt(d))  # effect constant within groups
    lows, highs = _intervals(rng, 96, d, mu=0.6, v=v, width=0.001)
    decl = {"direction_rules": ["fixed_direction", "grouped_fixed_direction"], "groups": {"size": size}}
    raw, grouped = apply_direction_rules("x", lows, highs, np.zeros((96, d)), decl, 32, 64, 0.05)
    assert grouped["verdict"] == "DETECTED_POSITIVE"
    assert raw["verdict"] == "NOT_CONFIRMED"
    with pytest.raises(ValueError):
        apply_direction_rules("x", lows[:, :1000], highs[:, :1000], np.zeros((96, 1000)), decl, 32, 64, 0.05)
