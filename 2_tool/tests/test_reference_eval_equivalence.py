"""The equivalence axis of the decision layer (analysis.equivalence / add_equivalence)."""
import numpy as np

from kernel_analyzer.reference_eval.analysis import _t_approximation, assess_units, equivalence


def test_nonzero_and_equivalence_are_separate_axes():
    rng = np.random.default_rng(0)
    units, d = 96, 64
    ref = 3.0 + rng.normal(size=(units, d))
    small = 1e-3 + 1e-2 * rng.normal(size=(units, d))  # certain but tiny bias
    rec, _ = assess_units("t", small, small, ref, np.ones((units, d), bool), 32, ["R1"], run_detector=False,
                          equivalence_rel=1e-2)
    r = rec["rules"][0]
    assert r["verdict"] == "DETECTED_NEGATIVE"  # R1's direction is -1/sqrt(n)
    assert r["equivalence"]["verdict"] == "WITHIN_DELTA"
    big = 5e-2 + 1e-2 * rng.normal(size=(units, d))
    rec, _ = assess_units("t", big, big, ref, np.ones((units, d), bool), 32, ["R1"], run_detector=False,
                          equivalence_rel=1e-2)
    assert rec["rules"][0]["equivalence"]["verdict"] == "NOT_SHOWN"


def test_equivalence_needs_both_endpoints_inside_the_margin():
    x = 0.9 + 0.01 * np.random.default_rng(2).normal(size=32)
    assert equivalence(x - 0.05, x + 0.05, 1.0, 0.05)["verdict"] == "WITHIN_DELTA"
    assert equivalence(x - 0.05, x + 0.2, 1.0, 0.05)["verdict"] == "NOT_SHOWN"  # E[h] reaches the margin
    assert equivalence(x[:1], x[:1], 1.0, 0.05)["verdict"] == "UNRESOLVED_SAMPLE"


def test_skewed_units_are_flagged_and_get_a_bootstrap_companion():
    rng = np.random.default_rng(1)
    z = rng.lognormal(0.0, 1.0, 64)
    assert _t_approximation(z)["t_approximation"].startswith("suspect")
    assert _t_approximation(rng.normal(size=64))["t_approximation"] == "ok"
    rec = equivalence(z, z, 100.0, 0.05)
    assert "robust" in rec and rec["robust"]["method"].startswith("bootstrap-t")


def test_zero_variance_leaves_both_axes_unresolved():
    """64 all-zero units say nothing about the spread of the population: no equivalence (formerly p = 0), and
    the nonzero test does not detect; the same guard covers a single endpoint without spread."""
    from kernel_analyzer.reference_eval.analysis import _summarize

    z = np.zeros(64)
    rec = equivalence(z, z, 1e-6, 0.05)
    assert rec["verdict"] == "UNRESOLVED_SAMPLE" and "p_value_tost" not in rec
    assert not str(_summarize("c", "R1", z, z, 0.05)["verdict"]).startswith("DETECTED")
    h = np.random.default_rng(3).normal(size=64)
    assert equivalence(z, np.abs(h), 1.0, 0.05)["verdict"] == "UNRESOLVED_SAMPLE"
    units, d = 96, 16
    ref = 3.0 + np.random.default_rng(4).normal(size=(units, d))
    e = np.zeros((units, d))
    rec, _ = assess_units("z", e, e, ref, np.ones((units, d), bool), 32, ["R1", "R2", "R3"], run_detector=False,
                          equivalence_rel=1e-2)
    assert all(r["equivalence"]["verdict"] == "UNRESOLVED_SAMPLE" for r in rec["rules"])
