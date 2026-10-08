"""Unit tests of the numeric contract v3 layer (scripts/essential/contract_v3.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract_v3 as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from kernel_analyzer.reference_eval import analysis as A  # noqa: E402


def test_zero_variance_is_degenerate_not_p0():
    rec = A._summarize("c", "R1", np.full(32, 0.5), np.full(32, 0.5), 0.05)
    assert C.statistical_judgment(rec)["judgment"] == "cannot judge (degenerate)"


def test_small_sample_cannot_judge():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.1, 8)
    rec = A._summarize("c", "R1", x, x, 0.05)
    assert rec["verdict"] == "DETECTED_POSITIVE"
    assert C.statistical_judgment(rec)["judgment"] == "cannot judge (sample)"


def test_strong_skew_small_n_cannot_judge_distribution():
    rng = np.random.default_rng(1)
    x = rng.lognormal(0, 1.2, 32) - 1.0
    rec = A._summarize("c", "R1", x, x, 0.05)
    assert abs(rec["unit_skewness"]) > C.S0
    assert C.statistical_judgment(rec)["judgment"] == "cannot judge (distribution)"


def test_strong_skew_at_n64_uses_bootstrap_companion():
    rng = np.random.default_rng(3)
    x = rng.lognormal(0, 1.2, 64) + 0.5
    rec = A._summarize("c", "R1", x, x, 0.05)
    assert abs(rec["unit_skewness"]) > C.S0
    j = C.statistical_judgment(rec)
    assert j["judgment"] == "nonzero (positive)" and "bootstrap" in j["basis"]


def test_regular_case_keeps_frozen_verdict():
    rng = np.random.default_rng(2)
    x = rng.normal(0.3, 1.0, 64)
    rec = A._summarize("c", "R1", x, x, 0.05)
    assert C.statistical_judgment(rec)["judgment"] in ("nonzero (positive)", "not confirmed")


def test_precision_invariance_classes():
    f = np.array([1.0, 2.0, 3.0, 4.0])
    k64 = f + np.array([1e-16, 0.0, 0.25, 1e-9])
    k32 = f + np.array([1e-7, 0.0, 0.25, 1e-7])
    r = C.precision_invariance(k32, k64, f)
    assert r["numerical"] == 1 and r["agree"] == 1 and r["semantic"] == 1 and r["undecided"] == 1
    assert r["condition"] == "semantic deviation present"


def test_precision_invariance_nonfinite_goes_to_column4():
    r = C.precision_invariance([np.nan, 1.0], [1.0, 1.0], [1.0, 1.0])
    assert r["nonfinite_f32_only"] == 1 and r["column4_nonfinite"]
