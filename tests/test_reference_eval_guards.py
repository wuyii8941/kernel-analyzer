"""Entry guards (CPU): directed residual construction, too few units and numerical failure never detect."""

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("scipy")

from kernel_analyzer.reference_eval.analysis import (  # noqa: E402
    _summarize, apply_direction_rules, apply_holm, assess_units, residual_interval)
from kernel_analyzer.reference_eval.detect import detect  # noqa: E402


def test_residual_interval_encloses_what_plain_subtraction_rounds_away():
    cand = np.array([1e-30, 3.0, -2.0 ** -1074])
    ref_lo = np.array([1e10, 3.0 - 2.0 ** -51, 1.0])
    ref_hi = np.array([1e10, 3.0 + 2.0 ** -51, 1.0])
    lo, hi = residual_interval(cand, ref_lo, ref_hi)
    for c, a, b, l, h in zip(cand, ref_lo, ref_hi, lo, hi):
        exact_lo = Fraction(float(c)) - Fraction(float(b))
        exact_hi = Fraction(float(c)) - Fraction(float(a))
        assert Fraction(float(l)) <= exact_lo and exact_hi <= Fraction(float(h))
    # the counterexample: the plain upper endpoint 1e-30 - 1e10 rounds to -1e10, below the exact residual
    assert Fraction(float(cand[0] - ref_lo[0])) < Fraction(1e-30) - Fraction(1e10)


def test_summary_cannot_judge_with_one_unit_or_numerical_failure():
    one = _summarize("x", "R1", np.array([5.0]), np.array([5.0]), 0.05)
    assert one["verdict"] == "UNRESOLVED_SAMPLE" and "p_value_two_sided_conservative" not in one
    nan = _summarize("x", "R1", np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]), 0.05)
    assert nan["verdict"] == "UNRESOLVED_NUMERICAL" and "p_value_two_sided_conservative" not in nan
    big = np.array([1e200, 3e200, 5e200, 7e200])  # the squared deviations overflow
    over = _summarize("x", "R1", big, big, 0.05)
    assert over["verdict"] == "UNRESOLVED_NUMERICAL"
    flat = _summarize("x", "R1", np.full(10, 2.0), np.full(10, 2.0) + np.arange(10), 0.05)
    assert flat["verdict"] == "UNRESOLVED_SAMPLE"  # zero spread of the endpoint that would carry the detection
    zero = _summarize("x", "R1", np.zeros(10), np.zeros(10), 0.05)
    assert zero["verdict"] == "NOT_CONFIRMED" and zero["p_value_two_sided_conservative"] == 1.0


def test_rules_with_one_confirmation_unit_stay_out_of_holm():
    rng = np.random.default_rng(0)
    e = rng.standard_normal((5, 16)) + 10.0  # a huge mean, but a single confirmation unit
    res = apply_direction_rules("x", e, e, e, {"direction_rules": ["R1", "R3", "R5"]}, 4, 1, 0.05)
    assert all(r["verdict"] == "UNRESOLVED_SAMPLE" for r in res)
    tested = [r for r in res if "p_value_two_sided_conservative" in r]
    apply_holm(tested, 0.05)
    assert not tested


@pytest.mark.parametrize("case", ["one_unit", "one_confirmation_unit", "nan", "overflow"])
def test_detector_never_detects_on_too_few_units_or_numerical_failure(case):
    rng = np.random.default_rng(1)
    if case == "one_unit":
        e, n_dev = rng.standard_normal((1, 32)) + 5.0, 0
    elif case == "one_confirmation_unit":
        e, n_dev = rng.standard_normal((9, 32)) + 5.0, 8
    elif case == "nan":
        e, n_dev = rng.standard_normal((40, 32)) + 5.0, 20
        e[3, 7] = np.nan
    else:
        e, n_dev = rng.standard_normal((40, 32)) * 1e300, 20
    k = rng.standard_normal(e.shape)
    r = detect(e, e, k, n_dev, seed=2)
    tests = r["vector_mean"]["tests"] + r["alignment"]["tests"]
    if case != "one_confirmation_unit":  # there the omnibus / total tests use all 9 units and may detect
        assert "DETECTED" not in (r["vector_mean"]["verdict"], r["alignment"]["verdict"])
    if case in ("one_unit", "nan"):
        assert r["vector_mean"]["verdict"] == "CANNOT_JUDGE" and r["alignment"]["verdict"] == "CANNOT_JUDGE"
    elif case == "overflow":  # every computation that overflowed cannot judge; a finite one may still test
        assert r["vector_mean"]["tests"][0]["verdict"] == "CANNOT_JUDGE"
        assert all(t["verdict"] in ("CANNOT_JUDGE", "NOT_CONFIRMED") for t in tests)
    else:
        learned = next(t for t in r["vector_mean"]["tests"] if t["test"] == "learned")
        assert learned["verdict"] == "CANNOT_JUDGE" and learned["p"] is None
    for t in tests:
        if t["verdict"] == "CANNOT_JUDGE":
            assert t["p"] is None and "holm_reject" not in t


def test_assess_units_with_one_confirmation_unit_reports_no_detection():
    rng = np.random.default_rng(3)
    e = rng.standard_normal((9, 24)) + 3.0
    ok = np.ones_like(e, dtype=bool)
    record, _ = assess_units("x", e, e, rng.standard_normal(e.shape), ok, 8, ["R1", "R2", "R3", "R5"])
    assert all(r["verdict"] == "UNRESOLVED_SAMPLE" for r in record["rules"])
    learned = next(t for t in record["default_detector"]["vector_mean"]["tests"] if t["test"] == "learned")
    assert learned["verdict"] == "CANNOT_JUDGE"
