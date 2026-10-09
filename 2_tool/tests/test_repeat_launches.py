"""Repeated launches (DSL v2 rc3 02 8.7) in check.run: the per-output execution status and the within-input mean.

CPU only, on synthetic per-unit rows of the shape check.run builds."""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np

from kernel_analyzer.check import _execution_status, _within_input_mean_residual


def _row(seed, k, reps, r_lo, r_hi, reasons=None, atomic=False):
    n = k.size
    return {"seed": seed, "k": k, "k_reps": reps, "r_lo": r_lo, "r_hi": r_hi, "ok": np.ones(n, bool),
            "written": np.ones(n, bool), "repeat_inputs_differ": False, "reasons": reasons or {},
            "atomic_written": np.full(n, atomic, bool)}


def test_bitwise_identical_repeats_keep_per_launch_statistics():
    k = np.array([1.0, 2.0])
    st = _execution_status([_row(0, k, [k.copy()], k, k)], [{"float_atomics": False}], 2)
    assert st["statistics"] == "per launch" and st["units_with_different_repeats"] == 0


def test_atomic_programs_with_different_repeats_are_averaged_within_the_input():
    # the differing elements are written by float atomics (element-level evidence since the audit F07 fix)
    rng = np.random.default_rng(0)
    rows = []
    for s in range(3):
        r_lo = rng.standard_normal(4)
        r_hi = r_lo + 1e-9
        ks = [r_lo + rng.standard_normal(4) * 1e-6 for _ in range(8)]
        rows.append(_row(s, ks[0], ks[1:], r_lo, r_hi, atomic=True))
    st = _execution_status(rows, [{"float_atomics": True}], 8)
    assert st["statistics"] == "within-input mean" and st["launches_per_input"] == 8
    lo, hi = _within_input_mean_residual(rows)
    for u, row in enumerate(rows):
        ks = [row["k"]] + row["k_reps"]
        for i in range(4):   # every real residual mean in [mean(k) - r_hi, mean(k) - r_lo] lies in the enclosure
            m = sum(F(float(k[i])) for k in ks) / len(ks)
            assert F(float(lo[u, i])) <= m - F(float(row["r_hi"][i]))
            assert m - F(float(row["r_lo"][i])) <= F(float(hi[u, i]))


def test_different_repeats_without_a_cause_withhold_the_statistics():
    k = np.array([1.0, 2.0])
    st = _execution_status([_row(0, k, [k + 1.0], k, k)], [{"float_atomics": False}], 2)
    assert st["statistics"] == "withheld" and "not established" in st["status"]


def test_a_race_found_by_the_reference_withholds_the_statistics():
    k = np.array([1.0, 2.0])
    reasons = {"not_established:execution race: read by another thread of the same program after a store@x": 1}
    st = _execution_status([_row(0, k, [k.copy()], k, k, reasons)], [{"float_atomics": True}], 8)
    assert st["statistics"] == "withheld" and st["race_findings"]
