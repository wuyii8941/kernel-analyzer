"""Bounded route and design sensitivity (DSL v2 rc3 02 8.2-8.5, 04 W6), CPU only."""
from __future__ import annotations

import numpy as np

from kernel_analyzer.reference_eval.analysis import _summarize
from kernel_analyzer.reference_eval.sensitivity import approximate_mde, bounded_route


def test_bounded_route_covers_the_true_mean_at_least_at_the_nominal_level():
    M, mu, sd, n, alpha = 1.0, 0.15, 0.5, 64, 0.05
    true_mu = float(np.clip(np.random.default_rng(123).normal(mu, sd, 2_000_000), -M, M).mean())
    rng = np.random.default_rng(0)
    miss = 0
    for _ in range(2000):
        a = np.clip(rng.normal(mu, sd, n), -M, M)
        lo, hi = bounded_route(a - 1e-9, a + 1e-9, M, alpha)["interval"]
        miss += not (lo <= true_mu <= hi)
    assert miss / 2000 <= alpha


def test_zero_variance_does_not_block_the_bounded_route():
    a = np.full(200, 0.9)
    b = bounded_route(a, a, 1.0, 0.05)
    assert b["verdict"] == "DETECTED_POSITIVE"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "essential"))
    import contract_v3 as CV
    t = _summarize("x", "R1", a, a, 0.05)
    t["per_unit_bounds"] = [[0.9, 0.9]] * a.size
    # the approximate route cannot judge an all-equal sample (contract_v3: degenerate)
    assert CV.statistical_judgment(t)["judgment"].startswith("cannot judge")


def test_endpoints_outside_the_bound_are_an_error_not_an_interval():
    b = bounded_route(np.array([2.0, 0.1]), np.array([2.5, 0.2]), 1.0, 0.05)
    assert b["verdict"] == "ERROR" and "interval" not in b


def test_sensitivity_shrinks_with_the_number_of_units():
    rng = np.random.default_rng(1)
    small = approximate_mde(*(2 * [rng.normal(0, 1, 16)]), 0.05)["mde"]
    large = approximate_mde(*(2 * [rng.normal(0, 1, 256)]), 0.05)["mde"]
    assert large < small
