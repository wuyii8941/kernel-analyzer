from __future__ import annotations

import unittest

from forkcert.bounds import (
    ErrorSource,
    assemble_logprob_bound,
    assemble_semi_certified_probability_bound,
    legal_sources_valid,
    phase2_decision,
    unit_roundoff,
)
from forkcert.inject import BugInjection, inject_logprob_bug
from forkcert.ladder import attribution_from_measurements


class BoundsLadderInjectTest(unittest.TestCase):
    def test_unit_roundoff(self) -> None:
        self.assertAlmostEqual(unit_roundoff("bf16"), 2.0**-8)
        self.assertAlmostEqual(unit_roundoff("fp32"), 2.0**-24)

    def test_assemble_bound(self) -> None:
        result = assemble_logprob_bound(
            [
                ErrorSource(
                    name="x",
                    mechanism="reduction_order",
                    dtype="fp32",
                    reduction_length=8,
                    sum_abs=10.0,
                    reduction="tree",
                    propagation=2.0,
                )
            ]
        )
        self.assertGreater(result.activation_bound_worst, 0)
        self.assertAlmostEqual(result.logprob_bound_worst, 2.0 * result.activation_bound_worst)

    def test_probability_budget_is_split_across_sources(self) -> None:
        result = assemble_logprob_bound(
            [
                ErrorSource(name="a", mechanism="reduction_order", dtype="fp32", reduction_length=8, sum_abs=1.0),
                ErrorSource(name="b", mechanism="reduction_order", dtype="fp32", reduction_length=8, sum_abs=1.0),
            ],
            delta=1e-6,
        )
        self.assertEqual(result.per_source[0]["probability_failure_budget"], 5e-7)
        self.assertEqual(result.per_source[1]["probability_failure_budget"], 5e-7)

    def test_semi_certified_keeps_propagation_inside_each_rss_term(self) -> None:
        common = dict(
            mechanism="reduction_order",
            dtype="fp32",
            reduction_length=8,
            sum_abs=1.0,
            logprob_lipschitz=1.0,
            difference_injection=True,
            shared_rounding_cancelled=True,
            local_error_independent=True,
            propagation_empirically_calibrated=True,
        )
        result = assemble_semi_certified_probability_bound(
            [
                ErrorSource(name="a", propagation=2.0, **common),
                ErrorSource(name="b", propagation=3.0, **common),
            ]
        )
        terms = [row["logprob_bound_prob_term"] for row in result["per_source"]]
        self.assertAlmostEqual(result["logprob_bound_prob"], (terms[0] ** 2 + terms[1] ** 2) ** 0.5)
        self.assertTrue(result["validation"])
        self.assertEqual(result["certificate_kind"], "semi_certified")

    def test_semi_certified_rejects_unverified_difference_source(self) -> None:
        result = assemble_semi_certified_probability_bound(
            [ErrorSource(name="x", mechanism="reduction_order", dtype="fp32")]
        )
        self.assertFalse(result["validation"])
        self.assertTrue(result["validation_failures"])

    def test_phase2_decision_requires_bound_to_cover_empirical_delta(self) -> None:
        self.assertTrue(phase2_decision(0.5, 1.0).startswith("VIOLATION"))
        self.assertTrue(phase2_decision(2.0, 1.0).startswith("GO"))
        self.assertTrue(phase2_decision(2000.0, 1.0).startswith("DOWNGRADE"))

    def test_legal_source_requires_explicit_provenance(self) -> None:
        source = ErrorSource(name="x", mechanism="reduction_order", dtype="fp32")
        ok, failures = legal_sources_valid([source])
        self.assertFalse(ok)
        self.assertIn("assumptions_verified", failures[0])

        source.assumptions_verified = True
        source.algorithm_order_known = True
        source.input_norm_measured = True
        source.propagation_certified = True
        ok, failures = legal_sources_valid([source])
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_attribution_is_relative_to_composite_not_additive(self) -> None:
        rows = attribution_from_measurements(
            [
                {
                    "level": "L1",
                    "variable": "a",
                    "mechanism": "m",
                    "first_observed_diff_l2": 1,
                    "max_activation_diff_l2": 2,
                    "propagation_gain_first_to_last": 2,
                    "final_logprob_delta": 3,
                },
                {
                    "level": "L6",
                    "variable": "b",
                    "mechanism": "m",
                    "first_observed_diff_l2": 1,
                    "max_activation_diff_l2": 2,
                    "propagation_gain_first_to_last": 2,
                    "final_logprob_delta": 1,
                },
            ]
        )
        self.assertAlmostEqual(rows[0].relative_to_composite_percent, 300.0)
        self.assertAlmostEqual(rows[1].relative_to_composite_percent, 100.0)
        self.assertFalse(rows[0].additive_attribution_valid)

    def test_inject_logprob_bug(self) -> None:
        rows = [{"case_id": "c", "token_index": 0, "logp_ref": -1.0, "logp_alt": -1.0, "metadata": {}}]
        injected = inject_logprob_bug(rows, BugInjection("bug", "desc", 0.5))
        self.assertAlmostEqual(injected[0]["logp_alt"], -0.5)
        self.assertAlmostEqual(injected[0]["logprob_delta"], 0.5)
        self.assertEqual(injected[0]["metadata"]["bug_injection"]["name"], "bug")


if __name__ == "__main__":
    unittest.main()
