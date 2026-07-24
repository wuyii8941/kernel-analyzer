from __future__ import annotations

import unittest

from theory_oracle.bias_oracle_contributor_v0_1 import (
    ContributorArmRecord,
    apply_contribution_sensitivity,
    apply_injection_sensitivity,
    estimate_injection_contribution,
    estimate_repair_contribution,
)


def records(
    trajectory_values: list[tuple[float, float, float]],
    *,
    repeat_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
) -> list[ContributorArmRecord]:
    rows = []
    for trajectory_index, (reference, candidate, repair) in enumerate(trajectory_values):
        for phase in ("early", "middle", "late"):
            for state_index in range(2):
                for repeat_id, offsets in enumerate(repeat_offsets):
                    for arm, value, offset in zip(
                        ("REFERENCE", "FULL_CANDIDATE", "CANDIDATE_REPAIR"),
                        (reference, candidate, repair),
                        offsets,
                        strict=True,
                    ):
                        rows.append(
                            ContributorArmRecord(
                                trajectory_id=f"t{trajectory_index}",
                                phase=phase,
                                state_id=f"{phase}-s{state_index}",
                                repeat_id=repeat_id,
                                arm=arm,
                                outcome=value + offset,
                            )
                        )
    return rows


class ContributorProfileTests(unittest.TestCase):
    def test_reference_context_injection_is_a_separate_estimand(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase in ("early", "middle", "late"):
                for state_index in range(2):
                    for repeat_id in (0, 1):
                        common = {
                            "trajectory_id": f"t{trajectory_index}",
                            "phase": phase,
                            "state_id": f"{phase}-s{state_index}",
                            "repeat_id": repeat_id,
                        }
                        rows.extend(
                            [
                                ContributorArmRecord(
                                    **common, arm="REFERENCE", outcome=0.0
                                ),
                                ContributorArmRecord(
                                    **common, arm="FULL_CANDIDATE", outcome=1.0
                                ),
                                ContributorArmRecord(
                                    **common,
                                    arm="REFERENCE_INJECTION",
                                    outcome=0.25,
                                ),
                            ]
                        )
        result = estimate_injection_contribution(
            rows,
            required_phases=("early", "middle", "late"),
            frozen_bias_direction=1,
        )
        self.assertEqual(result["study_type"], "REFERENCE_CONTEXT_INJECTION")
        self.assertAlmostEqual(
            result["profiles"]["reference_context_injection_minus_reference"][
                "B"
            ]["estimate"],
            0.25,
        )
        self.assertEqual(
            result["directional_gate"]["directional_injection_verdict"],
            "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_INJECTION",
        )
        self.assertIn(
            "reference-context injection is not sufficiency", result["nonclaims"]
        )
        sensitivity = apply_injection_sensitivity(result, decision_alpha=0.05)
        self.assertEqual(
            sensitivity["post_sensitivity_verdict"],
            "DIRECTIONAL_INJECTION_SENSITIVITY_SUPPORTED_INTEGRITY_PENDING",
        )

    def estimate(self, rows, **kwargs):
        return estimate_repair_contribution(
            rows,
            required_phases=("early", "middle", "late"),
            frozen_bias_direction=1,
            **kwargs,
        )

    def test_phase_target_does_not_use_global_cancellation(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase, repair_outcome in (
                ("early", 0.0),
                ("middle", 1.0),
                ("late", 2.0),
            ):
                for state_index in range(2):
                    for repeat_id in (0, 1):
                        common = {
                            "trajectory_id": f"t{trajectory_index}",
                            "phase": phase,
                            "state_id": f"{phase}-s{state_index}",
                            "repeat_id": repeat_id,
                        }
                        rows.extend(
                            [
                                ContributorArmRecord(
                                    **common, arm="REFERENCE", outcome=0.0
                                ),
                                ContributorArmRecord(
                                    **common, arm="FULL_CANDIDATE", outcome=1.0
                                ),
                                ContributorArmRecord(
                                    **common,
                                    arm="CANDIDATE_REPAIR",
                                    outcome=repair_outcome,
                                ),
                            ]
                        )
        result = self.estimate(rows, target_phase="early")
        self.assertEqual(result["construction"]["target_phase"], "early")
        self.assertAlmostEqual(
            result["profiles"]["repair_contribution_candidate_minus_repair"][
                "B"
            ]["estimate"],
            1.0,
        )

    def test_small_stable_directional_contribution_is_detected(self) -> None:
        result = self.estimate(records([(0.0, 0.01, 0.006)] * 8))
        self.assertEqual(
            result["directional_gate"]["baseline_transport_verdict"],
            "BASELINE_BIAS_TRANSPORTED_IN_FROZEN_DIRECTION",
        )
        self.assertEqual(
            result["directional_gate"]["directional_contribution_verdict"],
            "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTION",
        )
        self.assertEqual(
            result["primary_interval_verdict"],
            "PRIMARY_INTERVAL_SUPPORTS_DIRECTIONAL_CONTRIBUTOR_SENSITIVITY_PENDING",
        )
        self.assertFalse(result["final_claim_allowed"])

    def test_reference_cancels_from_contribution_but_not_baseline(self) -> None:
        values = [(float(index), float(index) + 1.0, float(index) + 0.5) for index in range(8)]
        result = self.estimate(records(values))
        contribution = result["profiles"]["repair_contribution_candidate_minus_repair"]
        self.assertAlmostEqual(contribution["B"]["estimate"], 0.5)
        self.assertAlmostEqual(contribution["H"]["between_trajectory_variance"], 0.0)
        self.assertTrue(result["construction"]["same_state_arm_covariance_retained"])

    def test_paired_repeat_variability_is_N_not_state_H(self) -> None:
        result = self.estimate(
            records(
                [(0.0, 1.0, 0.5)] * 8,
                repeat_offsets=((0.0, -0.2, 0.2), (0.0, 0.2, -0.2)),
            )
        )
        contribution = result["profiles"]["repair_contribution_candidate_minus_repair"]
        self.assertGreater(contribution["N"]["mean_same_state_paired_effect_variance"], 0.0)
        self.assertAlmostEqual(
            contribution["H"]["mean_within_phase_state_variance_repeat_corrected"],
            0.0,
        )

    def test_repair_that_increases_bias_is_opposite_contribution(self) -> None:
        result = self.estimate(records([(0.0, 1.0, 2.0)] * 8))
        self.assertEqual(
            result["directional_gate"]["directional_contribution_verdict"],
            "REPAIR_MOVES_AGAINST_FROZEN_BIAS_DIRECTION",
        )

    def test_directional_contribution_can_overshoot_and_worsen_absolute_bias(self) -> None:
        result = self.estimate(
            records([(0.0, 1.0, -2.0)] * 8),
            absolute_reduction_enabled=True,
            absolute_reduction_interval_alpha=0.05 / 3.0,
        )
        self.assertTrue(result["absolute_reduction_gate"]["overshoot_established"])
        self.assertEqual(
            result["absolute_reduction_gate"]["verdict"],
            "ABSOLUTE_BIAS_REDUCTION_NOT_ESTABLISHED",
        )
        self.assertEqual(
            result["primary_interval_verdict"],
            "DIRECTIONAL_CONTRIBUTOR_WITH_OVERSHOOT_SENSITIVITY_PENDING",
        )

    def test_simultaneous_intervals_can_support_absolute_reduction(self) -> None:
        result = self.estimate(
            records([(0.0, 1.0, 0.25)] * 8),
            absolute_reduction_enabled=True,
            absolute_reduction_interval_alpha=0.05 / 3.0,
            minimum_absolute_reduction=0.5,
        )
        self.assertEqual(
            result["absolute_reduction_gate"]["verdict"],
            "SIMULTANEOUS_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCTION",
        )
        self.assertEqual(
            result["primary_interval_verdict"],
            "PRIMARY_INTERVAL_SUPPORTS_ABSOLUTE_BIAS_REDUCER_SENSITIVITY_PENDING",
        )

    def test_incomplete_arm_cell_fails_closed(self) -> None:
        rows = records([(0.0, 1.0, 0.5)] * 8)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete contributor arm cells"):
            self.estimate(rows)

    def test_sensitivity_supports_stable_primary_but_keeps_integrity_pending(self) -> None:
        profile = self.estimate(records([(0.0, 1.0, 0.5)] * 8))
        result = apply_contribution_sensitivity(profile, decision_alpha=0.05)
        self.assertTrue(result["sensitivity"]["supports_primary_contribution"])
        self.assertIn("SENSITIVITY_SUPPORTED_INTEGRITY_PENDING", result["post_sensitivity_verdict"])
        self.assertFalse(result["final_claim_allowed"])

    def test_sensitivity_never_promotes_primary_failure(self) -> None:
        profile = self.estimate(records([(0.0, 1.0, 1.0)] * 8))
        result = apply_contribution_sensitivity(profile, decision_alpha=0.05)
        self.assertEqual(
            result["status"],
            "NOT_RUN_PRIMARY_INTERVAL_DID_NOT_SUPPORT_CONTRIBUTION",
        )
        self.assertFalse(result["final_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
