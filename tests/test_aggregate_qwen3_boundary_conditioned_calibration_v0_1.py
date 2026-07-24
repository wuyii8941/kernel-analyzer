from __future__ import annotations

import unittest

import torch

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    aggregate_state_profiles,
    state_profile,
)


class BoundaryConditionedCalibrationTests(unittest.TestCase):
    def test_no_eligible_decisions_is_uninstantiated_not_zero_or_nan(self) -> None:
        row = state_profile(
            old_logps=torch.zeros((1, 2)),
            advantages=torch.zeros(1),
            completion_mask=torch.ones((1, 2), dtype=torch.int64),
            reference_logps=torch.zeros((1, 2)),
            candidate_logps=torch.zeros((1, 2)),
            recorded_reference_decisions=torch.zeros((1, 2), dtype=torch.bool),
            recorded_candidate_decisions=torch.zeros((1, 2), dtype=torch.bool),
            taus=[0.01],
        )
        self.assertEqual(row["eligible_decisions"], 0)
        self.assertEqual(
            row["all_eligible_endpoint_status"],
            "UNINSTANTIATED_NO_ELIGIBLE_DECISIONS",
        )
        self.assertIsNone(row["all_eligible_mean_margin_shift"])
        self.assertIsNone(row["reference_to_candidate_on_rate"])
        self.assertIsNone(row["candidate_to_reference_off_rate"])

    def test_nonfinite_margin_on_eligible_decision_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "candidate margin is nonfinite on an eligible decision"
        ):
            state_profile(
                old_logps=torch.zeros((1, 1)),
                advantages=torch.ones(1),
                completion_mask=torch.ones((1, 1), dtype=torch.int64),
                reference_logps=torch.zeros((1, 1)),
                candidate_logps=torch.tensor([[float("nan")]]),
                recorded_reference_decisions=torch.tensor([[False]]),
                recorded_candidate_decisions=torch.tensor([[False]]),
                taus=[0.01],
            )

    def test_zero_global_mean_can_hide_boundary_conditioned_direction(self) -> None:
        # Positive-advantage decisions: reference margins are approximately
        # [-0.001, 0.5]. Candidate shifts them by [+0.002, -0.002]. The global
        # mean cancels, while the predeclared near-boundary subset moves upward
        # and crosses the clipping boundary.
        old = torch.zeros((1, 2))
        reference_ratio = torch.tensor([[1.199, 1.7]])
        candidate_ratio = torch.tensor([[1.201, 1.698]])
        ref_logps = torch.log(reference_ratio)
        cand_logps = torch.log(candidate_ratio)
        advantages = torch.ones(1)
        mask = torch.ones((1, 2), dtype=torch.int64)
        ref_decisions = torch.tensor([[False, True]])
        cand_decisions = torch.tensor([[True, True]])
        row = state_profile(
            old_logps=old,
            advantages=advantages,
            completion_mask=mask,
            reference_logps=ref_logps,
            candidate_logps=cand_logps,
            recorded_reference_decisions=ref_decisions,
            recorded_candidate_decisions=cand_decisions,
            taus=[0.01],
        )
        self.assertAlmostEqual(row["all_eligible_mean_margin_shift"], 0.0, places=5)
        near = row["tau_profiles"]["0.01"]
        self.assertGreater(near["mean_margin_shift"], 0.0)
        self.assertEqual(near["directional_event_shift"], 1.0)
        self.assertEqual(len(near["condition_mask_sha256"]), 64)

    def test_reference_anchor_can_fix_condition_mask_across_runtime_repeats(self) -> None:
        old = torch.zeros((1, 1))
        advantages = torch.ones(1)
        mask = torch.ones((1, 1), dtype=torch.int64)
        # Current reference is far from the boundary, but an independently fixed
        # reference-anchor replicate is near it. The token remains in the
        # conditional set instead of changing membership with this repeat.
        row = state_profile(
            old_logps=old,
            advantages=advantages,
            completion_mask=mask,
            reference_logps=torch.log(torch.tensor([[1.5]])),
            candidate_logps=torch.log(torch.tensor([[1.6]])),
            recorded_reference_decisions=torch.tensor([[True]]),
            recorded_candidate_decisions=torch.tensor([[True]]),
            taus=[0.01],
            condition_reference_logps=torch.log(torch.tensor([[1.199]])),
        )
        self.assertEqual(row["tau_profiles"]["0.01"]["exposures"], 1)
        self.assertEqual(
            len(row["tau_profiles"]["0.01"]["condition_mask_sha256"]), 64
        )

    def test_aggregate_is_state_weighted_not_exposure_pooled(self) -> None:
        rows = [
            {
                "phase": "early",
                "all_eligible_mean_margin_shift": 1.0,
                "tau_profiles": {
                    "0.1": {
                        "exposures": 1,
                        "mean_margin_shift": 1.0,
                        "directional_event_shift": 1.0,
                        "semantic_disagreement": 1.0,
                    }
                },
            },
            {
                "phase": "early",
                "all_eligible_mean_margin_shift": -1.0,
                "tau_profiles": {
                    "0.1": {
                        "exposures": 100,
                        "mean_margin_shift": -1.0,
                        "directional_event_shift": -1.0,
                        "semantic_disagreement": 1.0,
                    }
                },
            },
        ]
        result = aggregate_state_profiles(rows, [0.1])
        self.assertEqual(
            result["tau_profiles"]["0.1"]["state_weighted_mean_margin_shift"],
            0.0,
        )
        self.assertIsNone(
            result["tau_profiles"]["0.1"][
                "phase_balanced_state_weighted_mean_margin_shift"
            ]
        )

    def test_phase_balanced_conditional_does_not_reweight_by_available_states(self) -> None:
        rows = []
        for phase, effects in (
            ("early", [1.0, 1.0, 1.0]),
            ("middle", [-1.0, -1.0]),
            ("late", [-1.0, -1.0]),
        ):
            for effect in effects:
                rows.append(
                    {
                        "phase": phase,
                        "all_eligible_mean_margin_shift": effect,
                        "tau_profiles": {
                            "0.1": {
                                "exposures": 1,
                                "mean_margin_shift": effect,
                                "directional_event_shift": 0.0,
                                "semantic_disagreement": 0.0,
                            }
                        },
                    }
                )
        result = aggregate_state_profiles(rows, [0.1])["tau_profiles"]["0.1"]
        self.assertAlmostEqual(result["state_weighted_mean_margin_shift"], -1.0 / 7.0)
        self.assertAlmostEqual(
            result["phase_balanced_state_weighted_mean_margin_shift"], -1.0 / 3.0
        )

    def test_all_eligible_phase_mean_fails_closed_instead_of_complete_case_deletion(self) -> None:
        rows = [
            {
                "state_id": "identified",
                "phase": "early",
                "all_eligible_endpoint_status": "IDENTIFIED",
                "all_eligible_mean_margin_shift": 1.0,
                "tau_profiles": {
                    "0.1": {
                        "exposures": 0,
                        "mean_margin_shift": None,
                        "directional_event_shift": None,
                        "semantic_disagreement": None,
                    }
                },
            },
            {
                "state_id": "no-eligible-decisions",
                "phase": "early",
                "all_eligible_endpoint_status": "UNINSTANTIATED_NO_ELIGIBLE_DECISIONS",
                "all_eligible_mean_margin_shift": None,
                "tau_profiles": {
                    "0.1": {
                        "exposures": 0,
                        "mean_margin_shift": None,
                        "directional_event_shift": None,
                        "semantic_disagreement": None,
                    }
                },
            },
        ]
        result = aggregate_state_profiles(rows, [0.1])
        self.assertIsNone(result["all_eligible_phase_means"]["early"])
        identification = result["all_eligible_phase_identification"]["early"]
        self.assertFalse(identification["phase_estimand_identified"])
        self.assertEqual(
            identification["uninstantiated_state_ids"],
            ["no-eligible-decisions"],
        )


if __name__ == "__main__":
    unittest.main()
