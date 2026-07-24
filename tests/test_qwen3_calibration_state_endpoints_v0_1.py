from __future__ import annotations

import unittest
import copy

from theory_oracle.evaluate_qwen3_calibration_state_endpoints_v0_1 import (
    TASK_ENDPOINT_RANDOMNESS_SCOPE,
    endpoint_profile,
)


class CalibrationStateEndpointTests(unittest.TestCase):
    def test_evaluator_repeats_are_nested_inside_transition_repeats(self) -> None:
        rows = {
            "reference": [
                {"transition_repeat": 1, "evaluator_repeat": 1, "loss": 1.0},
                {"transition_repeat": 1, "evaluator_repeat": 2, "loss": 1.2},
                {"transition_repeat": 2, "evaluator_repeat": 1, "loss": 2.0},
                {"transition_repeat": 2, "evaluator_repeat": 2, "loss": 2.2},
            ],
            "candidate": [
                {"transition_repeat": 1, "evaluator_repeat": 1, "loss": 1.4},
                {"transition_repeat": 1, "evaluator_repeat": 2, "loss": 1.6},
                {"transition_repeat": 2, "evaluator_repeat": 1, "loss": 1.8},
                {"transition_repeat": 2, "evaluator_repeat": 2, "loss": 2.0},
            ],
        }
        profile = endpoint_profile(rows, "loss")
        effects = [
            row["paired_effect"]
            for row in profile["paired_transition_repeat_effects"]
        ]
        self.assertAlmostEqual(effects[0], 0.4)
        self.assertAlmostEqual(effects[1], -0.2)
        self.assertAlmostEqual(profile["state_effect_signed_mean"], 0.1)
        self.assertAlmostEqual(profile["N_transition_paired_effect_variance"], 0.18)
        self.assertEqual(profile["B_status"], "NOT_POPULATION_B_ONE_STATE")
        self.assertEqual(
            TASK_ENDPOINT_RANDOMNESS_SCOPE["t1a_bank_sampling_variance"],
            "UNIDENTIFIED_ONE_FROZEN_BANK_PER_STATE",
        )

    def test_evaluator_repeat_ids_must_pair_exactly(self) -> None:
        rows = {
            "reference": [
                {"transition_repeat": 1, "evaluator_repeat": 1, "loss": 1.0},
                {"transition_repeat": 1, "evaluator_repeat": 2, "loss": 1.1},
            ],
            "candidate": [
                {"transition_repeat": 1, "evaluator_repeat": 1, "loss": 1.2},
                {"transition_repeat": 1, "evaluator_repeat": 3, "loss": 1.3},
            ],
        }
        with self.assertRaisesRegex(ValueError, "unbalanced evaluator repeats"):
            endpoint_profile(copy.deepcopy(rows), "loss")


if __name__ == "__main__":
    unittest.main()
