from __future__ import annotations

import math
import unittest

from theory_oracle.analyze_qwen3_partial_core_endpoint_influence_v0_1 import (
    summarize_records,
)
from theory_oracle.bias_oracle_population_v0_2 import EffectRecord


class PartialCoreEndpointInfluenceTests(unittest.TestCase):
    def test_state_sign_runtime_variance_and_phase_influence_remain_separate(self) -> None:
        records = []
        values = {
            "early-a": (1.0, 1.0),
            "early-b": (-2.0, -2.0),
            "early-c": (1.0, 1.5),
        }
        for state, effects in values.items():
            for repeat, effect in enumerate(effects, start=1):
                records.append(
                    EffectRecord(
                        trajectory_id="calibration-0",
                        phase="early",
                        state_id=state,
                        repeat_id=repeat,
                        effect=effect,
                    )
                )
        result = summarize_records(records)
        self.assertTrue(result["valid"])
        self.assertEqual(result["state_effect_sign_counts"], {"positive": 2, "zero": 0, "negative": 1})
        self.assertEqual(result["states_with_observed_nonzero_runtime_variance"], 1)
        self.assertFalse(
            result["phase_profiles"]["early"][
                "sign_stable_to_any_single_state_deletion"
            ]
        )
        self.assertFalse(result["population_B_claim_allowed"])

    def test_nonfinite_effect_invalidates_instead_of_changing_sample(self) -> None:
        records = [
            EffectRecord(
                trajectory_id="calibration-0",
                phase="early",
                state_id="bad-state",
                repeat_id=repeat,
                effect=effect,
            )
            for repeat, effect in ((1, 1.0), (2, math.nan))
        ]
        result = summarize_records(records)
        self.assertFalse(result["valid"])
        self.assertEqual(result["states"], 0)
        self.assertIn("bad-state: nonfinite paired effect", result["errors"])


if __name__ == "__main__":
    unittest.main()
