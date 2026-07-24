from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenBoundaryValueTests(unittest.TestCase):
    def test_complete_bank_nonredundancy_record(self) -> None:
        result = json.loads(
            (
                ROOT
                / "results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/boundary_value_v0_1.json"
            ).read_text(encoding="utf-8")
        )
        parent = json.loads(
            (
                ROOT
                / "results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/evaluation.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(result["applicable_tokens"], parent["applicable_tokens"])
        self.assertEqual(result["parent_state_clusters"], parent["total_rollout_states"])
        self.assertEqual(result["stable_event_count"], len(parent["events"]))
        self.assertTrue(result["nonredundant_on_frozen_bank"])
        self.assertFalse(result["rankings_identical"])
        self.assertEqual([row["boundary_rank"] for row in result["events"]], [1, 2])
        self.assertGreater(
            result["max_raw_non_event"]["raw_score"],
            result["min_raw_event"]["raw_score"],
        )
        self.assertEqual(result["correctness"], "NO CLAIM")
        self.assertEqual(result["predictive_generalization"], "NO CLAIM")


if __name__ == "__main__":
    unittest.main()
