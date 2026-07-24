from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from theory_oracle.aggregate_qwen3_calibration_u2_multi_trajectory_v0_1 import (
    aggregate_trajectory_tensor,
    validate_inputs,
)


class MultiTrajectoryU2Tests(unittest.TestCase):
    def test_signed_mean_can_cancel_large_trajectory_effects(self) -> None:
        tensors = [
            torch.tensor([-1.0], dtype=torch.float64),
            torch.tensor([-1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        ]
        mean, between = aggregate_trajectory_tensor(tensors)
        self.assertAlmostEqual(float(mean.item()), 0.0)
        self.assertAlmostEqual(between, 4.0 / 3.0)

    def test_fixed_trajectory_shift_has_zero_between_H(self) -> None:
        tensors = [torch.tensor([0.25], dtype=torch.float64) for _ in range(4)]
        mean, between = aggregate_trajectory_tensor(tensors)
        self.assertAlmostEqual(float(mean.item()), 0.25)
        self.assertAlmostEqual(between, 0.0)

    def test_input_identity_requires_exact_four_frozen_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for index in range(4):
                path = root / f"t{index}.json"
                value = {
                    "valid": True,
                    "construction": {
                        "trajectory_id": f"calibration-{index}",
                        "trajectory_count": 1,
                        "states": 24,
                        "query_id": "Q-R",
                        "parameters": 1,
                    },
                    "parameter_rows": [{"parameter_name": "weight"}],
                    "source_fingerprint": f"source-{index}",
                }
                path.write_text(json.dumps(value), encoding="utf-8")
                summaries.append((path, value))
            by_trajectory, _, errors = validate_inputs(summaries)
            self.assertFalse(errors)
            self.assertEqual(set(by_trajectory), {f"calibration-{i}" for i in range(4)})

            summaries[3][1]["construction"]["trajectory_id"] = "replacement"
            _, _, errors = validate_inputs(summaries)
            self.assertTrue(any("calibration-0..calibration-3" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
