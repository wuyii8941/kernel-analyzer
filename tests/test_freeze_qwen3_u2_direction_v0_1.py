from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.freeze_qwen3_u2_direction_v0_1 import freeze_direction


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeU2DirectionTests(unittest.TestCase):
    def fixture(self, root: Path, *, norm: float = 2.0, cosine: float = 0.95):
        shard = root / "direction.safetensors"
        shard.write_bytes(b"identity-only-test")
        multi = {
            "schema_version": "forkcert.qwen3-calibration-u2-multi-trajectory.v0.1",
            "valid": True,
            "verdict": "VALID_COMPLETE_FOUR_TRAJECTORY_VECTOR_CALIBRATION",
            "calibration_mean_field": {
                "l2": norm,
                "shards": [
                    {"path": str(shard), "sha256": sha(shard), "tensor_key": "w"}
                ],
            },
        }
        multi_path = root / "multi.json"
        multi_path.write_text(json.dumps(multi), encoding="utf-8")
        diagnostic = {
            "schema_version": "forkcert.qwen3-u2-direction-stability.v0.1",
            "valid": True,
            "verdict": "VALID_U2_DIRECTION_STABILITY_CALIBRATION_DIAGNOSTIC",
            "inputs": {"multi_summary": {"sha256": sha(multi_path)}},
            "diagnostics": {
                "full_calibration_mean_norm": norm,
                "minimum_full_vs_leave_one_out_cosine": cosine,
                "crossfit_projection_sample_variance": 0.2,
                "leave_one_out_rows": [
                    {"held_out_projection_on_leave_one_out_direction": value}
                    for value in (1.0, 1.2, 0.8, 1.1)
                ],
            },
        }
        diagnostic_path = root / "diagnostic.json"
        diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
        spec = {
            "schema_version": "forkcert.qwen3-u2-direction-freeze-spec.v0.1",
            "status": "FROZEN_BEFORE_DIRECTION_DIAGNOSTIC_REVIEW",
            "vector_measurement_floor_l2": 0.5,
            "minimum_full_vs_leave_one_out_cosine": 0.8,
            "desired_projection_half_width": 0.1,
            "projection_variance_floor_sd": 0.01,
            "projection_shift_existence_floor": 0.0,
            "threshold_source": "independent synthetic controls",
            "claim_role": "MEMBER_OF_JOINT_CONFIRMATION_FAMILY",
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        return multi, multi_path, diagnostic, diagnostic_path, spec, spec_path

    def test_passing_independent_thresholds_freeze_exact_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixture(Path(directory))
            result = freeze_direction(*values)
            self.assertTrue(result["valid"])
            self.assertEqual(result["endpoint_name"], "U2_calibration_direction_shift")
            self.assertEqual(result["direction"]["normalization_l2"], 2.0)

    def test_small_or_unstable_direction_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixture(Path(directory), norm=0.2, cosine=0.5)
            result = freeze_direction(*values)
            self.assertFalse(result["valid"])
            self.assertEqual(result["verdict"], "UNINSTANTIATED_DIRECTION")
            self.assertEqual(len(result["reason"]), 2)

    def test_uninstantiated_thresholds_fail_before_observed_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = list(self.fixture(Path(directory)))
            spec = values[4]
            spec["status"] = "UNINSTANTIATED_DO_NOT_FREEZE_DIRECTION"
            with self.assertRaisesRegex(ValueError, "not frozen"):
                freeze_direction(*values)


if __name__ == "__main__":
    unittest.main()
