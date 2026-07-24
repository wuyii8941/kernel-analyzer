from __future__ import annotations

import unittest

from theory_oracle.audit_qwen3_calibration_capture_batch_v0_1 import (
    validate_source_binding,
)


class CalibrationSourceBindingTests(unittest.TestCase):
    def test_source_config_is_bound_to_plan_identity(self) -> None:
        digest = "a" * 64
        plan = {
            "identity": {
                "trajectory_seed": 17,
                "data_slice_id": "forkcert_builtin_arithmetic[64:128]",
            }
        }
        config = {
            "dataset": {
                "name": "forkcert_builtin_arithmetic",
                "offset": 64,
                "max_prompts": 64,
            },
            "training": {"seed": 17, "max_steps": 300},
        }
        metadata = {
            "config": config,
            "compile_audit": {"backend_compiles": 0, "runtime_invocations": 0},
            "transition_capture_targets": [
                {"plan_digest": digest} for _ in range(24)
            ],
        }
        self.assertTrue(all(validate_source_binding(plan, digest, config, metadata).values()))

    def test_declared_seed_cannot_hide_actual_seed_drift(self) -> None:
        digest = "b" * 64
        plan = {
            "identity": {
                "trajectory_seed": 17,
                "data_slice_id": "forkcert_builtin_arithmetic[64:128]",
            }
        }
        config = {
            "dataset": {
                "name": "forkcert_builtin_arithmetic",
                "offset": 64,
                "max_prompts": 64,
            },
            "training": {"seed": 18, "max_steps": 300},
        }
        metadata = {
            "config": config,
            "compile_audit": {"backend_compiles": 0, "runtime_invocations": 0},
            "transition_capture_targets": [
                {"plan_digest": digest} for _ in range(24)
            ],
        }
        checks = validate_source_binding(plan, digest, config, metadata)
        self.assertFalse(checks["source_seed_matches_plan"])


if __name__ == "__main__":
    unittest.main()
