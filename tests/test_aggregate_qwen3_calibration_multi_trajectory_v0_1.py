from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from theory_oracle.aggregate_qwen3_calibration_multi_trajectory_v0_1 import (
    complete_endpoint_payload,
    validate_capture_audit,
    validate_calibration_layout,
    validate_task_semantics,
)
from theory_oracle.evaluate_qwen3_calibration_state_endpoints_v0_1 import endpoint_profile


ROOT = Path(__file__).resolve().parents[1]


def frozen_plans() -> list[dict]:
    return [
        json.loads(
            (
                ROOT
                / "theory_oracle"
                / f"QWEN3_BIAS_ORACLE_CALIBRATION_{index}_CAPTURE_PLAN_V0_1.json"
            ).read_text(encoding="utf-8")
        )
        for index in range(4)
    ]


class MultiTrajectoryCalibrationAggregateTests(unittest.TestCase):
    def test_legacy_task_output_is_revalidated_under_current_pairing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_1 = root / "bank1.json"
            bank_2 = root / "bank2.json"
            bank_1.write_text("{}\n", encoding="utf-8")
            bank_2.write_text("{}\n", encoding="utf-8")
            arm_results = {
                "reference": [
                    {"transition_repeat": r, "evaluator_repeat": e, "loss": 1.0, "mean_nll": 2.0}
                    for r in (1, 2) for e in (1, 2)
                ],
                "candidate": [
                    {"transition_repeat": r, "evaluator_repeat": e, "loss": 1.25, "mean_nll": 2.5}
                    for r in (1, 2) for e in (1, 2)
                ],
            }
            task = {
                "schema_version": "forkcert.qwen3-calibration-state-endpoints.v0.1",
                "valid": True,
                "environment": {"evaluator_code_sha256": "legacy-code"},
                "artifacts": {
                    "T1a_bank": {"path": str(bank_1), "sha256": hashlib.sha256(bank_1.read_bytes()).hexdigest(), "content_sha256": "same"},
                    "T1a_bank_repeat": {"path": str(bank_2), "sha256": hashlib.sha256(bank_2.read_bytes()).hexdigest(), "content_sha256": "same"},
                },
                "T1a": {"status": "MEASURED", "arm_results": arm_results, "profile": endpoint_profile(arm_results, "loss")},
                "T1b": {"status": "MEASURED", "arm_results": arm_results, "profile": endpoint_profile(arm_results, "mean_nll")},
            }
            task_path = root / "task.json"
            task_path.write_text(json.dumps(task), encoding="utf-8")
            link = {"path": str(task_path), "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest()}
            bundle = {"arm_records": [{"provenance": {"task_evaluation": link}} for _ in range(4)]}
            evidence, errors = validate_task_semantics(bundle)
            self.assertFalse(errors)
            self.assertTrue(evidence["numeric_profiles_match_current_semantics"])
            self.assertEqual(evidence["randomness_scope"], "LEGACY_IMPLICIT_SCOPE_NUMERICALLY_REVALIDATED")

    def test_nonnegative_profile_is_not_serialized_as_conditional_B(self) -> None:
        estimate = {
            "B": 0.25,
            "conditional_B": [{"phase": "early", "estimate": 0.25}],
            "H": {},
            "N": {},
            "U": {},
            "trajectory_rows": [],
            "phase_rows": [],
            "state_rows": [],
        }
        result = complete_endpoint_payload(
            "NONNEGATIVE_EVENT_DISAGREEMENT_PROFILE_NOT_B",
            Counter({"MEASURED": 8}),
            [],
            estimate,
        )
        self.assertFalse(result["signed_B_candidate"])
        self.assertEqual(result["calibration_profile_mean"], 0.25)
        self.assertNotIn("calibration_average_estimate", result)
        self.assertNotIn("conditional_B", result)

    def test_capture_audit_is_bound_to_plan_source_and_state_census(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            audits = results / "snapshot_audits"
            audits.mkdir(parents=True)
            config = root / "config.yaml"
            metadata = results / "source_dump.metadata.json"
            config.write_text("seed: 1\n", encoding="utf-8")
            metadata.write_text("{}\n", encoding="utf-8")
            targets = [
                {"state_id": f"s-{index}"} for index in range(24)
            ]
            plan = {"capture_root": str(root / "captures"), "targets": targets}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            state_rows = []
            for target in targets:
                path = audits / f"{target['state_id']}.json"
                path.write_text("{}\n", encoding="utf-8")
                state_rows.append(
                    {
                        "state_id": target["state_id"],
                        "snapshot_valid": True,
                        "history_exact": True,
                        "target_identity_exact": True,
                        "audit_path": str(path),
                        "audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            audit = {
                "valid": True,
                "verdict": "VALID",
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "capture_root": plan["capture_root"],
                "checks": {"complete": True},
                "source_evidence": {
                    "checks": {"complete": True},
                    "config_path": str(config),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "metadata_path": str(metadata),
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                },
                "states": state_rows,
            }
            (results / "capture_batch_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            evidence, errors = validate_capture_audit(plan_path, plan, results)
            self.assertFalse(errors)
            self.assertEqual(evidence["state_audits"], 24)

            plan_path.write_text(json.dumps({**plan, "changed": True}), encoding="utf-8")
            _, errors = validate_capture_audit(plan_path, plan, results)
            self.assertTrue(any("plan hash mismatch" in error for error in errors))

    def test_frozen_four_trajectory_layout_is_valid(self) -> None:
        self.assertEqual(validate_calibration_layout(frozen_plans()), [])

    def test_duplicate_seed_and_incomplete_phase_fail_closed(self) -> None:
        plans = copy.deepcopy(frozen_plans())
        plans[1]["identity"]["trajectory_seed"] = plans[0]["identity"]["trajectory_seed"]
        plans[2]["targets"].pop()
        errors = validate_calibration_layout(plans)
        self.assertTrue(any("trajectory_seed" in error for error in errors))
        self.assertTrue(any("8x3" in error for error in errors))

    def test_renaming_a_trajectory_does_not_redefine_calibration(self) -> None:
        plans = copy.deepcopy(frozen_plans())
        plans[3]["identity"]["trajectory_id"] = "replacement-after-looking"
        errors = validate_calibration_layout(plans)
        self.assertTrue(any("calibration-0..calibration-3" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
