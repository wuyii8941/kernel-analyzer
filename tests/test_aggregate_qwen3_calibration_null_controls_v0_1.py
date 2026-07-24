import copy
import hashlib
import tempfile
import unittest
from pathlib import Path


from theory_oracle.aggregate_qwen3_calibration_null_controls_v0_1 import (
    extract_state_controls,
    summarize_state_controls,
    task_evaluator_contrasts,
)


def arm_record(arm: str, repeat_id: int, offset: float = 0.0) -> dict:
    return {
        "identity": {"arm": arm, "repeat_id": repeat_id},
        "outcomes": {
            "T1a_arm_loss": {"value": 1.0 + offset},
            "T1b_arm_nll": {"value": 2.0 + offset},
            "propagation_ledgers": {
                "training_loss": 3.0 + offset,
                "pre_clip_gradient_norm": 4.0 + offset,
                "parameter_update_l2": 5.0 + offset,
            },
            "parameter_update_artifact": {"sha256": f"{arm}-update"},
            "next_state_digests": {"parameter": f"{arm}-next"},
            "semantic_events": {"optimizer_step_skipped": False},
        },
    }


def task_evaluation() -> dict:
    task = {"valid": True}
    for endpoint, value_field in (("T1a", "loss"), ("T1b", "mean_nll")):
        task[endpoint] = {"arm_results": {}}
        for arm_index, arm in enumerate(("reference", "candidate")):
            rows = []
            for transition_repeat in (1, 2):
                for evaluator_repeat in (1, 2):
                    rows.append(
                        {
                            "transition_repeat": transition_repeat,
                            "evaluator_repeat": evaluator_repeat,
                            value_field: 10.0 + arm_index,
                        }
                    )
            task[endpoint]["arm_results"][arm] = rows
    return task


class CalibrationNullControlsTests(unittest.TestCase):
    def test_exact_controls_remain_separate_from_cross_implementation_effect(self) -> None:
        bundle = {
            "arm_records": [
                arm_record("reference", 1),
                arm_record("candidate", 1, offset=0.25),
                arm_record("reference", 2),
                arm_record("candidate", 2, offset=0.25),
            ]
        }
        states = []
        for phase in ("early", "middle", "late"):
            states.append(
                extract_state_controls(
                    {"state_id": f"state-{phase}", "phase": phase},
                    bundle,
                    task_evaluation(),
                    verify_update_artifacts=False,
                )
            )
        summary = summarize_state_controls(states)
        for arm in ("reference", "candidate"):
            controls = summary["within_implementation"][arm]
            self.assertEqual(
                controls["scalar_controls"]["training_loss"]["nonzero_state_count"],
                0,
            )
            self.assertTrue(
                controls["exact_artifact_event_controls"][
                    "parameter_update_artifact_sha_equal"
                ]["all_equal"]
            )
        self.assertEqual(
            summary["within_evaluator"]["T1a"]["candidate"][
                "max_absolute_contrast"
            ],
            0.0,
        )

    def test_evaluator_repeat_identity_mismatch_fails(self) -> None:
        task = task_evaluation()
        corrupted = copy.deepcopy(task)
        corrupted["T1a"]["arm_results"]["candidate"][1][
            "evaluator_repeat"
        ] = 1
        with self.assertRaisesRegex(ValueError, "duplicate evaluator repeat"):
            task_evaluator_contrasts(corrupted)

    def test_update_artifact_hash_is_verified(self) -> None:
        bundle = {
            "arm_records": [
                arm_record("reference", 1),
                arm_record("candidate", 1, offset=0.25),
                arm_record("reference", 2),
                arm_record("candidate", 2, offset=0.25),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "update.bin"
            artifact.write_bytes(b"frozen update")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            for row in bundle["arm_records"]:
                row["outcomes"]["parameter_update_artifact"] = {
                    "path": str(artifact),
                    "sha256": digest,
                }
            bundle["arm_records"][0]["outcomes"]["parameter_update_artifact"][
                "sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                extract_state_controls(
                    {"state_id": "state-early", "phase": "early"},
                    bundle,
                    task_evaluation(),
                    verify_update_artifacts=True,
                )


if __name__ == "__main__":
    unittest.main()
