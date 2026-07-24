from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "theory_oracle" / "QWEN3_BIAS_ORACLE_C0_MANIFEST_DRAFT_V0_1.json"


class CalibrationTrajectoryPlansTests(unittest.TestCase):
    def test_all_four_materialized_plans_match_frozen_manifest(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        frozen = {
            row["trajectory_id"]: row
            for row in manifest["sampling"]["calibration"]["trajectories"]
        }
        observed_slices = []
        for index in range(4):
            trajectory_id = f"calibration-{index}"
            expected = frozen[trajectory_id]
            plan_path = (
                ROOT
                / "theory_oracle"
                / f"QWEN3_BIAS_ORACLE_CALIBRATION_{index}_CAPTURE_PLAN_V0_1.json"
            )
            config_path = (
                ROOT / "configs" / f"qwen3_bias_oracle_calibration_{index}_v0_1.yaml"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["identity"]["trajectory_id"], trajectory_id)
            self.assertEqual(plan["identity"]["trajectory_seed"], expected["seed"])
            self.assertEqual(config["training"]["seed"], expected["seed"])
            self.assertEqual(config["dataset"]["offset"], expected["dataset_offset"])
            self.assertEqual(config["training"]["max_steps"], 300)
            self.assertEqual(config["dataset"]["max_prompts"], 64)
            observed_slices.append(
                set(range(expected["dataset_offset"], expected["dataset_offset"] + 64))
            )

            targets = plan["targets"]
            self.assertEqual(len(targets), 24)
            self.assertEqual(len({row["state_id"] for row in targets}), 24)
            self.assertEqual(len({row["optimizer_step"] for row in targets}), 24)
            for phase, eligible in (
                ("early", "1:100"),
                ("middle", "101:200"),
                ("late", "201:300"),
            ):
                phase_targets = [row for row in targets if row["phase"] == phase]
                self.assertEqual(len(phase_targets), 8)
                self.assertEqual(
                    [row["optimizer_step"] for row in phase_targets],
                    expected["selected_pre_steps"][phase],
                )
                self.assertTrue(
                    all(row["eligible_step_population"] == eligible for row in phase_targets)
                )
                self.assertTrue(
                    all(
                        row["history_selection"] == "EVERY_OPTIMIZER_PRE_STEP"
                        for row in phase_targets
                    )
                )

        for left in range(len(observed_slices)):
            for right in range(left + 1, len(observed_slices)):
                self.assertTrue(observed_slices[left].isdisjoint(observed_slices[right]))


if __name__ == "__main__":
    unittest.main()
