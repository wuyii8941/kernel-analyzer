from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase0_grpo_train import load_transition_capture_plan  # noqa: E402


class MultiTransitionCapturePlanTests(unittest.TestCase):
    def write_plan(self, directory: Path, targets: list[dict]) -> Path:
        path = directory / "plan.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "forkcert.multi-transition-capture-plan.v0.1",
                    "capture_root": "captures",
                    "targets": targets,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_plan_is_sorted_and_paths_stay_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plan = self.write_plan(
                directory,
                [
                    {"optimizer_step": 5, "state_id": "s5", "relative_dir": "step5"},
                    {"optimizer_step": 2, "state_id": "s2", "relative_dir": "step2"},
                ],
            )
            rows = load_transition_capture_plan(plan)
            self.assertEqual([row["optimizer_step"] for row in rows], [2, 5])
            self.assertEqual(rows[0]["capture_dir"], directory / "captures" / "step2")
            self.assertEqual(len({row["plan_digest"] for row in rows}), 1)
            self.assertEqual(
                {row["history_selection"] for row in rows},
                {"FINAL_POLICY_ITERATION_ONLY"},
            )

    def test_population_plan_can_capture_every_optimizer_pre_step(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            plan = self.write_plan(
                Path(raw),
                [
                    {
                        "optimizer_step": 10,
                        "state_id": "s10",
                        "history_selection": "EVERY_OPTIMIZER_PRE_STEP",
                    }
                ],
            )
            rows = load_transition_capture_plan(plan)
            self.assertEqual(rows[0]["optimizer_step"], 10)
            self.assertEqual(rows[0]["history_selection"], "EVERY_OPTIMIZER_PRE_STEP")

    def test_duplicate_steps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            plan = self.write_plan(
                Path(raw),
                [
                    {"optimizer_step": 2, "state_id": "a"},
                    {"optimizer_step": 2, "state_id": "b"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "optimizer steps must be unique"):
                load_transition_capture_plan(plan)

    def test_relative_directory_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            plan = self.write_plan(
                Path(raw),
                [{"optimizer_step": 2, "state_id": "a", "relative_dir": "../escape"}],
            )
            with self.assertRaisesRegex(ValueError, "escapes capture_root"):
                load_transition_capture_plan(plan)


if __name__ == "__main__":
    unittest.main()
