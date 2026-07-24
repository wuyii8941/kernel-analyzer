from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.run_qwen3_calibration_campaign_v0_1 import (
    TrajectorySpec,
    boundary_command,
    capture_audit_valid,
    count_valid_records,
    frozen_specs,
    multi_null_control_command,
    multi_boundary_command,
    null_control_command,
    partial_source_paths,
    sha256_file,
    validate_frozen_spec,
)


class Qwen3CalibrationCampaignTests(unittest.TestCase):
    def test_all_frozen_specs_match_their_plans(self) -> None:
        self.assertEqual(
            [error for spec in frozen_specs() for error in validate_frozen_spec(spec)],
            [],
        )

    def test_record_count_requires_valid_and_population_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {
                "targets": [
                    {"optimizer_step": 1},
                    {"optimizer_step": 2},
                    {"optimizer_step": 3},
                ]
            }
            for step, payload in (
                (1, {"valid": True, "population_eligible": True}),
                (2, {"valid": True, "population_eligible": False}),
                (3, {"valid": False, "population_eligible": True}),
            ):
                directory = root / f"step{step:03d}"
                directory.mkdir()
                (directory / "record_validation.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            self.assertEqual(count_valid_records(plan, root), 1)

    def test_capture_audit_is_bound_to_frozen_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}\n", encoding="utf-8")
            spec = TrajectorySpec(
                index=0,
                trajectory_id="calibration-0",
                config=root / "config.yaml",
                plan=plan,
                results_root=root / "results",
                data_root=root / "data",
            )
            spec.results_root.mkdir()
            spec.config.write_text("config\n", encoding="utf-8")
            metadata = spec.results_root / "source_dump.metadata.json"
            metadata.write_text("{}\n", encoding="utf-8")
            spec.capture_audit.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "verdict": "VALID",
                        "plan_sha256": sha256_file(plan),
                        "capture_root": str(spec.data_root / "captures"),
                        "source_evidence": {
                            "config_sha256": sha256_file(spec.config),
                            "metadata_sha256": sha256_file(metadata),
                            "checks": {"bound": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(capture_audit_valid(spec))
            plan.write_text('{"changed": true}\n', encoding="utf-8")
            self.assertFalse(capture_audit_valid(spec))

    def test_partial_source_cannot_be_silently_restarted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}\n", encoding="utf-8")
            spec = TrajectorySpec(
                index=1,
                trajectory_id="calibration-1",
                config=root / "config.yaml",
                plan=plan,
                results_root=root / "results",
                data_root=root / "data",
            )
            spec.results_root.mkdir()
            path = spec.results_root / "source_dump.jsonl"
            path.write_text("partial\n", encoding="utf-8")
            self.assertEqual(partial_source_paths(spec), [path])

    def test_null_control_command_is_strict_per_trajectory_postprocessing(self) -> None:
        spec = frozen_specs()[0]
        command = null_control_command(spec)
        self.assertIn(
            "theory_oracle/aggregate_qwen3_calibration_null_controls_v0_1.py",
            command,
        )
        self.assertEqual(command[command.index("--plan") + 1], str(spec.plan))
        self.assertEqual(
            command[command.index("--results-root") + 1], str(spec.results_root)
        )
        self.assertEqual(
            command[command.index("--out") + 1], str(spec.null_control_summary)
        )

    def test_multi_null_control_command_requires_all_four_summaries(self) -> None:
        specs = frozen_specs()
        out = Path("/tmp/four-null-controls.json")
        command = multi_null_control_command(specs, out)
        observed = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--trajectory-summary"
        ]
        self.assertEqual(observed, [str(spec.null_control_summary) for spec in specs])
        self.assertEqual(command[-2:], ["--out", str(out)])

    def test_boundary_commands_freeze_calibration_only_tau_grid(self) -> None:
        specs = frozen_specs()
        command = boundary_command(specs[0])
        self.assertIn(
            "theory_oracle/aggregate_qwen3_boundary_conditioned_calibration_v0_1.py",
            command,
        )
        self.assertEqual(command[command.index("--plan") + 1], str(specs[0].plan))
        self.assertEqual(
            command[command.index("--out") + 1], str(specs[0].boundary_summary)
        )
        out = Path("/tmp/four-boundary.json")
        multi = multi_boundary_command(specs, out)
        observed = [
            multi[index + 1]
            for index, value in enumerate(multi)
            if value == "--trajectory-summary"
        ]
        self.assertEqual(observed, [str(spec.boundary_summary) for spec in specs])


if __name__ == "__main__":
    unittest.main()
