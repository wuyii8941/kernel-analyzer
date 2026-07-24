from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from theory_oracle.materialize_qwen3_bias_confirmation_bank_v0_1 import (
    eligible_offsets,
    ranked_offsets,
    selected_steps,
    trajectory_seeds,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads(
    (
        ROOT
        / "theory_oracle"
        / "QWEN3_BIAS_ORACLE_CONFIRMATION_BANK_DESIGN_V0_1.json"
    ).read_text(encoding="utf-8")
)


class ConfirmationBankMaterializationTests(unittest.TestCase):
    def test_cli_materializes_only_the_frozen_first_j(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            precision = root / "precision.json"
            precision.write_text(
                json.dumps(
                    {
                        "schema_version": "forkcert.bias-oracle-confirmation-precision.v0.1",
                        "valid": True,
                        "verdict": "VALID_FROZEN_PRECISION_PLAN",
                        "planned_confirmation_trajectories": 8,
                    }
                ),
                encoding="utf-8",
            )
            out = root / "bank.json"
            command = [
                sys.executable,
                str(
                    ROOT
                    / "theory_oracle"
                    / "materialize_qwen3_bias_confirmation_bank_v0_1.py"
                ),
                "--design",
                str(
                    ROOT
                    / "theory_oracle"
                    / "QWEN3_BIAS_ORACLE_CONFIRMATION_BANK_DESIGN_V0_1.json"
                ),
                "--precision",
                str(precision),
                "--config-dir",
                str(root / "configs"),
                "--plan-dir",
                str(root / "plans"),
                "--results-root",
                str(root / "results"),
                "--data-root",
                str(root / "data"),
                "--out",
                str(out),
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            bank = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(bank["trajectory_specs"]), 8)
            self.assertEqual(
                bank["selection_audit"]["selected_offsets_in_order"],
                ranked_offsets(DESIGN)[:8],
            )
            self.assertTrue(
                all(
                    Path(row["source_config_path"]).is_file()
                    and Path(row["capture_plan_path"]).is_file()
                    for row in bank["trajectory_specs"]
                )
            )
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)

    def test_population_excludes_all_calibration_blocks(self) -> None:
        offsets = eligible_offsets(DESIGN)
        self.assertEqual(len(offsets), 108)
        self.assertTrue(
            {3200, 3840, 5696, 7296}.isdisjoint(offsets)
        )

    def test_ranking_and_seeds_are_deterministic_and_unique(self) -> None:
        self.assertEqual(ranked_offsets(DESIGN), ranked_offsets(copy.deepcopy(DESIGN)))
        seeds = trajectory_seeds(DESIGN, 32)
        self.assertEqual(seeds, trajectory_seeds(copy.deepcopy(DESIGN), 32))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(
            set(DESIGN["calibration_seeds_excluded"]).isdisjoint(seeds)
        )

    def test_each_trajectory_has_eight_fixed_steps_per_phase(self) -> None:
        first = selected_steps(DESIGN, "confirmation-v0-000")
        second = selected_steps(DESIGN, "confirmation-v0-001")
        self.assertEqual(first, selected_steps(DESIGN, "confirmation-v0-000"))
        self.assertNotEqual(first, second)
        for phase, bounds in DESIGN["state_selection"]["phases"].items():
            self.assertEqual(len(first[phase]), 8)
            self.assertEqual(len(set(first[phase])), 8)
            self.assertTrue(
                all(bounds[0] <= step <= bounds[1] for step in first[phase])
            )


if __name__ == "__main__":
    unittest.main()
