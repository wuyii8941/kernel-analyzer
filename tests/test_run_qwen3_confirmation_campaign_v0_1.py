from __future__ import annotations

import json
import unittest
from pathlib import Path

from theory_oracle.run_qwen3_confirmation_campaign_v0_1 import (
    preflight_manifest,
    row_to_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class Qwen3ConfirmationCampaignTests(unittest.TestCase):
    def test_uninstantiated_template_is_rejected_before_collection(self) -> None:
        path = (
            ROOT
            / "theory_oracle"
            / "QWEN3_BIAS_ORACLE_CONFIRMATION_MANIFEST_TEMPLATE_V0_1.json"
        )
        _, rows, errors = preflight_manifest(path)
        self.assertFalse(rows)
        self.assertTrue(errors)

    def test_manifest_row_maps_exactly_to_execution_spec(self) -> None:
        row = {
            "trajectory_id": "confirmation-v0-007",
            "source_config_path": "/tmp/config.yaml",
            "capture_plan_path": "/tmp/plan.json",
            "results_root": "/tmp/results",
            "data_root": "/tmp/data",
        }
        spec = row_to_spec(7, row)
        self.assertEqual(spec.trajectory_id, row["trajectory_id"])
        self.assertEqual(spec.config, Path(row["source_config_path"]))
        self.assertEqual(spec.plan, Path(row["capture_plan_path"]))
        self.assertEqual(spec.results_root, Path(row["results_root"]))
        self.assertEqual(spec.data_root, Path(row["data_root"]))


if __name__ == "__main__":
    unittest.main()
