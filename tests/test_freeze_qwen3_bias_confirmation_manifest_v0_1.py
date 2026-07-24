from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from theory_oracle.freeze_qwen3_bias_confirmation_manifest_v0_1 import (
    build_manifest,
    sha256_file,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    EXPECTED_CALIBRATION_ANALYSIS_FILES,
    EXPECTED_ENDPOINT_ROLE_CATALOG,
    plan_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class FreezeConfirmationManifestTests(unittest.TestCase):
    def test_materialized_bank_freezes_to_self_validating_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = {
                "valid": True,
                "construction": {"trajectories": 4, "top_level_df": 3},
                "analysis_code": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in EXPECTED_CALIBRATION_ANALYSIS_FILES.items()
                },
                "endpoints": {
                    "U1_reference_aligned_shift": {
                        "status": "COMPLETE_FOUR_TRAJECTORY_CALIBRATION_DESCRIPTION",
                        "endpoint_class": "SIGNED_UPDATE_GEOMETRY_ENDPOINT",
                        "trajectory_rows": [
                            {"trajectory_id": f"calibration-{index}", "mean_effect": value}
                            for index, value in enumerate((-0.03, -0.01, 0.01, 0.03))
                        ],
                    }
                },
            }
            spec = {
                "schema_version": "forkcert.bias-oracle-confirmation-precision-spec.v0.1",
                "status": "FROZEN_BEFORE_CONFIRMATION",
                "endpoint_family": ["U1_reference_aligned_shift"],
                "phase_conditioned_endpoint_family": [],
                "endpoint_role_catalog": EXPECTED_ENDPOINT_ROLE_CATALOG,
                "family_alpha": 0.05,
                "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
                "variance_upper_confidence": 0.8,
                "minimum_confirmation_trajectories": 8,
                "resource_cap": 32,
                "tail": {"scope": "REGULARITY_CONDITIONAL_ONLY"},
                "endpoints": {
                    "U1_reference_aligned_shift": {
                        "desired_half_width": 1.0,
                        "variance_floor_sd": 0.01,
                        "shift_existence_floor": 0.0,
                        "threshold_sources": {
                            "desired_half_width": {
                                "kind": "INDEPENDENT_MEASUREMENT_RESOLUTION",
                                "description": "unit-test desired-width contract",
                                "selection_rule": "fixed before calibration effects are inspected",
                                "uses_calibration_candidate_mean_or_sign": False,
                            },
                            "variance_floor_sd": {
                                "kind": "NEGATIVE_CONTROL_ENVELOPE",
                                "description": "unit-test variance-floor contract",
                                "selection_rule": "fixed before calibration effects are inspected",
                                "uses_calibration_candidate_mean_or_sign": False,
                            },
                            "shift_existence_floor": {
                                "kind": "EXACT_ZERO_NULL",
                                "description": "unit-test shift-floor contract",
                                "selection_rule": "fixed before calibration effects are inspected",
                                "uses_calibration_candidate_mean_or_sign": False,
                            },
                        },
                    }
                },
            }
            calibration_path = root / "calibration.json"
            spec_path = root / "spec.json"
            write_json(calibration_path, calibration)
            write_json(spec_path, spec)
            precision = plan_confirmation(calibration, spec)
            self.assertTrue(precision["valid"], precision["errors"])
            self.assertEqual(precision["planned_confirmation_trajectories"], 8)
            precision["inputs"] = {
                "calibration": {
                    "path": str(calibration_path),
                    "sha256": sha256_file(calibration_path),
                },
                "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            }
            precision_path = root / "precision.json"
            write_json(precision_path, precision)
            bank_path = root / "bank.json"
            materialize = [
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
                str(precision_path),
                "--config-dir",
                str(root / "configs"),
                "--plan-dir",
                str(root / "plans"),
                "--results-root",
                str(root / "results"),
                "--data-root",
                str(root / "data"),
                "--out",
                str(bank_path),
            ]
            completed = subprocess.run(
                materialize, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest_path = root / "manifest.json"
            freeze = [
                sys.executable,
                str(
                    ROOT
                    / "theory_oracle"
                    / "freeze_qwen3_bias_confirmation_manifest_v0_1.py"
                ),
                "--precision",
                str(precision_path),
                "--bank",
                str(bank_path),
                "--out",
                str(manifest_path),
            ]
            completed = subprocess.run(
                freeze, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FROZEN_BEFORE_CONFIRMATION")
            self.assertEqual(len(manifest["trajectory_inputs"]), 8)
            from theory_oracle.run_qwen3_confirmation_campaign_v0_1 import (
                preflight_manifest,
            )

            _, rows, errors = preflight_manifest(manifest_path)
            self.assertFalse(errors)
            self.assertEqual(len(rows), 8)

    def test_outcome_before_freeze_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            precision = {
                "schema_version": "forkcert.bias-oracle-confirmation-precision.v0.1",
                "valid": True,
                "verdict": "VALID_FROZEN_PRECISION_PLAN",
                "planned_confirmation_trajectories": 8,
                "multiplicity": {
                    "endpoint_family": ["U1_reference_aligned_shift"],
                    "method": "NAMED_ENDPOINTS_NO_JOINT_CLAIM_TWO_SIDED",
                    "per_interval_alpha": 0.05,
                },
                "tail": {"scope": "REGULARITY_CONDITIONAL_ONLY"},
                "sensitivity": {},
            }
            precision_path = root / "precision.json"
            write_json(precision_path, precision)
            rows = [
                {
                    "results_root": str(root / f"results-{index}"),
                }
                for index in range(8)
            ]
            bank = {
                "schema_version": "forkcert.qwen3-bias-oracle-confirmation-bank.v0.1",
                "valid": True,
                "verdict": "VALID_FROZEN_CONFIRMATION_TRAJECTORY_BANK",
                "precision": {
                    "sha256": sha256_file(precision_path),
                    "planned_confirmation_trajectories": 8,
                },
                "trajectory_specs": rows,
            }
            bank_path = root / "bank.json"
            write_json(bank_path, bank)
            result_root = Path(rows[0]["results_root"])
            result_root.mkdir()
            (result_root / "source_dump.metadata.json").write_text(
                "outcome", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "outcome exists before manifest freeze"):
                build_manifest(precision, precision_path, bank, bank_path)


if __name__ == "__main__":
    unittest.main()
