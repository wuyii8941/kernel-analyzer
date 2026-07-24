from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
    SCHEMA_VERSION as BOUNDARY_SCHEMA_VERSION,
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.analyze_qwen3_partial_boundary_influence_v0_1 import analyze


class PartialBoundaryInfluenceTests(unittest.TestCase):
    def test_one_state_can_flip_phase_mean_without_semantic_flip(self) -> None:
        summary = {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "construction": {
                "all_eligible_weighting_contract_id": ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
                "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                "tau_grid": [0.01],
            },
            "states": [
                {
                    "state_id": "middle-negative",
                    "phase": "middle",
                    "all_eligible_mean_margin_shift": -1.0,
                    "tau_profiles": {
                        "0.01": {
                            "exposures": 1,
                            "mean_margin_shift": -1.0,
                            "directional_event_shift": 0.0,
                            "semantic_disagreement": 0.0,
                        }
                    },
                },
                {
                    "state_id": "middle-positive-large",
                    "phase": "middle",
                    "all_eligible_mean_margin_shift": 2.0,
                    "tau_profiles": {
                        "0.01": {
                            "exposures": 1,
                            "mean_margin_shift": 2.0,
                            "directional_event_shift": 0.0,
                            "semantic_disagreement": 0.0,
                        }
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            result = analyze(summary, path)
        profile = result["phase_profiles"]["middle"][
            "all_eligible_mean_margin_shift"
        ]
        self.assertEqual(profile["full_sample_sign"], 1)
        self.assertEqual(profile["leave_one_out_signs"], [-1, 1])
        self.assertFalse(profile["sign_stable_to_any_single_state_deletion"])
        disagreement = result["phase_profiles"]["middle"]["tau_profiles"]["0.01"][
            "semantic_disagreement"
        ]
        self.assertEqual(disagreement["full_sample_mean"], 0.0)
        self.assertTrue(disagreement["sign_stable_to_any_single_state_deletion"])
        self.assertFalse(disagreement["stability_interpretable"])
        self.assertFalse(result["construction"]["population_inference_allowed"])

    def test_nonfinite_boundary_value_is_invalid_not_missing_support(self) -> None:
        summary = {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "construction": {
                "all_eligible_weighting_contract_id": ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
                "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                "tau_grid": [0.01],
            },
            "states": [
                {
                    "state_id": "bad-state",
                    "phase": "early",
                    "all_eligible_endpoint_status": "IDENTIFIED",
                    "all_eligible_mean_margin_shift": math.nan,
                    "tau_profiles": {
                        "0.01": {
                            "exposures": 1,
                            "mean_margin_shift": math.inf,
                            "directional_event_shift": 0.0,
                            "semantic_disagreement": 0.0,
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text("corrupt nonfinite fixture", encoding="utf-8")
            result = analyze(summary, path)
        self.assertFalse(result["valid"])
        self.assertIn(
            "bad-state: nonfinite all-eligible margin shift", result["errors"]
        )
        self.assertIn(
            "bad-state: nonfinite mean_margin_shift at tau=0.01",
            result["errors"],
        )

    def test_all_eligible_weighting_contract_drift_is_invalid(self) -> None:
        summary = {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "construction": {
                "all_eligible_weighting_contract_id": "EXPOSURE_WEIGHTED_OTHER_ESTIMAND",
                "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                "tau_grid": [],
            },
            "states": [{"state_id": "state-a", "phase": "early"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            result = analyze(summary, path)
        self.assertFalse(result["valid"])
        self.assertIn("all-eligible weighting contract drifted", result["errors"])


if __name__ == "__main__":
    unittest.main()
