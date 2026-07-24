from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.freeze_qwen3_boundary_condition_family_v0_1 import (
    CALIBRATION_SCHEMA,
    CALIBRATION_STATUS,
    CANDIDATE_TAUS,
    EXPECTED_ANALYSIS,
    freeze_family,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


def complete_calibration() -> dict:
    tau_profiles = {}
    for tau in CANDIDATE_TAUS:
        supported = tau >= 0.01
        tau_profiles[str(tau)] = {
            "trajectory_rows": [
                {
                    "trajectory_id": f"calibration-{index}",
                    "mean_margin_shift": 1000.0 if tau == 0.0001 else -1000.0,
                    "directional_event_shift": 1.0,
                    "reference_side_phase_support": {
                        phase: {
                            "states_with_exposure": 2 if supported else 1,
                        }
                        for phase in ("early", "middle", "late")
                    },
                }
                for index in range(4)
            ]
        }
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "valid": True,
        "status": CALIBRATION_STATUS,
        "construction": {
            "trajectories": 4,
            "states": 96,
            "tau_grid": CANDIDATE_TAUS,
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "reference_anchor_identification": {
                "all_four_trajectories_exact": True,
                "trajectories_with_all_24_exact": 4,
                "exact_states": 96,
                "observed_states": 96,
            },
        },
        "analysis_code": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in EXPECTED_ANALYSIS.items()
        },
        "tau_profiles": tau_profiles,
    }


class FreezeBoundaryFamilyTests(unittest.TestCase):
    def test_selection_uses_support_and_retains_all_supported_taus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(json.dumps(complete_calibration()), encoding="utf-8")
            result = freeze_family(json.loads(path.read_text()), path)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["retained_taus"], [0.01, 0.05])
        self.assertEqual(result["confirmatory_comparisons"], 6)
        self.assertEqual(
            result["disagreement_endpoint_role"],
            "CONFIRMABLE_NONNEGATIVE_SEMANTIC_IMPACT_NOT_B",
        )
        self.assertEqual(result["weighting_contract_id"], WEIGHTING_CONTRACT_ID)
        self.assertFalse(
            result["uses_candidate_margin_shift_or_event_effect_for_selection"]
        )
        self.assertEqual(
            result["support_audit"]["0.0001"]["candidate_effect_fields_read"],
            [],
        )

    def test_analysis_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            value = complete_calibration()
            value["analysis_code"]["multi_trajectory_aggregator"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            result = freeze_family(value, path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "INVALID_BOUNDARY_FAMILY_FREEZE")

    def test_weighting_contract_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            value = complete_calibration()
            value["construction"]["weighting_contract_id"] = "EXPOSURE_POOLED"
            path.write_text(json.dumps(value), encoding="utf-8")
            result = freeze_family(value, path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("weighting contract" in error for error in result["errors"]))

    def test_reference_anchor_inexactness_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            value = complete_calibration()
            value["construction"]["reference_anchor_identification"][
                "all_four_trajectories_exact"
            ] = False
            value["construction"]["reference_anchor_identification"][
                "exact_states"
            ] = 95
            path.write_text(json.dumps(value), encoding="utf-8")
            result = freeze_family(value, path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("reference anchor" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
