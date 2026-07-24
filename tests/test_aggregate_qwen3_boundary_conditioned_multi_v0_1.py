from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.aggregate_qwen3_boundary_conditioned_multi_v0_1 import combine
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    SCRIPT_PATH as ONE_SCRIPT,
    WEIGHTING_CONTRACT_ID,
    sha256_file,
)


class BoundaryConditionedMultiTests(unittest.TestCase):
    def test_combination_preserves_trajectory_as_top_level_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, effect in enumerate((1.0, -1.0, 2.0, -2.0)):
                path = root / f"trajectory-{index}.json"
                value = {
                    "status": "COMPLETE_CALIBRATION_BOUNDARY_DESCRIPTION",
                    "construction": {
                        "tau_grid": [0.01],
                        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                        "reference_anchor_identification": {
                            "observed_states_with_exact_reference_scorer_logps_across_repeats": 24,
                            "observed_states_total": 24,
                            "all_observed_states_exact": True,
                        },
                    },
                    "plan": {
                        "path": str(root / f"plan-{index}.json"),
                        "sha256": f"plan-{index}",
                    },
                    "analysis_code": {
                        "boundary_aggregator": {
                            "path": str(ONE_SCRIPT),
                            "sha256": sha256_file(ONE_SCRIPT),
                        }
                    },
                    "states": [
                        {"trajectory_id": f"calibration-{index}"} for _ in range(24)
                    ],
                    "aggregate": {
                        "tau_profiles": {
                            "0.01": {
                                "states_with_exposure": 4,
                                "state_weighted_mean_margin_shift": effect,
                                "state_weighted_directional_event_shift": 0.0,
                                "state_weighted_semantic_disagreement": 0.1,
                                "phase_balanced_state_weighted_mean_margin_shift": effect,
                                "phase_balanced_directional_event_shift": 0.0,
                                "phase_balanced_semantic_disagreement": 0.1,
                                "all_phase_conditionals_identified": True,
                                "phase_profiles": {
                                    phase: {
                                        "sampled_states": 8,
                                        "states_with_exposure": 2,
                                        "total_exposures_descriptive_only": 4,
                                    }
                                    for phase in ("early", "middle", "late")
                                },
                            }
                        }
                    },
                }
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            result = combine(paths)
            self.assertEqual(result["construction"]["top_level_df_for_future_inference"], 3)
            self.assertEqual(
                result["tau_profiles"]["0.01"][
                    "calibration_mean_of_trajectory_margin_shifts"
                ],
                0.0,
            )
            self.assertFalse(result["claims_allowed"]["population_conditional_B"])
            self.assertEqual(
                result["construction"]["weighting_contract_id"],
                WEIGHTING_CONTRACT_ID,
            )
            self.assertTrue(
                result["construction"]["reference_anchor_identification"][
                    "all_four_trajectories_exact"
                ]
            )

    def test_nonexact_reference_anchor_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(4):
                path = root / f"trajectory-{index}.json"
                anchor_exact = index != 2
                value = {
                    "status": "COMPLETE_CALIBRATION_BOUNDARY_DESCRIPTION",
                    "construction": {
                        "tau_grid": [0.01],
                        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                        "reference_anchor_identification": {
                            "observed_states_with_exact_reference_scorer_logps_across_repeats": 24 if anchor_exact else 23,
                            "observed_states_total": 24,
                            "all_observed_states_exact": anchor_exact,
                        },
                    },
                    "plan": {"path": "plan", "sha256": "plan"},
                    "analysis_code": {
                        "boundary_aggregator": {
                            "path": str(ONE_SCRIPT),
                            "sha256": sha256_file(ONE_SCRIPT),
                        }
                    },
                    "states": [
                        {"trajectory_id": f"calibration-{index}"} for _ in range(24)
                    ],
                    "aggregate": {"tau_profiles": {}},
                }
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "reference anchor"):
                combine(paths)


if __name__ == "__main__":
    unittest.main()
