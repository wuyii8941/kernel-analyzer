from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.evaluate_qwen3_boundary_conditional_confirmation_v0_1 import (
    SCHEMA_VERSION as CONFIRMATION_SCHEMA,
)
from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    freeze_bridge,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


class BoundaryAttributionBridgeTests(unittest.TestCase):
    def test_bridge_freezes_only_confirmed_exposed_state_and_mask_bank(self) -> None:
        endpoint = "boundary_margin_shift::tau=0.01"
        states = ["s-1", "s-2"]
        confirmation = {
            "schema_version": CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [endpoint],
                "automatic_operator_launch": False,
            },
            "endpoints": {
                endpoint: {
                    "final_shift_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
                    "operator_attribution_eligibility": {"eligible": True},
                    "estimate": {"B": {"estimate": 0.25}},
                    "support": {
                        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                        "exposed_state_ids_by_trajectory_phase": {
                            "confirmation-0::early": states,
                        },
                        "reference_anchor_condition_mask_sha256_by_state": {
                            state: str(index) * 64
                            for index, state in enumerate(states, start=1)
                        },
                    },
                }
            },
            "state_evidence": [{"state_id": state} for state in states],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation.json"
            path.write_text(json.dumps(confirmation), encoding="utf-8")
            result = freeze_bridge(confirmation, path, endpoint)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["exposed_state_count"], 2)
        self.assertEqual(result["weighting_contract_id"], WEIGHTING_CONTRACT_ID)
        self.assertEqual(result["frozen_effect_direction"], 1)
        self.assertFalse(result["population_operator_contribution_claim_allowed"])
        self.assertEqual(
            result["state_bank_role"],
            "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT",
        )
        self.assertFalse(
            result["intervention_contract"][
                "condition_mask_recomputed_after_intervention"
            ]
        )
        self.assertFalse(result["automatic_operator_launch"])

    def test_unconfirmed_endpoint_cannot_create_bridge(self) -> None:
        endpoint = "boundary_margin_shift::tau=0.01"
        confirmation = {
            "schema_version": CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [],
                "automatic_operator_launch": False,
            },
            "endpoints": {},
            "state_evidence": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation.json"
            path.write_text(json.dumps(confirmation), encoding="utf-8")
            result = freeze_bridge(confirmation, path, endpoint)
        self.assertFalse(result["valid"])
        self.assertIn(
            "endpoint is absent from boundary confirmation ready gate",
            result["errors"],
        )

    def test_semantic_disagreement_bridge_freezes_reduction_objective_not_bias_direction(self) -> None:
        endpoint = "boundary_semantic_disagreement::tau=0.01"
        states = ["s-1", "s-2"]
        confirmation = {
            "schema_version": CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [endpoint],
                "automatic_operator_launch": False,
            },
            "endpoints": {
                endpoint: {
                    "final_semantic_impact_verdict": "REPRODUCIBLE_SEMANTIC_DISAGREEMENT",
                    "operator_attribution_eligibility": {"eligible": True},
                    "semantic_impact_estimate": {
                        "population_mean_disagreement": {"estimate": 0.25}
                    },
                    "support": {
                        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
                        "exposed_state_ids_by_trajectory_phase": {
                            "confirmation-0::early": states,
                        },
                        "reference_anchor_condition_mask_sha256_by_state": {
                            state: str(index) * 64
                            for index, state in enumerate(states, start=1)
                        },
                    },
                }
            },
            "state_evidence": [{"state_id": state} for state in states],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation.json"
            path.write_text(json.dumps(confirmation), encoding="utf-8")
            result = freeze_bridge(confirmation, path, endpoint)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["frozen_effect_direction"], 1)
        self.assertEqual(
            result["target_effect_role"], "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
        )
        self.assertEqual(
            result["attribution_objective"],
            "REPAIR_REDUCES_OR_INJECTION_INCREASES_SEMANTIC_DISAGREEMENT",
        )

    def test_confirmed_endpoint_with_weighting_drift_cannot_create_bridge(self) -> None:
        endpoint = "boundary_margin_shift::tau=0.01"
        confirmation = {
            "schema_version": CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [endpoint],
                "automatic_operator_launch": False,
            },
            "endpoints": {
                endpoint: {
                    "final_shift_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
                    "operator_attribution_eligibility": {"eligible": True},
                    "estimate": {"B": {"estimate": 0.25}},
                    "support": {
                        "weighting_contract_id": "EXPOSURE_POOLED",
                        "exposed_state_ids_by_trajectory_phase": {
                            "confirmation-0::early": ["s-1"]
                        },
                        "reference_anchor_condition_mask_sha256_by_state": {
                            "s-1": "1" * 64
                        },
                    },
                }
            },
            "state_evidence": [{"state_id": "s-1"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation.json"
            path.write_text(json.dumps(confirmation), encoding="utf-8")
            result = freeze_bridge(confirmation, path, endpoint)
        self.assertFalse(result["valid"])
        self.assertTrue(any("weighting contract" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
