from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    FIXED_RESOURCE_EXISTENCE,
)
from theory_oracle.bias_oracle_population_v0_2 import EffectRecord
from theory_oracle.evaluate_qwen3_boundary_conditional_confirmation_v0_1 import (
    SCHEMA_VERSION as BOUNDARY_CONFIRMATION_SCHEMA,
    collect_boundary_effect_records,
    evaluate_endpoint_records,
    parse_endpoint,
)
from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    freeze_bridge,
)


PHASES = ("early", "middle", "late")
TRAJECTORIES = 8
ENDPOINT = "boundary_margin_shift::tau=0.01"


def endpoint_plan() -> dict:
    return {
        "planning_mode": FIXED_RESOURCE_EXISTENCE,
        "shift_existence_floor": 0.0,
    }


def make_counterexample() -> tuple[list[EffectRecord], list[dict]]:
    """Global effects cancel while the predeclared boundary subset stays positive."""
    global_records: list[EffectRecord] = []
    boundary_rows: list[dict] = []
    for trajectory_index in range(TRAJECTORIES):
        trajectory = f"confirmation-{trajectory_index}"
        for phase in PHASES:
            for state_index, effect in enumerate((1.0, 1.0, -1.0, -1.0)):
                state_id = f"{trajectory}-{phase}-{state_index}"
                for repeat_id in (1, 2):
                    global_records.append(
                        EffectRecord(
                            trajectory_id=trajectory,
                            phase=phase,
                            state_id=state_id,
                            repeat_id=repeat_id,
                            effect=effect,
                        )
                    )
                # Membership is fixed from the reference-side margin.  The two
                # +1 states are near the boundary; the cancelling -1 states are not.
                tau_profiles = (
                    {
                        "0.01": {
                            "exposures": 4,
                            "condition_mask_sha256": hashlib.sha256(
                                f"reference-mask::{state_id}".encode()
                            ).hexdigest(),
                            "mean_margin_shift": effect,
                            "directional_event_shift": 0.5,
                            "semantic_disagreement": 0.5,
                        }
                    }
                    if state_index < 2
                    else {}
                )
                boundary_rows.append(
                    {
                        "trajectory_id": trajectory,
                        "phase": phase,
                        "state_id": state_id,
                        "reference_anchor_stability": {
                            "reference_scorer_logps_exact_across_repeats": True,
                            "formal_confirmation_allowed": True,
                        },
                        "repeat_profiles": [
                            {
                                "repeat_id": repeat_id,
                                "tau_profiles": tau_profiles,
                            }
                            for repeat_id in (1, 2)
                        ],
                    }
                )
    return global_records, boundary_rows


class GlobalVsBoundaryConditionalRoutingTests(unittest.TestCase):
    def test_global_cancellation_does_not_block_confirmed_conditional_effect(self) -> None:
        global_records, boundary_rows = make_counterexample()
        global_result = evaluate_endpoint_records(
            global_records,
            endpoint_plan=endpoint_plan(),
            interval_alpha=0.05,
            planned_trajectories=TRAJECTORIES,
        )
        self.assertEqual(global_result["estimate"]["B"]["estimate"], 0.0)
        self.assertEqual(
            global_result["final_shift_verdict"], "NO_STABLE_AVERAGE_DETECTED"
        )
        self.assertFalse(global_result["operator_attribution_eligibility"]["eligible"])

        conditional_records, support, errors = collect_boundary_effect_records(
            boundary_rows, ENDPOINT
        )
        self.assertEqual(errors, [])
        conditional_result = evaluate_endpoint_records(
            conditional_records,
            endpoint_plan=endpoint_plan(),
            interval_alpha=0.05,
            planned_trajectories=TRAJECTORIES,
        )
        self.assertEqual(conditional_result["estimate"]["B"]["estimate"], 1.0)
        self.assertEqual(
            conditional_result["final_shift_verdict"],
            "REPRODUCIBLE_AVERAGE_SHIFT",
        )
        self.assertTrue(
            conditional_result["operator_attribution_eligibility"]["eligible"]
        )
        self.assertIn(
            "not contribution to global B",
            conditional_result["operator_attribution_eligibility"]["claim_scope"],
        )

        state_ids = sorted(
            state_id
            for group in support["exposed_state_ids_by_trajectory_phase"].values()
            for state_id in group
        )
        confirmation = {
            "schema_version": BOUNDARY_CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [ENDPOINT],
                "automatic_operator_launch": False,
            },
            "endpoints": {
                ENDPOINT: {
                    **conditional_result,
                    "support": support,
                }
            },
            "state_evidence": [{"state_id": state_id} for state_id in state_ids],
        }
        with tempfile.TemporaryDirectory() as temporary:
            confirmation_path = Path(temporary) / "boundary-confirmation.json"
            confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
            bridge = freeze_bridge(confirmation, confirmation_path, ENDPOINT)
        self.assertTrue(bridge["valid"], bridge["errors"])
        self.assertEqual(bridge["exposed_state_count"], TRAJECTORIES * 3 * 2)
        self.assertEqual(bridge["target_endpoint"], ENDPOINT)

    def test_disagreement_has_a_distinct_supported_endpoint_identity(self) -> None:
        kind, tau = parse_endpoint("boundary_semantic_disagreement::tau=0.01")
        self.assertEqual(kind, "boundary_semantic_disagreement")
        self.assertEqual(tau, 0.01)

    def test_zero_net_semantic_shift_does_not_block_disagreement_attribution(self) -> None:
        _, rows = make_counterexample()
        rows = deepcopy(rows)
        for row in rows:
            if not row["repeat_profiles"][0]["tau_profiles"]:
                continue
            state_index = int(row["state_id"].rsplit("-", 1)[1])
            directional = 0.5 if state_index == 0 else -0.5
            for repeat in row["repeat_profiles"]:
                profile = repeat["tau_profiles"]["0.01"]
                profile["directional_event_shift"] = directional
                profile["semantic_disagreement"] = 0.5

        signed_endpoint = "boundary_clip_directional_shift::tau=0.01"
        signed_records, _, errors = collect_boundary_effect_records(
            rows, signed_endpoint
        )
        self.assertEqual(errors, [])
        signed_result = evaluate_endpoint_records(
            signed_records,
            endpoint=signed_endpoint,
            endpoint_plan=endpoint_plan(),
            interval_alpha=0.05,
            planned_trajectories=TRAJECTORIES,
        )
        self.assertEqual(signed_result["estimate"]["B"]["estimate"], 0.0)
        self.assertFalse(signed_result["operator_attribution_eligibility"]["eligible"])

        impact_endpoint = "boundary_semantic_disagreement::tau=0.01"
        impact_records, support, errors = collect_boundary_effect_records(
            rows, impact_endpoint
        )
        self.assertEqual(errors, [])
        impact_result = evaluate_endpoint_records(
            impact_records,
            endpoint=impact_endpoint,
            endpoint_plan={
                "planning_mode": FIXED_RESOURCE_EXISTENCE,
                "semantic_impact_existence_floor": 0.0,
            },
            interval_alpha=0.05,
            planned_trajectories=TRAJECTORIES,
        )
        self.assertEqual(
            impact_result["final_semantic_impact_verdict"],
            "REPRODUCIBLE_SEMANTIC_DISAGREEMENT",
        )
        self.assertTrue(impact_result["operator_attribution_eligibility"]["eligible"])
        self.assertNotIn("estimate", impact_result)
        self.assertNotIn("B", impact_result["semantic_impact_estimate"])

        state_ids = sorted(
            state_id
            for group in support["exposed_state_ids_by_trajectory_phase"].values()
            for state_id in group
        )
        confirmation = {
            "schema_version": BOUNDARY_CONFIRMATION_SCHEMA,
            "valid": True,
            "verdict": "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION",
            "operator_attribution_gate": {
                "ready_endpoints": [impact_endpoint],
                "automatic_operator_launch": False,
            },
            "endpoints": {
                impact_endpoint: {**impact_result, "support": support}
            },
            "state_evidence": [{"state_id": state_id} for state_id in state_ids],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "semantic-confirmation.json"
            path.write_text(json.dumps(confirmation), encoding="utf-8")
            bridge = freeze_bridge(confirmation, path, impact_endpoint)
        self.assertTrue(bridge["valid"], bridge["errors"])
        self.assertEqual(
            bridge["target_effect_role"], "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
        )


if __name__ == "__main__":
    unittest.main()
