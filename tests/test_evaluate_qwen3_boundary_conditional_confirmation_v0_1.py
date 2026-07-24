from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    FIXED_RESOURCE_EXISTENCE,
)
from theory_oracle.evaluate_qwen3_boundary_conditional_confirmation_v0_1 import (
    EXPECTED_ANALYSIS,
    SPEC_VERSION,
    collect_boundary_effect_records,
    evaluate_endpoint_records,
    validate_boundary_spec,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.freeze_qwen3_boundary_condition_family_v0_1 import (
    SCHEMA_VERSION as FAMILY_SCHEMA_VERSION,
)
from theory_oracle.plan_qwen3_boundary_confirmation_resource_v0_1 import (
    SCHEMA_VERSION as RESOURCE_PLAN_SCHEMA,
    SCRIPT_PATH as RESOURCE_PLANNER_PATH,
)


def state_rows(missing_last_late_state: bool = False) -> list[dict]:
    rows = []
    for trajectory in range(8):
        for phase in ("early", "middle", "late"):
            count = 1 if missing_last_late_state and trajectory == 7 and phase == "late" else 2
            for state in range(count):
                rows.append(
                    {
                        "trajectory_id": f"confirmation-{trajectory}",
                        "phase": phase,
                        "state_id": f"confirmation-{trajectory}-{phase}-{state}",
                        "reference_anchor_stability": {
                            "reference_scorer_logps_exact_across_repeats": True,
                            "formal_confirmation_allowed": True,
                        },
                        "repeat_profiles": [
                            {
                                "repeat_id": repeat,
                                "tau_profiles": {
                                    "0.01": {
                                        "exposures": 4,
                                        "condition_mask_sha256": f"mask-{trajectory}-{phase}-{state}",
                                        "mean_margin_shift": 1.0,
                                        "directional_event_shift": 0.5,
                                        "semantic_disagreement": 0.5,
                                    }
                                },
                            }
                            for repeat in (1, 2)
                        ],
                    }
                )
    return rows


class BoundaryConditionalConfirmationTests(unittest.TestCase):
    def test_stable_conditional_effect_can_be_confirmed_as_its_own_estimand(self) -> None:
        endpoint = "boundary_margin_shift::tau=0.01"
        records, support, errors = collect_boundary_effect_records(
            state_rows(), endpoint
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 8 * 3 * 2 * 2)
        result = evaluate_endpoint_records(
            records,
            endpoint_plan={
                "planning_mode": FIXED_RESOURCE_EXISTENCE,
                "reference_anchor_protocol": {
                    "mode": "DETERMINISTIC_REFERENCE_WITH_REPEAT_EXACTNESS_GATE",
                    "reference_conditional_output_is_point_mass_assumption": True,
                    "two_repeat_exactness_is_diagnostic_not_proof": True,
                    "fail_if_reference_repeats_differ": True,
                    "fallback_requires_independent_anchor_execution": True,
                },
                "shift_existence_floor": 0.0,
            },
            interval_alpha=0.025,
            planned_trajectories=8,
        )
        self.assertEqual(
            result["final_shift_verdict"], "REPRODUCIBLE_AVERAGE_SHIFT"
        )
        self.assertTrue(result["operator_attribution_eligibility"]["eligible"])
        self.assertEqual(
            support["condition_anchor"],
            "REFERENCE_REPEAT_1_MARGIN_MASK_WITH_EXACT_REFERENCE_REPEATS;_"
            "STOCHASTIC_REFERENCE_REQUIRES_INDEPENDENT_ANCHOR_PROTOCOL",
        )
        self.assertEqual(
            len(support["reference_anchor_condition_mask_sha256_by_state"]),
            8 * 3 * 2,
        )

    def test_missing_two_state_phase_support_fails_closed(self) -> None:
        _, _, errors = collect_boundary_effect_records(
            state_rows(missing_last_late_state=True),
            "boundary_clip_directional_shift::tau=0.01",
        )
        self.assertTrue(any("confirmation-7/late" in error for error in errors))

    def test_disagreement_is_confirmed_as_semantic_impact_not_bias(self) -> None:
        endpoint = "boundary_semantic_disagreement::tau=0.01"
        records, _, errors = collect_boundary_effect_records(state_rows(), endpoint)
        self.assertEqual(errors, [])
        result = evaluate_endpoint_records(
            records,
            endpoint=endpoint,
            endpoint_plan={
                "planning_mode": FIXED_RESOURCE_EXISTENCE,
                "semantic_impact_existence_floor": 0.0,
            },
            interval_alpha=0.025,
            planned_trajectories=8,
        )
        self.assertEqual(
            result["final_semantic_impact_verdict"],
            "REPRODUCIBLE_SEMANTIC_DISAGREEMENT",
        )
        self.assertTrue(result["operator_attribution_eligibility"]["eligible"])
        self.assertTrue(
            result["semantic_impact_estimate"]["semantic_impact_is_not_B"]
        )
        self.assertNotIn("estimate", result)
        self.assertNotIn("B", result["semantic_impact_estimate"])

    def test_disagreement_outside_probability_range_fails(self) -> None:
        rows = state_rows()
        rows[0]["repeat_profiles"][0]["tau_profiles"]["0.01"][
            "semantic_disagreement"
        ] = 1.5
        records, _, errors = collect_boundary_effect_records(
            rows, "boundary_semantic_disagreement::tau=0.01"
        )
        self.assertEqual(records, [])
        self.assertTrue(any("must lie in [0, 1]" in error for error in errors))

    def test_stochastic_reference_anchor_fails_closed(self) -> None:
        rows = state_rows()
        rows[0]["reference_anchor_stability"][
            "reference_scorer_logps_exact_across_repeats"
        ] = False
        rows[0]["reference_anchor_stability"]["formal_confirmation_allowed"] = False
        records, _, errors = collect_boundary_effect_records(rows, "boundary_margin_shift::tau=0.01")
        self.assertEqual(records, [])
        self.assertTrue(any("stochastic reference anchor" in error for error in errors))

    def test_spec_binds_family_manifest_confirmation_bank_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "confirmation_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            family_path = root / "boundary_family.json"
            family = {
                "schema_version": FAMILY_SCHEMA_VERSION,
                "valid": True,
                "status": "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY",
                "endpoint_family": ["boundary_margin_shift::tau=0.01"],
                "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            }
            family_path.write_text(json.dumps(family), encoding="utf-8")
            resource_path = root / "boundary_resource.json"
            resource = {
                "schema_version": RESOURCE_PLAN_SCHEMA,
                "valid": True,
                "status": "VALID_FROZEN_BOUNDARY_RESOURCE_REQUIREMENT",
                "boundary_family": {
                    "path": str(family_path),
                    "sha256": hashlib.sha256(family_path.read_bytes()).hexdigest(),
                },
                "minimum_trajectories_for_signflip_resolution": 8,
                "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
                "family_alpha": 0.05,
                "confirmatory_comparisons": 1,
                "per_interval_alpha": 0.05,
                "candidate_effect_mean_sign_or_variance_used": False,
                "analysis_code": {
                    "path": str(RESOURCE_PLANNER_PATH),
                    "sha256": hashlib.sha256(
                        RESOURCE_PLANNER_PATH.read_bytes()
                    ).hexdigest(),
                },
            }
            resource_path.write_text(json.dumps(resource), encoding="utf-8")
            spec = {
                "schema_version": SPEC_VERSION,
                "status": "FROZEN_BEFORE_CONFIRMATION",
                "planning_mode": FIXED_RESOURCE_EXISTENCE,
                "reference_anchor_protocol": {
                    "mode": "DETERMINISTIC_REFERENCE_WITH_REPEAT_EXACTNESS_GATE",
                    "reference_conditional_output_is_point_mass_assumption": True,
                    "two_repeat_exactness_is_diagnostic_not_proof": True,
                    "fail_if_reference_repeats_differ": True,
                    "fallback_requires_independent_anchor_execution": True,
                },
                "confirmation_manifest": {
                    "path": str(manifest),
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "boundary_family": {
                    "path": str(family_path),
                    "sha256": hashlib.sha256(family_path.read_bytes()).hexdigest(),
                },
                "fixed_resource_source": {
                    "kind": "INHERITED_FROZEN_CONFIRMATION_BANK",
                    "description": "unit-test frozen bank",
                    "selection_rule": "inherit its trajectory count before outcomes",
                    "uses_boundary_confirmation_outcomes": False,
                },
                "boundary_resource_plan": {
                    "path": str(resource_path),
                    "sha256": hashlib.sha256(resource_path.read_bytes()).hexdigest(),
                },
                "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
                "cross_family_joint_claim_allowed": False,
                "family_alpha": 0.05,
                "endpoint_family": family["endpoint_family"],
                "endpoints": {
                    "boundary_margin_shift::tau=0.01": {
                        "shift_existence_floor": 0.0,
                        "shift_existence_floor_source": {
                            "kind": "EXACT_ZERO_NULL",
                            "description": "unit-test exact null",
                            "selection_rule": "zero fixed before outcomes",
                            "uses_calibration_candidate_mean_or_sign": False,
                        },
                    }
                },
                "analysis_code": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in EXPECTED_ANALYSIS.items()
                },
            }
            frozen, errors = validate_boundary_spec(
                spec, family, family_path, manifest
            )
            self.assertEqual(errors, [])
            self.assertEqual(frozen["interval_alpha"], 0.05)
            spec["analysis_code"]["boundary_measurement"]["sha256"] = "0" * 64
            _, errors = validate_boundary_spec(spec, family, family_path, manifest)
            self.assertTrue(any("boundary_measurement" in error for error in errors))

    def test_spec_rejects_unacknowledged_repeat_anchor_assumption(self) -> None:
        # The full valid fixture is exercised above.  Here a structurally incomplete
        # spec must fail before any outcome can be interpreted.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            family_path = root / "family.json"
            family = {
                "schema_version": FAMILY_SCHEMA_VERSION,
                "valid": True,
                "status": "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY",
                "endpoint_family": ["boundary_margin_shift::tau=0.01"],
                "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            }
            family_path.write_text(json.dumps(family), encoding="utf-8")
            _, errors = validate_boundary_spec({}, family, family_path, manifest)
        self.assertTrue(any("reference-anchor identification" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
