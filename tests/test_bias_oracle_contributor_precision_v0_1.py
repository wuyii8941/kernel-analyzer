from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from theory_oracle.bias_oracle_contributor_precision_v0_1 import (
    plan_contributor_precision,
)


def candidate_freeze(count: int = 1) -> dict:
    return {
        "schema_version": "forkcert.qwen3-bias-contributor-candidate-freeze.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
        "target_endpoint": "U1_reference_aligned_shift",
        "primary_candidates": [
            {"candidate_id": f"candidate-{index}"} for index in range(count)
        ],
    }


def contributor_threshold_sources() -> dict:
    common = {
        "description": "unit-test contributor threshold contract",
        "selection_rule": "fixed before reading contributor pilot effects",
        "uses_pilot_contribution_mean_or_sign": False,
    }
    return {
        "desired_halfwidth": {
            **common,
            "kind": "INDEPENDENT_MEASUREMENT_RESOLUTION",
        },
        "directional_contribution_floor": {
            **common,
            "kind": "EXACT_ZERO_NULL",
        },
        "variance_floor_sd": {
            **common,
            "kind": "NEGATIVE_CONTROL_ENVELOPE",
        },
    }


def spec(count: int = 1) -> dict:
    ids = [f"candidate-{index}" for index in range(count)]
    return {
        "schema_version": "forkcert.qwen3-bias-contributor-precision-input-spec.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PRECISION_PLANNING",
        "target_endpoint": "U1_reference_aligned_shift",
        "frozen_bias_direction": {
            "source": "ENDPOINT_CONFIRMATION_ONLY",
            "kind": "SCALAR_SIGN",
            "scalar_sign": 1,
            "vector_direction_artifact_path": None,
            "vector_direction_artifact_sha256": None,
        },
        "candidate_freeze": {"artifact_path": "candidate.json", "artifact_sha256": "frozen"},
        "precision_basis": {
            "mode": "INDEPENDENT_CONTRIBUTOR_PILOT_VARIANCE_ONLY",
            "contributor_pilot_summary_path": "pilot.json",
            "contributor_pilot_summary_sha256": "frozen",
            "external_fixed_rationale": None,
        },
        "multiplicity": {
            "primary_repair_family_members": ids,
            "method": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
            "family_alpha": 0.05,
            "injection_is_separate_family": True,
        },
        "variance_upper_confidence": 0.8,
        "candidate_precision_inputs": {
            candidate_id: {
                "desired_halfwidth": 0.5,
                "directional_contribution_floor": 0.0,
                "variance_floor_sd": 0.01,
                "threshold_sources": contributor_threshold_sources(),
            }
            for candidate_id in ids
        },
        "global_design": {
            "minimum_confirmation_trajectories": 8,
            "resource_cap_trajectories": 500,
            "tail": {"scope": "REGULARITY_CONDITIONAL_ONLY"},
            "no_optional_stopping_on_mean_or_sign": True,
        },
        "sensitivity": {
            "method": "TRAJECTORY_RADEMACHER_SIGN_FLIP_STUDENTIZED",
            "role": "VETO_PRIMARY_CONTRIBUTION_ONLY",
            "multiplicity_adjusted_alpha_required": True,
            "exact_max_trajectories": 16,
            "monte_carlo_draws": 99999,
            "monte_carlo_seed": 172904,
        },
    }


def pilot(count: int = 1, effects: list[float] | None = None) -> dict:
    ids = [f"candidate-{index}" for index in range(count)]
    effects = effects or [-1.5, -0.5, 0.5, 1.5]
    return {
        "schema_version": "forkcert.qwen3-bias-contributor-pilot-summary.v0.1",
        "valid": True,
        "target_endpoint": "U1_reference_aligned_shift",
        "candidate_ids": ids,
        "construction": {
            "trajectories": len(effects),
            "role": "DISPERSION_ONLY_NO_MEAN_OR_SIGN_SELECTION",
        },
        "trajectory_specs": [
            {
                "trajectory_id": f"pilot-{index}",
                "trajectory_seed": 1000 + index,
                "data_slice_id": f"pilot-slice-{index}",
            }
            for index in range(len(effects))
        ],
        "candidates": {
            candidate_id: {
                "status": "COMPLETE_CONTRIBUTOR_PILOT_DESCRIPTION",
                "trajectory_rows": [
                    {"trajectory_id": f"pilot-{index}", "mean_effect": effect}
                    for index, effect in enumerate(effects)
                ],
            }
            for candidate_id in ids
        },
    }


class ContributorPrecisionPlannerTests(unittest.TestCase):
    def test_candidate_thresholds_without_auditable_source_fail_closed(self) -> None:
        value = spec()
        del value["candidate_precision_inputs"]["candidate-0"]["threshold_sources"]
        result = plan_contributor_precision(candidate_freeze(), value, pilot())
        self.assertFalse(result["valid"])
        self.assertTrue(any("threshold_sources" in error for error in result["errors"]))

    def test_contributor_threshold_source_roles_cannot_be_swapped(self) -> None:
        value = spec()
        sources = value["candidate_precision_inputs"]["candidate-0"][
            "threshold_sources"
        ]
        sources["directional_contribution_floor"]["kind"] = (
            "EXTERNAL_CONSERVATIVE_VARIANCE_FLOOR"
        )
        sources["variance_floor_sd"]["kind"] = "EXACT_ZERO_NULL"
        result = plan_contributor_precision(candidate_freeze(), value, pilot())
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "unsupported directional_contribution_floor" in error
                for error in result["errors"]
            )
        )
        self.assertTrue(
            any("unsupported variance_floor_sd" in error for error in result["errors"])
        )

    def test_valid_pilot_plan_excludes_mean_and_sign(self) -> None:
        result = plan_contributor_precision(candidate_freeze(), spec(), pilot())
        self.assertTrue(result["valid"], result["errors"])
        row = result["candidate_precision"]["candidate-0"]
        self.assertFalse(row["pilot_mean_or_sign_used_for_sizing"])
        self.assertNotIn("mean", row)
        self.assertGreaterEqual(result["global_design"]["planned_confirmation_trajectories"], 8)
        self.assertEqual(
            result["injection_precision"]["status"],
            "UNINSTANTIATED_SEPARATE_PRECISION_PLAN_REQUIRED",
        )

    def test_translation_and_sign_do_not_change_plan(self) -> None:
        first = plan_contributor_precision(candidate_freeze(), spec(), pilot())
        shifted = plan_contributor_precision(
            candidate_freeze(), spec(), pilot(effects=[8.5, 9.5, 10.5, 11.5])
        )
        reversed_result = plan_contributor_precision(
            candidate_freeze(), spec(), pilot(effects=[1.5, 0.5, -0.5, -1.5])
        )
        self.assertEqual(
            first["global_design"]["planned_confirmation_trajectories"],
            shifted["global_design"]["planned_confirmation_trajectories"],
        )
        self.assertEqual(
            first["global_design"]["planned_confirmation_trajectories"],
            reversed_result["global_design"]["planned_confirmation_trajectories"],
        )

    def test_zero_pilot_variance_without_floor_fails(self) -> None:
        value = spec()
        value["candidate_precision_inputs"]["candidate-0"]["variance_floor_sd"] = 0.0
        result = plan_contributor_precision(
            candidate_freeze(), value, pilot(effects=[0.0, 0.0, 0.0, 0.0])
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("zero planning scale" in error for error in result["errors"]))

    def test_ten_candidate_multiplicity_raises_signflip_minimum_to_nine(self) -> None:
        value = spec(10)
        for row in value["candidate_precision_inputs"].values():
            row["desired_halfwidth"] = 100.0
        result = plan_contributor_precision(candidate_freeze(10), value, pilot(10))
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["multiplicity"]["per_interval_alpha"], 0.005)
        self.assertEqual(result["sensitivity"]["minimum_trajectories_for_p_value_resolution"], 9)
        self.assertEqual(result["global_design"]["planned_confirmation_trajectories"], 9)

    def test_external_fixed_variance_mode_is_explicit(self) -> None:
        value = spec()
        value["precision_basis"] = {
            "mode": "EXTERNAL_FIXED_CONSERVATIVE_VARIANCE",
            "contributor_pilot_summary_path": None,
            "contributor_pilot_summary_sha256": None,
            "external_fixed_rationale": "externally justified conservative bound",
        }
        value["candidate_precision_inputs"]["candidate-0"]["external_variance_plan"] = 1.0
        result = plan_contributor_precision(candidate_freeze(), value, None)
        self.assertTrue(result["valid"], result["errors"])
        self.assertIsNone(result["candidate_precision"]["candidate-0"]["pilot_sample_variance"])

    def test_uninstantiated_input_template_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        template = json.loads(
            (root / "theory_oracle" / "QWEN3_BIAS_CONTRIBUTOR_PRECISION_INPUT_SPEC_TEMPLATE_V0_1.json").read_text(encoding="utf-8")
        )
        result = plan_contributor_precision(candidate_freeze(), template, None)
        self.assertFalse(result["valid"])
        self.assertTrue(any("not frozen" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
