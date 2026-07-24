from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    EXPECTED_CALIBRATION_ANALYSIS_FILES,
    EXPECTED_ENDPOINT_ROLE_CATALOG,
    FIXED_RESOURCE_EXISTENCE,
    minimum_attainable_signflip_p,
    plan_confirmation,
    required_trajectories_for_half_width,
    signflip_resolution_requirement,
    tail_trajectory_requirement,
    variance_upper_bound,
)


def calibration(effects: list[float]) -> dict:
    return {
        "valid": True,
        "construction": {"trajectories": 4, "top_level_df": 3},
        "analysis_code": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in EXPECTED_CALIBRATION_ANALYSIS_FILES.items()
        },
        "endpoints": {
            "U1_reference_aligned_shift": {
                "status": "COMPLETE_FOUR_TRAJECTORY_CALIBRATION_DESCRIPTION",
                "endpoint_class": "SIGNED_UPDATE_GEOMETRY_ENDPOINT",
                "trajectory_rows": [
                    {"trajectory_id": f"calibration-{index}", "mean_effect": effect}
                    for index, effect in enumerate(effects)
                ],
            }
        },
    }


def threshold_sources() -> dict:
    common = {
        "description": "unit-test threshold contract",
        "selection_rule": "fixed by the test before reading calibration effects",
        "uses_calibration_candidate_mean_or_sign": False,
    }
    return {
        "desired_half_width": {
            **common,
            "kind": "INDEPENDENT_MEASUREMENT_RESOLUTION",
        },
        "variance_floor_sd": {
            **common,
            "kind": "NEGATIVE_CONTROL_ENVELOPE",
        },
        "shift_existence_floor": {
            **common,
            "kind": "EXACT_ZERO_NULL",
        },
    }


def spec() -> dict:
    return {
        "schema_version": "forkcert.bias-oracle-confirmation-precision-spec.v0.1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "endpoint_family": ["U1_reference_aligned_shift"],
        "phase_conditioned_endpoint_family": [],
        "endpoint_role_catalog": copy.deepcopy(EXPECTED_ENDPOINT_ROLE_CATALOG),
        "family_alpha": 0.05,
        "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
        "variance_upper_confidence": 0.8,
        "minimum_confirmation_trajectories": 8,
        "resource_cap": 500,
        "tail": {"scope": "REGULARITY_CONDITIONAL_ONLY"},
        "endpoints": {
            "U1_reference_aligned_shift": {
                "desired_half_width": 0.5,
                "variance_floor_sd": 0.01,
                "shift_existence_floor": 0.0,
                "threshold_sources": threshold_sources(),
            }
        },
    }


def fixed_resource_spec() -> dict:
    value = spec()
    value["planning_mode"] = FIXED_RESOURCE_EXISTENCE
    value["fixed_confirmation_trajectories"] = 8
    value["fixed_resource_source"] = {
        "kind": "EXTERNAL_COMPUTE_BUDGET",
        "description": "unit-test fixed resource allocation",
        "selection_rule": "eight independent trajectories fixed before confirmation outcomes",
        "uses_calibration_candidate_mean_or_sign": False,
    }
    value["variance_upper_confidence"] = None
    endpoint = value["endpoints"]["U1_reference_aligned_shift"]
    endpoint["desired_half_width"] = None
    endpoint["variance_floor_sd"] = None
    endpoint["threshold_sources"] = {
        "shift_existence_floor": threshold_sources()["shift_existence_floor"]
    }
    return value


class ConfirmationPrecisionTests(unittest.TestCase):
    def test_fixed_resource_mode_does_not_invent_precision_threshold(self) -> None:
        result = plan_confirmation(
            calibration([0.0, 0.0, 0.0, 0.0]), fixed_resource_spec()
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["planning_mode"], FIXED_RESOURCE_EXISTENCE)
        self.assertEqual(result["planned_confirmation_trajectories"], 8)
        endpoint = result["endpoints"]["U1_reference_aligned_shift"]
        self.assertEqual(endpoint["status"], "PLANNED_FIXED_RESOURCE_EXISTENCE")
        self.assertIsNone(endpoint["desired_half_width"])
        self.assertIsNone(endpoint["variance_floor_sd"])
        self.assertEqual(endpoint["calibration_sample_variance"], 0.0)

    def test_fixed_resource_mode_must_meet_signflip_resolution(self) -> None:
        value = fixed_resource_spec()
        value["family_alpha"] = 0.005
        result = plan_confirmation(
            calibration([-1.5, -0.5, 0.5, 1.5]), value
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("sign-flip p-value resolution" in error for error in result["errors"])
        )

    def test_fixed_resource_count_requires_outcome_independent_source(self) -> None:
        value = fixed_resource_spec()
        del value["fixed_resource_source"]
        result = plan_confirmation(
            calibration([-1.5, -0.5, 0.5, 1.5]), value
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("fixed_resource_source" in error for error in result["errors"])
        )

    def test_endpoint_role_catalog_is_mandatory_and_frozen(self) -> None:
        value = spec()
        del value["endpoint_role_catalog"]
        result = plan_confirmation(
            calibration([-1.5, -0.5, 0.5, 1.5]), value
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("endpoint_role_catalog" in error for error in result["errors"])
        )

    def test_nonnegative_disagreement_profile_cannot_size_bias_confirmation(self) -> None:
        source = calibration([-1.5, -0.5, 0.5, 1.5])
        source["endpoints"]["U1_reference_aligned_shift"][
            "endpoint_class"
        ] = "NONNEGATIVE_EVENT_DISAGREEMENT_PROFILE_NOT_B"
        result = plan_confirmation(source, spec())
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "cannot size a signed-bias confirmation" in error
                for error in result["errors"]
            )
        )

    def test_calibration_analysis_hash_drift_fails_closed(self) -> None:
        source = calibration([-1.5, -0.5, 0.5, 1.5])
        source["analysis_code"]["record_loader"]["sha256"] = "0" * 64
        result = plan_confirmation(source, spec())
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("record_loader hash drifted" in error for error in result["errors"])
        )

    def test_optional_practical_tolerance_requires_independent_external_source(self) -> None:
        value = spec()
        endpoint = value["endpoints"]["U1_reference_aligned_shift"]
        endpoint["practical_tolerance"] = 0.75
        endpoint["practical_tolerance_source"] = {
            "kind": "EXTERNAL_SCIENTIFIC_TOLERANCE",
            "description": "application-level tolerance fixed independently",
            "selection_rule": "declared before confirmation and without calibration mean/sign",
            "uses_calibration_candidate_mean_or_sign": False,
        }
        result = plan_confirmation(calibration([-1.5, -0.5, 0.5, 1.5]), value)
        self.assertTrue(result["valid"], result["errors"])
        planned = result["endpoints"]["U1_reference_aligned_shift"]
        self.assertEqual(planned["practical_tolerance"], 0.75)

        invalid = copy.deepcopy(value)
        invalid["endpoints"]["U1_reference_aligned_shift"][
            "practical_tolerance_source"
        ]["kind"] = "INDEPENDENT_MEASUREMENT_RESOLUTION"
        result = plan_confirmation(
            calibration([-1.5, -0.5, 0.5, 1.5]), invalid
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("external scientific tolerance" in error for error in result["errors"])
        )

    def test_phase_conditioned_family_is_planned_without_posthoc_selection(self) -> None:
        source = calibration([-1.5, -0.5, 0.5, 1.5])
        source["endpoints"]["U1_reference_aligned_shift"]["phase_rows"] = [
            {
                "trajectory_id": f"calibration-{trajectory}",
                "phase": phase,
                "mean_effect": effect + offset,
            }
            for trajectory, effect in enumerate((-1.5, -0.5, 0.5, 1.5))
            for phase, offset in (("early", 1.0), ("middle", 0.0), ("late", -1.0))
        ]
        value = spec()
        value["phase_conditioned_endpoint_family"] = [
            "U1_reference_aligned_shift"
        ]
        result = plan_confirmation(source, value)
        self.assertTrue(result["valid"], result["errors"])
        self.assertAlmostEqual(result["multiplicity"]["per_interval_alpha"], 0.05 / 4)
        self.assertEqual(result["multiplicity"]["confirmatory_comparisons"], 4)
        self.assertEqual(
            set(
                result["endpoints"]["U1_reference_aligned_shift"][
                    "phase_conditioned_plans"
                ]
            ),
            {"early", "middle", "late"},
        )

    def test_u2_uses_crossfit_projection_dispersion_from_frozen_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direction = {
                "schema_version": "forkcert.qwen3-u2-frozen-direction.v0.1",
                "valid": True,
                "verdict": "VALID_FROZEN_U2_CALIBRATION_DIRECTION",
                "status": "FROZEN_BEFORE_CONFIRMATION",
                "precision_contract": {
                    "desired_projection_half_width": 0.5,
                    "projection_variance_floor_sd": 0.01,
                    "projection_shift_existence_floor": 0.0,
                },
                "stability": {
                    "crossfit_projections": [-1.5, -0.5, 0.5, 1.5]
                },
            }
            path = root / "direction.json"
            path.write_text(json.dumps(direction), encoding="utf-8")
            value = spec()
            value["endpoint_family"] = ["U2_calibration_direction_shift"]
            value["endpoints"] = {
                "U2_calibration_direction_shift": {
                    "desired_half_width": 0.5,
                    "variance_floor_sd": 0.01,
                    "shift_existence_floor": 0.0,
                    "threshold_sources": copy.deepcopy(
                        spec()["endpoints"]["U1_reference_aligned_shift"][
                            "threshold_sources"
                        ]
                    ),
                }
            }
            value["U2_directional_replication"] = {
                "status": "FROZEN_VALID_DIRECTION",
                "direction_manifest_path": str(path),
                "direction_manifest_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
            result = plan_confirmation(calibration([0.0] * 4), value)
            self.assertTrue(result["valid"], result["errors"])
            endpoint = result["endpoints"]["U2_calibration_direction_shift"]
            self.assertEqual(
                endpoint["direction"]["planning_dispersion"],
                "leave-one-trajectory-out cross-fitted projections",
            )

    def test_signflip_resolution_can_raise_eight_trajectory_minimum(self) -> None:
        self.assertAlmostEqual(minimum_attainable_signflip_p(8), 2 / 256)
        self.assertEqual(signflip_resolution_requirement(0.005, 8, 16), 9)

    def test_signflip_resolution_is_applied_to_integrated_precision_plan(self) -> None:
        value = spec()
        value["endpoint_family"] = [
            name
            for role in (
                "core_signed_bias_candidates",
                "optional_signed_numerical_bias_candidates",
                "optional_signed_semantic_bias_candidates",
            )
            for name in EXPECTED_ENDPOINT_ROLE_CATALOG[role]
            if name != "U2_calibration_direction_shift"
        ]
        value["family_alpha"] = 0.045
        value["endpoints"] = {
            name: {
                "desired_half_width": 100.0,
                "variance_floor_sd": 0.01,
                "shift_existence_floor": 0.0,
                "threshold_sources": copy.deepcopy(
                    spec()["endpoints"]["U1_reference_aligned_shift"][
                        "threshold_sources"
                    ]
                ),
            }
            for name in value["endpoint_family"]
        }
        source = calibration([-1.5, -0.5, 0.5, 1.5])
        exemplar = source["endpoints"]["U1_reference_aligned_shift"]
        source["endpoints"] = {
            name: copy.deepcopy(exemplar) for name in value["endpoint_family"]
        }
        result = plan_confirmation(source, value)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["multiplicity"]["per_interval_alpha"], 0.005)
        self.assertEqual(result["sensitivity"]["minimum_trajectories_for_p_value_resolution"], 9)
        self.assertEqual(result["planned_confirmation_trajectories"], 9)

    def test_uninstantiated_template_fails_closed_without_crashing(self) -> None:
        root = Path(__file__).resolve().parents[1]
        template = json.loads(
            (
                root
                / "theory_oracle"
                / "QWEN3_CONFIRMATION_PRECISION_SPEC_TEMPLATE_V0_1.json"
            ).read_text(encoding="utf-8")
        )
        result = plan_confirmation(calibration([-1.5, -0.5, 0.5, 1.5]), template)
        self.assertFalse(result["valid"])
        self.assertEqual(result["verdict"], "UNINSTANTIATED_OR_INFEASIBLE")
        self.assertTrue(any("not FROZEN" in error for error in result["errors"]))

    def test_thresholds_without_auditable_source_fail_closed(self) -> None:
        value = spec()
        del value["endpoints"]["U1_reference_aligned_shift"]["threshold_sources"]
        result = plan_confirmation(calibration([-1.5, -0.5, 0.5, 1.5]), value)
        self.assertFalse(result["valid"])
        self.assertTrue(any("threshold_sources" in error for error in result["errors"]))

    def test_threshold_source_roles_cannot_be_swapped(self) -> None:
        value = spec()
        sources = value["endpoints"]["U1_reference_aligned_shift"][
            "threshold_sources"
        ]
        sources["shift_existence_floor"]["kind"] = (
            "EXTERNAL_SCIENTIFIC_TOLERANCE"
        )
        sources["variance_floor_sd"]["kind"] = "EXACT_ZERO_NULL"
        result = plan_confirmation(
            calibration([-1.5, -0.5, 0.5, 1.5]), value
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("unsupported shift_existence_floor" in error for error in result["errors"])
        )
        self.assertTrue(
            any("unsupported variance_floor_sd" in error for error in result["errors"])
        )

    def test_variance_upper_bound_exceeds_pilot_variance(self) -> None:
        self.assertGreater(variance_upper_bound(1.0, 4, 0.8), 1.0)

    def test_tighter_half_width_requires_at_least_as_many_trajectories(self) -> None:
        loose = required_trajectories_for_half_width(1.0, 1.0, 0.05, 8, 500)
        tight = required_trajectories_for_half_width(1.0, 0.25, 0.05, 8, 500)
        self.assertIsNotNone(loose)
        self.assertIsNotNone(tight)
        self.assertGreaterEqual(tight, loose)

    def test_calibration_mean_and_sign_do_not_change_precision_plan(self) -> None:
        centered = plan_confirmation(calibration([-1.5, -0.5, 0.5, 1.5]), spec())
        shifted = plan_confirmation(calibration([8.5, 9.5, 10.5, 11.5]), spec())
        reversed_result = plan_confirmation(
            calibration([1.5, 0.5, -0.5, -1.5]), spec()
        )
        self.assertTrue(centered["valid"], centered["errors"])
        self.assertEqual(
            centered["planned_confirmation_trajectories"],
            shifted["planned_confirmation_trajectories"],
        )
        self.assertEqual(
            centered["planned_confirmation_trajectories"],
            reversed_result["planned_confirmation_trajectories"],
        )

    def test_bonferroni_is_not_less_conservative_than_named_endpoint_intervals(self) -> None:
        two_endpoint_calibration = calibration([-1.5, -0.5, 0.5, 1.5])
        second = copy.deepcopy(
            two_endpoint_calibration["endpoints"]["U1_reference_aligned_shift"]
        )
        two_endpoint_calibration["endpoints"]["T1a_heldout_grpo_shift"] = second
        base = spec()
        base["endpoint_family"].append("T1a_heldout_grpo_shift")
        base["endpoints"]["T1a_heldout_grpo_shift"] = copy.deepcopy(
            base["endpoints"]["U1_reference_aligned_shift"]
        )
        bonferroni = plan_confirmation(two_endpoint_calibration, base)
        named_spec = copy.deepcopy(base)
        named_spec["multiplicity"] = "NAMED_ENDPOINTS_NO_JOINT_CLAIM_TWO_SIDED"
        named = plan_confirmation(two_endpoint_calibration, named_spec)
        self.assertTrue(bonferroni["valid"], bonferroni["errors"])
        self.assertTrue(named["valid"], named["errors"])
        self.assertGreaterEqual(
            bonferroni["planned_confirmation_trajectories"],
            named["planned_confirmation_trajectories"],
        )

    def test_zero_pilot_variance_without_floor_fails_closed(self) -> None:
        value = spec()
        value["endpoints"]["U1_reference_aligned_shift"]["variance_floor_sd"] = 0.0
        result = plan_confirmation(calibration([0.0, 0.0, 0.0, 0.0]), value)
        self.assertFalse(result["valid"])
        self.assertTrue(any("zero planning scale" in error for error in result["errors"]))

    def test_five_percent_tail_scope_requires_fifty_nine_trajectories(self) -> None:
        required, result = tail_trajectory_requirement(
            {
                "scope": "EXPLICIT_PREVALENCE_COVERAGE",
                "minimum_prevalence": 0.05,
                "alpha": 0.05,
            }
        )
        self.assertEqual(required, 59)
        self.assertEqual(result["required_trajectories"], 59)


if __name__ == "__main__":
    unittest.main()
    signflip_resolution_requirement,
