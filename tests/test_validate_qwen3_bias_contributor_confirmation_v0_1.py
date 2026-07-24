from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.bias_oracle_contributor_precision_v0_1 import (
    plan_contributor_precision,
)
from theory_oracle.validate_qwen3_bias_contributor_confirmation_v0_1 import (
    resolve_confirmed_endpoint,
    validate_candidate_universe,
    validate_contributor_manifest,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets() -> list[dict]:
    rows = []
    step = 1
    for phase in ("early", "middle", "late"):
        for _ in range(8):
            rows.append({"phase": phase, "optimizer_step": step, "state_id": f"s-{step}"})
            step += 1
    return rows


def frozen_fixture(root: Path) -> tuple[Path, dict]:
    repository_root = Path(__file__).resolve().parents[1]
    validator_path = repository_root / "theory_oracle" / "validate_qwen3_bias_contributor_confirmation_v0_1.py"
    precision_planner_path = repository_root / "theory_oracle" / "bias_oracle_contributor_precision_v0_1.py"
    estimator_path = repository_root / "theory_oracle" / "bias_oracle_contributor_v0_1.py"
    sensitivity_path = repository_root / "theory_oracle" / "bias_oracle_trajectory_signflip_v0_1.py"
    query_path = root / "query.json"
    write_json(query_path, {"query_id": "Q-R", "status": "FROZEN"})

    calibration_ids = [f"calibration-{index}" for index in range(4)]
    calibration_seeds = [101, 102, 103, 104]
    calibration_slices = [f"cal-slice-{index}" for index in range(4)]
    confirmation_inputs = [
        {
            "trajectory_id": f"endpoint-confirmation-{index}",
            "trajectory_seed": 200 + index,
            "data_slice_id": f"endpoint-slice-{index}",
            "capture_plan_path": f"endpoint-plan-{index}.json",
            "capture_plan_sha256": f"endpoint-plan-sha-{index}",
            "results_root": f"endpoint-results-{index}",
        }
        for index in range(8)
    ]
    endpoint_manifest = {
        "trajectory_inputs": confirmation_inputs,
        "calibration_exclusion": {
            "trajectory_ids": calibration_ids,
            "trajectory_seeds": calibration_seeds,
            "data_slice_ids": calibration_slices,
        },
    }
    endpoint_manifest_path = root / "endpoint-confirmation-manifest.json"
    write_json(endpoint_manifest_path, endpoint_manifest)
    endpoint = "U1_reference_aligned_shift"
    endpoint_evaluation = {
        "schema_version": "forkcert.qwen3-bias-oracle-confirmation.v0.1",
        "valid": True,
        "manifest": {"path": str(endpoint_manifest_path), "sha256": sha(endpoint_manifest_path)},
        "endpoints": {
            endpoint: {
                "final_shift_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
                "estimate": {"B": {"estimate": 0.02}},
            }
        },
        "operator_attribution_gate": {
            "ready_endpoints": [endpoint],
            "automatic_operator_launch": False,
        },
        "state_evidence": [
            {"state_id": "confirmation-state-0"},
            {"state_id": "confirmation-state-1"},
        ],
    }
    endpoint_evaluation_path = root / "endpoint-confirmation-evaluation.json"
    write_json(endpoint_evaluation_path, endpoint_evaluation)

    realization_map = {
        "schema_version": "forkcert.qwen3-bias-contributor-state-realization-map.v0.1",
        "status": "COMPLETE_VALID_BEFORE_CONTRIBUTOR_PILOT",
        "candidate_id": "candidate-0",
        "claim_unit_type": "GENERATED_KERNEL_INVOCATION",
        "intervention_version": "v0.1",
        "state_bank_identity": {
            "path": str(endpoint_evaluation_path),
            "sha256": sha(endpoint_evaluation_path),
        },
        "state_rows": [
            {
                "state_id": state_id,
                "realized": True,
                "exact_call_ids": [f"candidate-0::{state_id}::call-0"],
                "expected_call_count": 1,
                "call_order_digest": f"order-{state_id}",
                "absence_verified": False,
            }
            for state_id in ("confirmation-state-0", "confirmation-state-1")
        ],
    }
    realization_map_path = root / "state-realization-map.json"
    write_json(realization_map_path, realization_map)
    intervention_spec = {
        "schema_version": "forkcert.qwen3-bias-contributor-intervention.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
        "candidate_id": "candidate-0",
        "declared_intervention_unit": "GENERATED_KERNEL_INVOCATION",
        "intervention_version": "v0.1",
        "identity_semantics": "EXACT_GENERATED_LAUNCH_WITH_STATE_REALIZATION_MAP",
        "structural_identity": {
            "identity_digest": "generated-launch-candidate-0",
            "description": "one exact generated launch identity",
        },
        "state_realization_map": {
            "path": str(realization_map_path),
            "sha256": sha(realization_map_path),
        },
    }
    intervention_path = root / "intervention.json"
    write_json(intervention_path, intervention_spec)
    raw_census_path = root / "raw-census.json"
    write_json(
        raw_census_path,
        {
            "states": ["confirmation-state-0", "confirmation-state-1"],
            "observed_unit_ids": ["candidate-0"],
        },
    )
    coverage = {
        "schema_version": "forkcert.qwen3-bias-contributor-universe-coverage.v0.1",
        "status": "COMPLETE_VALID_BEFORE_CONTRIBUTOR_PILOT",
        "scope_kind": "ALL_GENERATED_KERNEL_INVOCATIONS_IN_FROZEN_STATE_BANK",
        "state_bank_identity": {
            "path": str(endpoint_evaluation_path),
            "sha256": sha(endpoint_evaluation_path),
        },
        "enumerated_unit_ids": ["candidate-0"],
        "state_rows": [
            {
                "state_id": state_id,
                "census_complete": True,
                "observed_unit_ids": ["candidate-0"],
            }
            for state_id in ("confirmation-state-0", "confirmation-state-1")
        ],
        "census_artifacts": [
            {"path": str(raw_census_path), "sha256": sha(raw_census_path)}
        ],
        "census_claim_scope": "ALL_OBSERVED_UNITS_IN_FROZEN_STATE_BANK",
    }
    coverage_path = root / "candidate-coverage.json"
    write_json(coverage_path, coverage)
    universe = {
        "schema_version": "forkcert.qwen3-bias-contributor-universe.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
        "target_endpoint": endpoint,
        "enumeration_scope": "one generated-kernel invocation in unit-test fixture",
        "scope_kind": "ALL_GENERATED_KERNEL_INVOCATIONS_IN_FROZEN_STATE_BANK",
        "state_bank_identity": {
            "path": str(endpoint_evaluation_path),
            "sha256": sha(endpoint_evaluation_path),
        },
        "coverage_mode": "EXHAUSTIVE_ELIGIBLE_UNIVERSE",
        "coverage_evidence": {
            "path": str(coverage_path),
            "sha256": sha(coverage_path),
        },
        "units": [
            {
                "candidate_id": "candidate-0",
                "claim_unit_type": "GENERATED_KERNEL_INVOCATION",
                "selection_eligible": True,
                "exclusion_reason": None,
            }
        ],
        "selected_candidate_ids": ["candidate-0"],
        "claim_scope": "ALL_ELIGIBLE_UNITS_IN_FROZEN_UNIVERSE",
        "all_eligible_units_covered_claim_allowed": True,
    }
    universe_path = root / "candidate-universe.json"
    write_json(universe_path, universe)
    candidate_artifact = {
        "schema_version": "forkcert.qwen3-bias-contributor-candidate-freeze.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
        "target_endpoint": endpoint,
        "endpoint_confirmation_evaluation": {
            "path": str(endpoint_evaluation_path),
            "sha256": sha(endpoint_evaluation_path),
            "required_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
        },
        "selection_data_role": "DISCOVERY_ONLY",
        "selection_rule": "structural census then one predeclared generated invocation",
        "candidate_universe": {
            "path": str(universe_path),
            "sha256": sha(universe_path),
        },
        "primary_candidates": [
            {
                "candidate_id": "candidate-0",
                "claim_unit_type": "GENERATED_KERNEL_INVOCATION",
                "intervention_version": "v0.1",
                "intervention_spec_path": str(intervention_path),
                "intervention_spec_sha256": sha(intervention_path),
                "selection_source": "CALIBRATION_DISCOVERY",
            }
        ],
        "claim_unit_must_equal_intervention_unit": True,
        "freeze_timestamp_utc": "2026-07-20T08:00:00Z",
    }
    candidate_path = root / "candidate-freeze.json"
    write_json(candidate_path, candidate_artifact)

    precision_spec = {
        "schema_version": "forkcert.qwen3-bias-contributor-precision-input-spec.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_PRECISION_PLANNING",
        "target_endpoint": endpoint,
        "frozen_bias_direction": {
            "source": "ENDPOINT_CONFIRMATION_ONLY",
            "kind": "SCALAR_SIGN",
            "scalar_sign": 1,
            "vector_direction_artifact_path": None,
            "vector_direction_artifact_sha256": None,
        },
        "candidate_freeze": {
            "artifact_path": str(candidate_path),
            "artifact_sha256": sha(candidate_path),
        },
        "precision_basis": {
            "mode": "EXTERNAL_FIXED_CONSERVATIVE_VARIANCE",
            "contributor_pilot_summary_path": None,
            "contributor_pilot_summary_sha256": None,
            "external_fixed_rationale": "conservative fixed eight-trajectory construction fixture",
        },
        "multiplicity": {
            "primary_repair_family_members": ["candidate-0"],
            "method": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
            "family_alpha": 0.05,
            "injection_is_separate_family": True,
        },
        "variance_upper_confidence": 0.8,
        "candidate_precision_inputs": {
            "candidate-0": {
                "desired_halfwidth": 0.01,
                "directional_contribution_floor": 0.0,
                "variance_floor_sd": 0.001,
                "external_variance_plan": 0.0001,
                "threshold_sources": {
                    "desired_halfwidth": {
                        "kind": "INDEPENDENT_MEASUREMENT_RESOLUTION",
                        "description": "unit-test desired-width contract",
                        "selection_rule": "fixed before contributor effects are inspected",
                        "uses_pilot_contribution_mean_or_sign": False,
                    },
                    "directional_contribution_floor": {
                        "kind": "EXACT_ZERO_NULL",
                        "description": "unit-test contribution-floor contract",
                        "selection_rule": "fixed before contributor effects are inspected",
                        "uses_pilot_contribution_mean_or_sign": False,
                    },
                    "variance_floor_sd": {
                        "kind": "NEGATIVE_CONTROL_ENVELOPE",
                        "description": "unit-test variance-floor contract",
                        "selection_rule": "fixed before contributor effects are inspected",
                        "uses_pilot_contribution_mean_or_sign": False,
                    },
                },
            }
        },
        "global_design": {
            "minimum_confirmation_trajectories": 8,
            "resource_cap_trajectories": 16,
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
    precision_spec_path = root / "contributor-precision-input-spec.json"
    write_json(precision_spec_path, precision_spec)
    precision = plan_contributor_precision(candidate_artifact, precision_spec, None)
    assert precision["valid"], precision["errors"]
    precision["candidate_freeze"].update(
        {"artifact_path": str(candidate_path), "artifact_sha256": sha(candidate_path)}
    )
    precision["inputs"] = {
        "spec": {"path": str(precision_spec_path), "sha256": sha(precision_spec_path)}
    }
    precision_path = root / "contributor-precision.json"
    write_json(precision_path, precision)

    trajectory_specs = []
    for index in range(8):
        row = {
            "trajectory_id": f"contributor-confirmation-{index}",
            "trajectory_seed": 400 + index,
            "data_slice_id": f"contributor-slice-{index}",
            "capture_plan_path": str(root / f"contributor-plan-{index}.json"),
            "results_root": str(root / f"contributor-results-{index}"),
        }
        plan = {
            "schema_version": "forkcert.multi-transition-capture-plan.v0.1",
            "identity": {
                "trajectory_id": row["trajectory_id"],
                "trajectory_seed": row["trajectory_seed"],
                "data_slice_id": row["data_slice_id"],
            },
            "targets": targets(),
        }
        plan_path = Path(row["capture_plan_path"])
        write_json(plan_path, plan)
        row["capture_plan_sha256"] = sha(plan_path)
        trajectory_specs.append(row)

    exclusions = []
    for trajectory_id in calibration_ids:
        exclusions.append({"role": "CALIBRATION", "trajectory_id": trajectory_id})
    for seed in calibration_seeds:
        exclusions.append({"role": "CALIBRATION", "trajectory_seed": seed})
    for data_slice_id in calibration_slices:
        exclusions.append({"role": "CALIBRATION", "data_slice_id": data_slice_id})
    exclusions.extend({"role": "ENDPOINT_CONFIRMATION", **{key: row[key] for key in ("trajectory_id", "trajectory_seed", "data_slice_id")}} for row in confirmation_inputs)

    manifest = {
        "schema_version": "forkcert.qwen3-bias-contributor-confirmation-manifest.v0.1",
        "status": "FROZEN_BEFORE_CONTRIBUTOR_CONFIRMATION",
        "query": {
            "query_manifest_path": str(query_path),
            "query_manifest_sha256": sha(query_path),
            "target_state_distribution": "Q-R",
            "target_state_condition": {"kind": "GLOBAL"},
            "reference_implementation": "eager_baseline_not_truth",
            "candidate_implementation": "compiled-frozen",
            "randomness_protocol": "paired-frozen",
        },
        "target_endpoint": {
            "name": endpoint,
            "signed_direction_definition": "positive means along confirmed B",
            "practical_contribution_criterion": 0.0,
            "magnitude_only_endpoint_prohibited": True,
        },
        "endpoint_confirmation_gate": {
            "evaluation_path": str(endpoint_evaluation_path),
            "evaluation_sha256": sha(endpoint_evaluation_path),
            "required_shift_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
            "method_sensitivity_conflict_prohibited": True,
            "observed_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
        },
        "candidate_freeze": {"artifact_path": str(candidate_path), "artifact_sha256": sha(candidate_path)},
        "contributor_precision_plan": {
            "path": str(precision_path),
            "sha256": sha(precision_path),
            "planned_confirmation_trajectories": 8,
            "pilot_trajectories_excluded_from_confirmation_df": True,
        },
        "analysis_implementation": {
            "design_validator_path": str(validator_path),
            "design_validator_sha256": sha(validator_path),
            "precision_planner_path": str(precision_planner_path),
            "precision_planner_sha256": sha(precision_planner_path),
            "profile_estimator_path": str(estimator_path),
            "profile_estimator_sha256": sha(estimator_path),
            "trajectory_sensitivity_path": str(sensitivity_path),
            "trajectory_sensitivity_sha256": sha(sensitivity_path),
        },
        "contributor_confirmation_bank": {
            "independent_of_calibration": True,
            "independent_of_endpoint_confirmation": True,
            "independent_of_contributor_precision_pilot": True,
            "trajectory_count": 8,
            "trajectory_specs": trajectory_specs,
            "all_prior_role_exclusions": exclusions,
        },
        "arms": {
            "required": ["REFERENCE", "FULL_CANDIDATE", "CANDIDATE_REPAIR"],
            "optional": ["REFERENCE_INJECTION", "PAIR_OR_COALITION_REPAIR", "SHAM_CONTROL"],
            "paired_transition_repeats": 2,
        },
        "primary_estimand": {
            "name": "C_REPAIR",
            "definition": "E[(Y_candidate-Y_reference)-(Y_candidate_repair-Y_reference)]",
            "confirmation_scale": "PROJECTION_ON_ENDPOINT_CONFIRMATION_FROZEN_BIAS_DIRECTION",
            "cluster_unit": "independent_trajectory",
            "target_phase": None,
            "report_residual_bias": True,
            "report_state_heterogeneity": True,
            "report_same_state_runtime_variability": True,
        },
        "residual_claim": {
            "absolute_bias_reduction_is_primary": False,
            "simultaneous_uncertainty_rule": None,
            "overshoot_must_be_reported": True,
        },
        "hypothesis_family": {
            "members": ["candidate-0"],
            "multiplicity_method": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
            "family_alpha": 0.05,
            "adjusted_interval_alpha": 0.05,
            "injection_is_separate_family": True,
        },
        "intervention_integrity_gates": {
            "pre_state_identity": True,
            "baseline_anchor": True,
            "exact_call_identity_and_count": True,
            "exactly_declared_replacements": True,
            "no_unexpected_recompile": True,
            "unaffected_realization_identity": True,
            "restoration_anchor": True,
            "sham_control": True,
            "complete_endpoint_validity": True,
            "on_failure": "INVALID_INTERVENTION_NOT_NULL",
        },
        "interaction_policy": {
            "frozen_pairs_or_coalitions": [],
            "unmeasured_interactions_label": "INTERACTION_UNRESOLVED",
            "additivity_assumed": False,
        },
        "claim_boundary": {
            "correctness_authority": None,
            "root_cause_claim": False,
            "necessity_claim": False,
            "sufficiency_claim": False,
            "automatic_long_training_claim": False,
        },
    }
    manifest_path = root / "contributor-manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


class ContributorDesignValidationTests(unittest.TestCase):
    def test_universe_scope_kind_must_match_claim_unit_type(self) -> None:
        universe = {
            "schema_version": "forkcert.qwen3-bias-contributor-universe.v0.1",
            "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
            "target_endpoint": "U1_reference_aligned_shift",
            "enumeration_scope": "one generated-kernel invocation",
            "scope_kind": "ALL_SOURCE_OPERATOR_INVOCATIONS_IN_FROZEN_STATE_BANK",
            "state_bank_identity": {"path": "unused", "sha256": "unused"},
            "coverage_mode": "EXHAUSTIVE_ELIGIBLE_UNIVERSE",
            "units": [
                {
                    "candidate_id": "candidate-0",
                    "claim_unit_type": "GENERATED_KERNEL_INVOCATION",
                    "selection_eligible": True,
                }
            ],
            "selected_candidate_ids": ["candidate-0"],
            "claim_scope": "ALL_ELIGIBLE_UNITS_IN_FROZEN_UNIVERSE",
            "all_eligible_units_covered_claim_allowed": True,
        }
        errors = validate_candidate_universe(
            universe, "U1_reference_aligned_shift", ["candidate-0"]
        )
        self.assertTrue(any("type disagrees with scope_kind" in error for error in errors))

    def test_subset_universe_cannot_claim_exhaustive_coverage(self) -> None:
        universe = {
            "schema_version": "forkcert.qwen3-bias-contributor-universe.v0.1",
            "status": "FROZEN_BEFORE_CONTRIBUTOR_PILOT",
            "target_endpoint": "U1_reference_aligned_shift",
            "enumeration_scope": "two generated invocations",
            "scope_kind": "ALL_GENERATED_KERNEL_INVOCATIONS_IN_FROZEN_STATE_BANK",
            "state_bank_identity": {"path": "not-used-by-structural-unit-test", "sha256": "x"},
            "coverage_mode": "PREDECLARED_SUBSET_OF_UNIVERSE",
            "units": [
                {
                    "candidate_id": candidate_id,
                    "claim_unit_type": "GENERATED_KERNEL_INVOCATION",
                    "selection_eligible": True,
                }
                for candidate_id in ("candidate-0", "candidate-1")
            ],
            "selected_candidate_ids": ["candidate-0"],
            "claim_scope": "ALL_ELIGIBLE_UNITS_IN_FROZEN_UNIVERSE",
            "all_eligible_units_covered_claim_allowed": True,
        }
        errors = validate_candidate_universe(
            universe, "U1_reference_aligned_shift", ["candidate-0"]
        )
        self.assertTrue(any("subset universe claim scope" in error for error in errors))

    def test_phase_conditioned_endpoint_resolves_without_global_substitution(self) -> None:
        confirmation = {
            "endpoints": {
                "U1_reference_aligned_shift": {
                    "estimate": {"B": {"estimate": 99.0}},
                    "phase_conditioned_confirmation": {
                        "claims": {
                            "early": {
                                "final_shift_verdict": "REPRODUCIBLE_AVERAGE_SHIFT",
                                "estimate": {"estimate": -0.25},
                            }
                        }
                    },
                }
            }
        }
        claim, value = resolve_confirmed_endpoint(
            confirmation, "U1_reference_aligned_shift::phase=early"
        )
        self.assertEqual(claim["final_shift_verdict"], "REPRODUCIBLE_AVERAGE_SHIFT")
        self.assertEqual(value, -0.25)

    def test_frozen_independent_design_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            candidates, precision, errors = validate_contributor_manifest(manifest, path)
            self.assertFalse(errors)
            self.assertEqual(len(candidates["primary_candidates"]), 1)
            self.assertEqual(precision["global_design"]["planned_confirmation_trajectories"], 8)

    def test_candidate_universe_must_bind_exact_confirmation_state_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            candidate_path = Path(manifest["candidate_freeze"]["artifact_path"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            universe_path = Path(candidate["candidate_universe"]["path"])
            universe = json.loads(universe_path.read_text(encoding="utf-8"))
            other_bank_path = root / "other-endpoint-evaluation.json"
            write_json(other_bank_path, {"different": "state bank"})
            universe["state_bank_identity"] = {
                "path": str(other_bank_path),
                "sha256": sha(other_bank_path),
            }
            write_json(universe_path, universe)
            candidate["candidate_universe"]["sha256"] = sha(universe_path)
            write_json(candidate_path, candidate)
            manifest["candidate_freeze"]["artifact_sha256"] = sha(candidate_path)

            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(
                any(
                    "not bound to the endpoint-confirmation state bank" in error
                    for error in errors
                )
            )

    def test_exhaustive_claim_requires_complete_confirmation_state_census(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            candidate_path = Path(manifest["candidate_freeze"]["artifact_path"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            universe_path = Path(candidate["candidate_universe"]["path"])
            universe = json.loads(universe_path.read_text(encoding="utf-8"))
            coverage_path = Path(universe["coverage_evidence"]["path"])
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            coverage["state_rows"].pop()
            write_json(coverage_path, coverage)
            universe["coverage_evidence"]["sha256"] = sha(coverage_path)
            write_json(universe_path, universe)
            candidate["candidate_universe"]["sha256"] = sha(universe_path)
            write_json(candidate_path, candidate)
            manifest["candidate_freeze"]["artifact_sha256"] = sha(candidate_path)

            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(
                any(
                    "state census does not equal confirmation states" in error
                    for error in errors
                )
            )

    def test_operator_candidate_requires_exact_per_state_realization_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            candidate_path = Path(manifest["candidate_freeze"]["artifact_path"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            intervention_path = Path(
                candidate["primary_candidates"][0]["intervention_spec_path"]
            )
            intervention = json.loads(
                intervention_path.read_text(encoding="utf-8")
            )
            map_path = Path(intervention["state_realization_map"]["path"])
            realization_map = json.loads(map_path.read_text(encoding="utf-8"))
            realization_map["state_rows"].pop()
            write_json(map_path, realization_map)
            intervention["state_realization_map"]["sha256"] = sha(map_path)
            write_json(intervention_path, intervention)
            candidate["primary_candidates"][0]["intervention_spec_sha256"] = sha(
                intervention_path
            )
            write_json(candidate_path, candidate)
            manifest["candidate_freeze"]["artifact_sha256"] = sha(candidate_path)

            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(
                any("realization-map states/order mismatch" in error for error in errors)
            )

    def test_target_state_condition_cannot_drift_from_endpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["query"]["target_state_condition"] = {
                "kind": "PHASE",
                "value": "early",
            }
            _, _, errors = validate_contributor_manifest(changed, path)
            self.assertTrue(
                any("target_state_condition" in error for error in errors)
            )

    def test_prior_role_reuse_and_family_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["contributor_confirmation_bank"]["trajectory_specs"][0]["trajectory_seed"] = 101
            changed["hypothesis_family"]["members"] = []
            _, _, errors = validate_contributor_manifest(changed, path)
            self.assertTrue(any("reuses prior-role trajectory_seed" in error for error in errors))
            self.assertTrue(any("hypothesis family" in error for error in errors))

    def test_analysis_code_hash_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["analysis_implementation"]["profile_estimator_sha256"] = "0" * 64
            _, _, errors = validate_contributor_manifest(changed, path)
            self.assertTrue(any("profile_estimator hash mismatch" in error for error in errors))

    def test_sensitivity_resolution_blocks_under_sized_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision_path = Path(manifest["contributor_precision_plan"]["path"])
            precision = json.loads(precision_path.read_text(encoding="utf-8"))
            precision["multiplicity"]["family_alpha"] = 0.005
            precision["multiplicity"]["per_interval_alpha"] = 0.005
            write_json(precision_path, precision)
            manifest["contributor_precision_plan"]["sha256"] = sha(precision_path)
            manifest["hypothesis_family"]["family_alpha"] = 0.005
            manifest["hypothesis_family"]["adjusted_interval_alpha"] = 0.005
            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(any("cannot resolve adjusted sensitivity alpha" in error for error in errors))

    def test_global_plan_cannot_undercut_candidate_precision_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision_path = Path(manifest["contributor_precision_plan"]["path"])
            precision = json.loads(precision_path.read_text(encoding="utf-8"))
            precision["candidate_precision"]["candidate-0"]["planned_trajectories"] = 9
            write_json(precision_path, precision)
            manifest["contributor_precision_plan"]["sha256"] = sha(precision_path)
            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(any("global contributor plan is below" in error for error in errors))
            self.assertTrue(any("candidate_precision is not reproducible" in error for error in errors))

    def test_endpoint_without_confirmed_B_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            evaluation_path = Path(manifest["endpoint_confirmation_gate"]["evaluation_path"])
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            endpoint = manifest["target_endpoint"]["name"]
            evaluation["endpoints"][endpoint]["final_shift_verdict"] = "NO_STABLE_AVERAGE_DETECTED"
            write_json(evaluation_path, evaluation)
            manifest["endpoint_confirmation_gate"]["evaluation_sha256"] = sha(evaluation_path)
            candidate_path = Path(manifest["candidate_freeze"]["artifact_path"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["endpoint_confirmation_evaluation"]["sha256"] = sha(evaluation_path)
            write_json(candidate_path, candidate)
            manifest["candidate_freeze"]["artifact_sha256"] = sha(candidate_path)
            precision_path = Path(manifest["contributor_precision_plan"]["path"])
            precision = json.loads(precision_path.read_text(encoding="utf-8"))
            precision["candidate_freeze"]["artifact_sha256"] = sha(candidate_path)
            write_json(precision_path, precision)
            manifest["contributor_precision_plan"]["sha256"] = sha(precision_path)
            _, _, errors = validate_contributor_manifest(manifest, path)
            self.assertTrue(any("lacks REPRODUCIBLE_AVERAGE_SHIFT" in error for error in errors))

    def test_uninstantiated_template_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "theory_oracle" / "QWEN3_BIAS_CONTRIBUTOR_CONFIRMATION_MANIFEST_TEMPLATE_V0_1.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        candidates, precision, errors = validate_contributor_manifest(manifest, path)
        self.assertIsNone(candidates)
        self.assertIsNone(precision)
        self.assertTrue(any("not FROZEN" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
