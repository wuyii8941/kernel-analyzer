from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
import torch
from safetensors.torch import save_file

from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (
    ANALYSIS_CODE_PATHS,
    EXPECTED_SENSITIVITY,
    apply_trajectory_sensitivity,
    classify_oracle_disposition,
    classify_practical_materiality,
    collect_u2_direction_endpoint,
    evaluate_operator_attribution_eligibility,
    evaluate_phase_conditioned_claims,
    evaluate_realized_precision,
    realized_precision_decision_is_resolved,
    validate_confirmation_manifest,
    validate_confirmation_source_audit,
    validate_precision_provenance,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    EXPECTED_CALIBRATION_ANALYSIS_FILES,
    EXPECTED_ENDPOINT_ROLE_CATALOG,
    FIXED_RESOURCE_EXISTENCE,
    plan_confirmation,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_sources() -> dict:
    common = {
        "description": "unit-test threshold contract",
        "selection_rule": "fixed before calibration effects are inspected",
        "uses_calibration_candidate_mean_or_sign": False,
    }
    return {
        "desired_half_width": {**common, "kind": "INDEPENDENT_MEASUREMENT_RESOLUTION"},
        "variance_floor_sd": {**common, "kind": "NEGATIVE_CONTROL_ENVELOPE"},
        "shift_existence_floor": {**common, "kind": "EXACT_ZERO_NULL"},
    }


def target_rows(trajectory_id: str) -> list[dict]:
    rows = []
    step = 1
    for phase, population in (
        ("early", "1:100"),
        ("middle", "101:200"),
        ("late", "201:300"),
    ):
        for _ in range(8):
            rows.append(
                {
                    "optimizer_step": step,
                    "state_id": f"{trajectory_id}-{phase}-{step}",
                    "phase": phase,
                    "eligible_step_population": population,
                }
            )
            step += 1
    return rows


def frozen_fixture(root: Path) -> tuple[Path, dict]:
    calibration = {
        "valid": True,
        "construction": {"trajectories": 4, "top_level_df": 3},
        "analysis_code": {
            name: {"path": str(path), "sha256": sha(path)}
            for name, path in EXPECTED_CALIBRATION_ANALYSIS_FILES.items()
        },
        "endpoints": {
            "U1_reference_aligned_shift": {
                "status": "COMPLETE_FOUR_TRAJECTORY_CALIBRATION_DESCRIPTION",
                "endpoint_class": "SIGNED_UPDATE_GEOMETRY_ENDPOINT",
                "trajectory_rows": [
                    {
                        "trajectory_id": f"calibration-{index}",
                        "mean_effect": effect,
                    }
                    for index, effect in enumerate((-0.03, -0.01, 0.01, 0.03))
                ],
            }
        },
    }
    spec = {
        "schema_version": "forkcert.bias-oracle-confirmation-precision-spec.v0.1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "endpoint_family": ["U1_reference_aligned_shift"],
        "phase_conditioned_endpoint_family": [],
        "endpoint_role_catalog": copy.deepcopy(EXPECTED_ENDPOINT_ROLE_CATALOG),
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
                "threshold_sources": threshold_sources(),
            }
        },
    }
    calibration_path = root / "calibration.json"
    spec_path = root / "spec.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    precision = plan_confirmation(calibration, spec)
    if not precision["valid"] or precision["planned_confirmation_trajectories"] != 8:
        raise AssertionError(precision)
    precision["inputs"] = {
        "calibration": {"path": str(calibration_path), "sha256": sha(calibration_path)},
        "spec": {"path": str(spec_path), "sha256": sha(spec_path)},
    }
    precision_path = root / "precision.json"
    precision_path.write_text(json.dumps(precision), encoding="utf-8")
    inputs = []
    for index in range(8):
        trajectory_id = f"confirmation-{index}"
        seed = 9000 + index
        data_slice_id = f"slice-{index}"
        plan = {
            "schema_version": "forkcert.multi-transition-capture-plan.v0.1",
            "identity": {
                "query_id": "Q-R",
                "trajectory_id": trajectory_id,
                "trajectory_anchor": "EAGER_TRAJECTORY",
                "trajectory_seed": seed,
                "data_slice_id": data_slice_id,
                "state_selection_prng_seed": "frozen-confirmation-rank",
            },
            "targets": target_rows(trajectory_id),
        }
        plan_path = root / f"plan-{index}.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        config = {
            "dataset": {
                "name": "forkcert_builtin_arithmetic",
                "offset": 10000 + index * 64,
                "max_prompts": 64,
            },
            "training": {"seed": seed, "max_steps": 300},
        }
        data_slice_id = (
            f"forkcert_builtin_arithmetic[{10000 + index * 64}:"
            f"{10064 + index * 64}]"
        )
        plan["identity"]["data_slice_id"] = data_slice_id
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        config_path = root / f"config-{index}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        inputs.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_seed": seed,
                "data_slice_id": data_slice_id,
                "source_config_path": str(config_path),
                "source_config_sha256": sha(config_path),
                "capture_plan_path": str(plan_path),
                "capture_plan_sha256": sha(plan_path),
                "results_root": str(root / f"results-{index}"),
                "data_root": str(root / f"data-{index}"),
            }
        )
    design = {
        "schema_version": "forkcert.qwen3-bias-oracle-confirmation-bank-design.v0.1",
        "status": "FROZEN_BEFORE_COMPLETE_CALIBRATION_RESULTS",
    }
    design_path = root / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    bank = {
        "schema_version": "forkcert.qwen3-bias-oracle-confirmation-bank.v0.1",
        "valid": True,
        "verdict": "VALID_FROZEN_CONFIRMATION_TRAJECTORY_BANK",
        "design": {"path": str(design_path), "sha256": sha(design_path)},
        "precision": {
            "path": str(precision_path),
            "sha256": sha(precision_path),
            "planned_confirmation_trajectories": 8,
        },
        "trajectory_specs": inputs,
    }
    bank_path = root / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    manifest = {
        "schema_version": "forkcert.qwen3-bias-oracle-confirmation-manifest.v0.1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "query_id": "Q-R",
        "trajectory_anchor": "EAGER_TRAJECTORY",
        "precision_plan": {
            "path": str(precision_path),
            "sha256": sha(precision_path),
            "planned_confirmation_trajectories": 8,
        },
        "trajectory_bank": {"path": str(bank_path), "sha256": sha(bank_path)},
        "evaluator": {
            "path": str(
                Path(__file__).resolve().parents[1]
                / "theory_oracle"
                / "evaluate_qwen3_bias_oracle_confirmation_v0_1.py"
            ),
            "sha256": sha(
                Path(__file__).resolve().parents[1]
                / "theory_oracle"
                / "evaluate_qwen3_bias_oracle_confirmation_v0_1.py"
            ),
        },
        "analysis_code": {
            name: {"path": str(code_path), "sha256": sha(code_path)}
            for name, code_path in ANALYSIS_CODE_PATHS.items()
        },
        "calibration_exclusion": {
            "trajectory_ids": [f"calibration-{index}" for index in range(4)],
            "trajectory_seeds": [2001284755, 1810598814, 1677250702, 797459759],
            "data_slice_ids": [
                "forkcert_builtin_arithmetic[7296:7360]",
                "forkcert_builtin_arithmetic[3840:3904]",
                "forkcert_builtin_arithmetic[5696:5760]",
                "forkcert_builtin_arithmetic[3200:3264]",
            ],
        },
        "trajectory_inputs": inputs,
        "analysis": {
            "endpoint_family": ["U1_reference_aligned_shift"],
            "phase_conditioned_endpoint_family": [],
            "confirmatory_comparisons": 1,
            "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
            "per_interval_alpha": 0.05,
            "tail_scope": "REGULARITY_CONDITIONAL_ONLY",
            "sensitivity": precision["sensitivity"],
            "calibration_trajectories_in_confirmation_df": False,
            "correctness_authority": None,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


class ConfirmationManifestTests(unittest.TestCase):
    def test_fixed_resource_mode_reports_width_without_false_precision_gate(self) -> None:
        estimate = {"B": {"trajectory_t_interval": [0.2, 0.8]}}
        precision = evaluate_realized_precision(
            estimate,
            {
                "planning_mode": FIXED_RESOURCE_EXISTENCE,
                "desired_half_width": None,
            },
        )
        self.assertEqual(
            precision["verdict"],
            "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
        )
        self.assertAlmostEqual(precision["realized_half_width"], 0.3)
        eligibility = evaluate_operator_attribution_eligibility(
            "REPRODUCIBLE_AVERAGE_SHIFT", precision["verdict"]
        )
        self.assertTrue(eligibility["eligible"])
        self.assertEqual(
            classify_oracle_disposition(
                "REPRODUCIBLE_AVERAGE_SHIFT", precision["verdict"]
            ),
            "CONFIRMED_IMPLEMENTATION_RELATIVE_AVERAGE_SHIFT_FIXED_RESOURCE",
        )
        self.assertEqual(
            classify_oracle_disposition(
                "NO_STABLE_AVERAGE_DETECTED", precision["verdict"]
            ),
            "NO_STABLE_AVERAGE_DETECTED_AT_FIXED_RESOURCE",
        )
        self.assertTrue(
            realized_precision_decision_is_resolved(precision["verdict"])
        )

    def test_fixed_resource_phase_claim_uses_reported_interval_without_width_gate(self) -> None:
        estimate = {
            "conditional_B": {
                "predeclared_phase_rows": [
                    {
                        "phase": phase,
                        "estimate": effect,
                        "trajectory_t_interval": interval,
                        "trajectory_rows": [
                            {
                                "trajectory_id": f"confirmation-{index}",
                                "mean_effect": effect,
                            }
                            for index in range(8)
                        ],
                    }
                    for phase, effect, interval in (
                        ("early", 1.0, [0.8, 1.2]),
                        ("middle", 0.0, [-0.2, 0.2]),
                        ("late", -1.0, [-1.2, -0.8]),
                    )
                ]
            }
        }
        endpoint_plan = {
            "planning_mode": FIXED_RESOURCE_EXISTENCE,
            "shift_existence_floor": 0.0,
            "practical_tolerance": None,
            "phase_conditioned_plans": {
                phase: {"planning_mode": FIXED_RESOURCE_EXISTENCE}
                for phase in ("early", "middle", "late")
            },
        }
        result = evaluate_phase_conditioned_claims(
            estimate,
            endpoint_plan,
            0.05,
            {
                "exact_max_trajectories": 16,
                "monte_carlo_draws": 99999,
                "monte_carlo_seed": 172904,
            },
        )
        self.assertEqual(
            result["claims"]["early"]["realized_precision"]["verdict"],
            "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
        )
        self.assertTrue(
            result["claims"]["early"]["operator_attribution_eligibility"][
                "eligible"
            ]
        )

    def test_frozen_phase_claim_can_be_operator_eligible_without_global_promotion(self) -> None:
        estimate = {
            "conditional_B": {
                "predeclared_phase_rows": [
                    {
                        "phase": phase,
                        "estimate": effect,
                        "trajectory_t_interval": interval,
                        "trajectory_effects": [effect] * 8,
                        "trajectory_rows": [
                            {
                                "trajectory_id": f"confirmation-{index}",
                                "mean_effect": effect,
                            }
                            for index in range(8)
                        ],
                    }
                    for phase, effect, interval in (
                        ("early", 1.0, [0.95, 1.05]),
                        ("middle", 0.0, [-0.05, 0.05]),
                        ("late", -1.0, [-1.05, -0.95]),
                    )
                ]
            }
        }
        endpoint_plan = {
            "desired_half_width": 0.1,
            "shift_existence_floor": 0.0,
            "phase_conditioned_plans": {
                phase: {"status": "PLANNED"}
                for phase in ("early", "middle", "late")
            },
        }
        result = evaluate_phase_conditioned_claims(
            estimate, endpoint_plan, 0.0125, EXPECTED_SENSITIVITY
        )
        self.assertEqual(
            result["status"], "MEASURED_FROZEN_ALL_PHASE_CONFIRMATION_FAMILY"
        )
        self.assertTrue(
            result["claims"]["early"]["operator_attribution_eligibility"][
                "eligible"
            ]
        )
        self.assertEqual(
            result["claims"]["early"]["decision_axes"]["long_run_training_impact"],
            "UNINSTANTIATED_ONE_STEP_ORACLE_ONLY",
        )
        self.assertFalse(
            result["claims"]["middle"]["operator_attribution_eligibility"][
                "eligible"
            ]
        )
        del estimate["conditional_B"]["predeclared_phase_rows"][0][
            "trajectory_rows"
        ]
        invalid = evaluate_phase_conditioned_claims(
            estimate, endpoint_plan, 0.0125, EXPECTED_SENSITIVITY
        )
        self.assertEqual(
            invalid["status"], "INVALID_INCOMPLETE_PHASE_CONDITIONED_FAMILY"
        )

    def test_operator_eligibility_requires_shift_and_realized_precision(self) -> None:
        ready = evaluate_operator_attribution_eligibility(
            "REPRODUCIBLE_AVERAGE_SHIFT", "ADEQUATE_REALIZED_PRECISION"
        )
        self.assertTrue(ready["eligible"])
        self.assertEqual(ready["blockers"], [])

        blocked = evaluate_operator_attribution_eligibility(
            "REPRODUCIBLE_AVERAGE_SHIFT", "INDETERMINATE_REALIZED_PRECISION"
        )
        self.assertFalse(blocked["eligible"])
        self.assertEqual(blocked["blockers"], ["INDETERMINATE_REALIZED_PRECISION"])

    def test_composite_disposition_does_not_turn_imprecision_into_no_shift(self) -> None:
        self.assertEqual(
            classify_oracle_disposition(
                "NO_STABLE_AVERAGE_DETECTED", "ADEQUATE_REALIZED_PRECISION"
            ),
            "NO_AVERAGE_SHIFT_BEYOND_FLOOR_DETECTED_AT_TARGET_PRECISION",
        )
        self.assertEqual(
            classify_oracle_disposition(
                "NO_STABLE_AVERAGE_DETECTED", "INDETERMINATE_REALIZED_PRECISION"
            ),
            "INDETERMINATE_NONDETECTION_WITH_INADEQUATE_PRECISION",
        )
        self.assertEqual(
            classify_oracle_disposition(
                "REPRODUCIBLE_AVERAGE_SHIFT", "INDETERMINATE_REALIZED_PRECISION"
            ),
            "AVERAGE_SHIFT_DETECTED_BUT_TARGET_PRECISION_MISSED",
        )

    def test_practical_materiality_requires_external_tolerance_axis(self) -> None:
        self.assertEqual(
            classify_practical_materiality(0.2, 0.4, None, "REPRODUCIBLE_AVERAGE_SHIFT"),
            "UNINSTANTIATED_MATERIALITY",
        )
        self.assertEqual(
            classify_practical_materiality(0.2, 0.4, 0.1, "REPRODUCIBLE_AVERAGE_SHIFT"),
            "MATERIAL_AVERAGE_SHIFT",
        )
        self.assertEqual(
            classify_practical_materiality(-0.05, 0.05, 0.1, "NO_STABLE_AVERAGE_DETECTED"),
            "PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT",
        )

    def test_realized_precision_is_separate_from_shift_existence(self) -> None:
        estimate = {
            "B": {
                # This interval establishes a positive shift relative to a zero
                # floor, but is much wider than the prospectively promised 0.1.
                "trajectory_t_interval": [0.01, 1.01]
            }
        }
        result = evaluate_realized_precision(
            estimate, {"desired_half_width": 0.1}
        )
        self.assertEqual(result["verdict"], "INDETERMINATE_REALIZED_PRECISION")
        self.assertAlmostEqual(result["realized_half_width"], 0.5)

        result = evaluate_realized_precision(
            {"B": {"trajectory_t_interval": [0.21, 0.39]}},
            {"desired_half_width": 0.1},
        )
        self.assertEqual(result["verdict"], "ADEQUATE_REALIZED_PRECISION")
        self.assertAlmostEqual(result["realized_half_width"], 0.09)

    def test_imported_analysis_dependency_hash_is_part_of_the_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["analysis_code"]["population_estimator"]["sha256"] = "0" * 64
            _, _, errors = validate_confirmation_manifest(changed, path)
            self.assertTrue(
                any("population_estimator hash mismatch" in error for error in errors)
            )

    def test_handwritten_valid_precision_cannot_bypass_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = frozen_fixture(root)
            precision = json.loads(
                Path(manifest["precision_plan"]["path"]).read_text(encoding="utf-8")
            )
            precision["planned_confirmation_trajectories"] += 1
            self.assertTrue(
                any(
                    "does not equal planner recomputation" in error
                    for error in validate_precision_provenance(precision)
                )
            )

    def test_u2_direction_becomes_signed_effect_records_not_l2_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "direction.safetensors"
            save_file({"w": torch.tensor([1.0, 0.0])}, shard)
            direction = {
                "schema_version": "forkcert.qwen3-u2-frozen-direction.v0.1",
                "valid": True,
                "verdict": "VALID_FROZEN_U2_CALIBRATION_DIRECTION",
                "status": "FROZEN_BEFORE_CONFIRMATION",
                "endpoint_name": "U2_calibration_direction_shift",
                "direction": {
                    "normalization_l2": 1.0,
                    "shards": [
                        {
                            "path": str(shard),
                            "sha256": sha(shard),
                            "tensor_key": "w",
                        }
                    ],
                },
            }
            direction_path = root / "direction.json"
            direction_path.write_text(json.dumps(direction), encoding="utf-8")
            pairs = []
            for repeat, sign in ((1, 1.0), (2, -1.0)):
                delta = root / f"delta-{repeat}.safetensors"
                save_file({"w": torch.tensor([sign, 0.0])}, delta)
                pairs.append(
                    {
                        "identity": {
                            "trajectory_id": "confirmation-0",
                            "phase": "early",
                            "state_id": "state-0",
                            "repeat_id": repeat,
                        },
                        "effects": {
                            "U2_delta": {
                                "status": "MEASURED",
                                "artifact": {
                                    "path": str(delta),
                                    "sha256": sha(delta),
                                },
                            }
                        },
                    }
                )
            records, availability, unavailable, errors = collect_u2_direction_endpoint(
                [({}, {"paired_effect_records": pairs})],
                {"path": str(direction_path), "sha256": sha(direction_path)},
            )
            self.assertFalse(errors)
            self.assertFalse(unavailable)
            self.assertEqual(availability["MEASURED"], 2)
            self.assertEqual([record.effect for record in records], [1.0, -1.0])

    def test_completed_source_requires_plan_config_and_metadata_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            data = root / "data"
            results.mkdir()
            metadata = results / "source_dump.metadata.json"
            metadata.write_text('{"source": "recorded"}', encoding="utf-8")
            row = {
                "results_root": str(results),
                "data_root": str(data),
                "capture_plan_sha256": "plan-hash",
                "source_config_sha256": "config-hash",
            }
            audit = {
                "valid": True,
                "verdict": "VALID",
                "plan_sha256": "plan-hash",
                "capture_root": str(data / "captures"),
                "source_evidence": {
                    "config_sha256": "config-hash",
                    "metadata_sha256": sha(metadata),
                    "checks": {"config_exact": True, "seed_exact": True},
                },
            }
            (results / "capture_batch_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            self.assertEqual(validate_confirmation_source_audit(row), [])
            audit["plan_sha256"] = "changed"
            (results / "capture_batch_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            self.assertTrue(
                any(
                    "plan hash" in error
                    for error in validate_confirmation_source_audit(row)
                )
            )

    def test_signflip_can_veto_but_never_promote_primary(self) -> None:
        effects = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.1]
        estimate = {
            "verdicts": {"shift_existence": "REPRODUCIBLE_AVERAGE_SHIFT"},
            "B": {"estimate": sum(effects) / len(effects)},
            "trajectory_rows": [
                {"trajectory_id": f"t{index}", "mean_effect": value}
                for index, value in enumerate(effects)
            ],
        }
        sensitivity, verdict = apply_trajectory_sensitivity(
            estimate,
            {"shift_existence_floor": 0.0},
            0.01,
            EXPECTED_SENSITIVITY,
        )
        self.assertFalse(sensitivity["supports_primary_shift"])
        self.assertEqual(verdict, "INDETERMINATE_METHOD_SENSITIVITY")

        estimate["verdicts"]["shift_existence"] = "NO_STABLE_AVERAGE_DETECTED"
        sensitivity, verdict = apply_trajectory_sensitivity(
            estimate,
            {"shift_existence_floor": 0.0},
            0.05,
            EXPECTED_SENSITIVITY,
        )
        self.assertEqual(sensitivity["status"], "NOT_RUN_PRIMARY_INTERVAL_DID_NOT_ESTABLISH_SHIFT")
        self.assertEqual(verdict, "NO_STABLE_AVERAGE_DETECTED")

    def test_frozen_independent_manifest_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision, inputs, errors = validate_confirmation_manifest(manifest, path)
            self.assertFalse(errors)
            self.assertTrue(precision["valid"])
            self.assertEqual(len(inputs), 8)

    def test_precision_sensitivity_resolution_metadata_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision_path = Path(manifest["precision_plan"]["path"])
            precision = json.loads(precision_path.read_text(encoding="utf-8"))
            precision["sensitivity"]["minimum_trajectories_for_p_value_resolution"] = 8
            precision["sensitivity"]["minimum_attainable_p_at_planned_count"] = 2 / 256
            precision_path.write_text(json.dumps(precision), encoding="utf-8")
            manifest["precision_plan"]["sha256"] = sha(precision_path)
            manifest["analysis"]["sensitivity"] = precision["sensitivity"]
            bank_path = Path(manifest["trajectory_bank"]["path"])
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            bank["precision"]["sha256"] = sha(precision_path)
            bank_path.write_text(json.dumps(bank), encoding="utf-8")
            manifest["trajectory_bank"]["sha256"] = sha(bank_path)
            _, inputs, errors = validate_confirmation_manifest(manifest, path)
            self.assertFalse(errors)
            self.assertEqual(len(inputs), 8)

    def test_calibration_reuse_and_analysis_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["trajectory_inputs"][0]["trajectory_seed"] = 2001284755
            changed["analysis"]["per_interval_alpha"] = 0.10
            _, _, errors = validate_confirmation_manifest(changed, path)
            self.assertTrue(any("reuses calibration trajectory_seed" in error for error in errors))
            self.assertTrue(any("per_interval_alpha" in error for error in errors))

    def test_trajectory_input_cannot_drift_from_prospective_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["trajectory_inputs"][0]["trajectory_seed"] += 1
            _, _, errors = validate_confirmation_manifest(changed, path)
            self.assertTrue(any("frozen trajectory bank" in error for error in errors))

    def test_uninstantiated_template_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "theory_oracle"
            / "QWEN3_BIAS_ORACLE_CONFIRMATION_MANIFEST_TEMPLATE_V0_1.json"
        )
        template = json.loads(path.read_text(encoding="utf-8"))
        precision, inputs, errors = validate_confirmation_manifest(template, path)
        self.assertIsNone(precision)
        self.assertFalse(inputs)
        self.assertTrue(any("not FROZEN" in error for error in errors))

    def test_unsupported_endpoint_cannot_be_frozen_without_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision_path = Path(manifest["precision_plan"]["path"])
            precision = json.loads(precision_path.read_text(encoding="utf-8"))
            precision["multiplicity"]["endpoint_family"] = ["invented_after_calibration"]
            precision["endpoints"] = {
                "invented_after_calibration": {"shift_existence_floor": 0.0}
            }
            precision_path.write_text(json.dumps(precision), encoding="utf-8")
            manifest["precision_plan"]["sha256"] = sha(precision_path)
            manifest["analysis"]["endpoint_family"] = ["invented_after_calibration"]
            _, _, errors = validate_confirmation_manifest(manifest, path)
            self.assertTrue(any("lacks extractors" in error for error in errors))

    def test_frozen_u2_direction_endpoint_is_a_supported_confirmation_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = frozen_fixture(root)
            precision_path = Path(manifest["precision_plan"]["path"])
            old_precision = json.loads(precision_path.read_text(encoding="utf-8"))
            calibration_path = Path(old_precision["inputs"]["calibration"]["path"])
            spec_path = Path(old_precision["inputs"]["spec"]["path"])
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            direction = {
                "schema_version": "forkcert.qwen3-u2-frozen-direction.v0.1",
                "valid": True,
                "verdict": "VALID_FROZEN_U2_CALIBRATION_DIRECTION",
                "status": "FROZEN_BEFORE_CONFIRMATION",
                "precision_contract": {
                    "desired_projection_half_width": 1.0,
                    "projection_variance_floor_sd": 0.01,
                    "projection_shift_existence_floor": 0.0,
                },
                "stability": {
                    "crossfit_projections": [-0.03, -0.01, 0.01, 0.03]
                },
            }
            direction_path = root / "frozen-direction.json"
            direction_path.write_text(json.dumps(direction), encoding="utf-8")
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["endpoint_family"] = ["U2_calibration_direction_shift"]
            spec["phase_conditioned_endpoint_family"] = []
            spec["endpoints"] = {
                "U2_calibration_direction_shift": {
                    "desired_half_width": 1.0,
                    "variance_floor_sd": 0.01,
                    "shift_existence_floor": 0.0,
                    "threshold_sources": threshold_sources(),
                }
            }
            spec["U2_directional_replication"] = {
                "direction_manifest_path": str(direction_path),
                "direction_manifest_sha256": sha(direction_path),
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            precision = plan_confirmation(calibration, spec)
            self.assertTrue(precision["valid"], precision["errors"])
            precision["inputs"] = {
                "calibration": {
                    "path": str(calibration_path),
                    "sha256": sha(calibration_path),
                },
                "spec": {"path": str(spec_path), "sha256": sha(spec_path)},
            }
            precision_path.write_text(json.dumps(precision), encoding="utf-8")
            manifest["precision_plan"]["sha256"] = sha(precision_path)
            manifest["analysis"].update(
                {
                    "endpoint_family": precision["multiplicity"]["endpoint_family"],
                    "phase_conditioned_endpoint_family": precision["multiplicity"][
                        "phase_conditioned_endpoint_family"
                    ],
                    "confirmatory_comparisons": precision["multiplicity"][
                        "confirmatory_comparisons"
                    ],
                    "multiplicity": precision["multiplicity"]["method"],
                    "per_interval_alpha": precision["multiplicity"][
                        "per_interval_alpha"
                    ],
                    "tail_scope": precision["tail"]["scope"],
                    "sensitivity": precision["sensitivity"],
                }
            )
            bank_path = Path(manifest["trajectory_bank"]["path"])
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            bank["precision"]["sha256"] = sha(precision_path)
            bank_path.write_text(json.dumps(bank), encoding="utf-8")
            manifest["trajectory_bank"]["sha256"] = sha(bank_path)
            _, inputs, errors = validate_confirmation_manifest(manifest, path)
            self.assertFalse(errors)
            self.assertEqual(len(inputs), 8)


if __name__ == "__main__":
    unittest.main()
