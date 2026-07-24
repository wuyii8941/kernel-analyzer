#!/usr/bin/env python
"""Evaluate a frozen independent Qwen3 Bias Oracle confirmation bank."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (  # noqa: E402
    collect_endpoint,
    endpoint_extractors,
    load_complete_state_bundles,
    load_json,
    sha256_file,
)
from theory_oracle.bias_oracle_population_v0_2 import (  # noqa: E402
    EffectRecord,
    estimate_scalar_population,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (  # noqa: E402
    FIXED_RESOURCE_EXISTENCE,
    SIGNED_B_ENDPOINT_CLASSES,
    plan_confirmation,
)
from theory_oracle.bias_oracle_trajectory_signflip_v0_1 import (  # noqa: E402
    trajectory_signflip_test,
)
from theory_oracle.qwen3_u2_direction_projection_v0_1 import (  # noqa: E402
    load_frozen_direction,
    project_delta_artifact,
)


SCHEMA_VERSION = "forkcert.qwen3-bias-oracle-confirmation.v0.1"
MANIFEST_VERSION = "forkcert.qwen3-bias-oracle-confirmation-manifest.v0.1"
PRECISION_VERSION = "forkcert.bias-oracle-confirmation-precision.v0.1"
BANK_VERSION = "forkcert.qwen3-bias-oracle-confirmation-bank.v0.1"
DESIGN_VERSION = "forkcert.qwen3-bias-oracle-confirmation-bank-design.v0.1"
REQUIRED_PHASES = ("early", "middle", "late")
EXPECTED_CALIBRATION_EXCLUSION = {
    "trajectory_ids": {f"calibration-{index}" for index in range(4)},
    "trajectory_seeds": {2001284755, 1810598814, 1677250702, 797459759},
    "data_slice_ids": {
        "forkcert_builtin_arithmetic[7296:7360]",
        "forkcert_builtin_arithmetic[3840:3904]",
        "forkcert_builtin_arithmetic[5696:5760]",
        "forkcert_builtin_arithmetic[3200:3264]",
    },
}
EXPECTED_SENSITIVITY = {
    "method": "TRAJECTORY_RADEMACHER_SIGN_FLIP_STUDENTIZED",
    "role": "VETO_PRIMARY_SHIFT_ONLY",
    "exact_max_trajectories": 16,
    "monte_carlo_draws": 99999,
    "monte_carlo_seed": 172904,
}
U2_ENDPOINT = "U2_calibration_direction_shift"
ANALYSIS_CODE_PATHS = {
    "precision_planner": ROOT
    / "theory_oracle"
    / "bias_oracle_confirmation_precision_v0_1.py",
    "population_estimator": ROOT / "theory_oracle" / "bias_oracle_population_v0_2.py",
    "trajectory_sensitivity": ROOT
    / "theory_oracle"
    / "bias_oracle_trajectory_signflip_v0_1.py",
    "record_loader": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_calibration_records_v0_1.py",
    "u2_direction_projector": ROOT
    / "theory_oracle"
    / "qwen3_u2_direction_projection_v0_1.py",
}
TRAJECTORY_FIELDS = (
    "trajectory_id",
    "trajectory_seed",
    "data_slice_id",
    "source_config_path",
    "source_config_sha256",
    "capture_plan_path",
    "capture_plan_sha256",
    "results_root",
    "data_root",
)


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_confirmation_source_audit(row: dict[str, Any]) -> list[str]:
    """Verify that one completed source run is bound to its frozen config/plan."""
    errors: list[str] = []
    results_root = Path(row["results_root"])
    data_root = Path(row["data_root"])
    audit_path = results_root / "capture_batch_audit.json"
    metadata_path = results_root / "source_dump.metadata.json"
    if not audit_path.is_file():
        return ["capture batch audit is missing"]
    if not metadata_path.is_file():
        return ["source metadata is missing"]
    try:
        audit = load_json(audit_path)
    except (OSError, json.JSONDecodeError):
        return ["capture batch audit is unreadable"]
    evidence = audit.get("source_evidence") or {}
    checks = evidence.get("checks") or {}
    if audit.get("valid") is not True or audit.get("verdict") != "VALID":
        errors.append("capture batch audit is not valid")
    if audit.get("plan_sha256") != row["capture_plan_sha256"]:
        errors.append("capture batch audit plan hash mismatch")
    if Path(audit.get("capture_root", "")).resolve() != data_root / "captures":
        errors.append("capture batch audit data root mismatch")
    if evidence.get("config_sha256") != row["source_config_sha256"]:
        errors.append("capture batch audit config hash mismatch")
    if evidence.get("metadata_sha256") != sha256_file(metadata_path):
        errors.append("capture batch audit metadata hash mismatch")
    if not checks or not all(value is True for value in checks.values()):
        errors.append("capture batch audit source-binding checks are incomplete")
    return errors


def validate_precision_provenance(precision: dict[str, Any]) -> list[str]:
    """Recompute a stored precision plan from its hashed calibration/spec inputs."""
    errors: list[str] = []
    inputs = precision.get("inputs")
    if not isinstance(inputs, dict):
        return ["precision plan lacks recomputable input provenance"]
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("calibration", "spec"):
        link = inputs.get(name, {})
        path_value = link.get("path") if isinstance(link, dict) else None
        expected = link.get("sha256") if isinstance(link, dict) else None
        if not isinstance(path_value, str) or not isinstance(expected, str):
            errors.append(f"precision {name} input link is missing")
            continue
        path = Path(path_value).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"precision {name} input identity failed")
            continue
        loaded[name] = load_json(path)
    if errors:
        return errors
    recomputed = plan_confirmation(loaded["calibration"], loaded["spec"])
    stored = {key: value for key, value in precision.items() if key != "inputs"}
    if stored != recomputed:
        errors.append("stored precision plan does not equal planner recomputation")
    return errors


def collect_u2_direction_endpoint(
    state_bundles: list[tuple[dict[str, Any], dict[str, Any]]],
    direction_link: dict[str, Any],
) -> tuple[list[EffectRecord], Counter[str], list[dict[str, Any]], list[str]]:
    records: list[EffectRecord] = []
    availability: Counter[str] = Counter()
    unavailable: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        direction = load_frozen_direction(direction_link)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return records, availability, unavailable, [str(error)]
    for _, bundle in state_bundles:
        for pair in bundle["paired_effect_records"]:
            identity = pair["identity"]
            u2 = pair.get("effects", {}).get("U2_delta", {})
            if u2.get("status") != "MEASURED":
                availability["UNAVAILABLE"] += 1
                unavailable.append(dict(identity))
                continue
            try:
                value = project_delta_artifact(u2.get("artifact", {}), direction)
            except (OSError, KeyError, TypeError, ValueError) as error:
                availability["INVALID_ARTIFACT"] += 1
                unavailable.append(dict(identity))
                errors.append(
                    f"{identity.get('trajectory_id')}/{identity.get('state_id')}/"
                    f"repeat-{identity.get('repeat_id')}: {error}"
                )
                continue
            availability["MEASURED"] += 1
            records.append(
                EffectRecord(
                    trajectory_id=str(identity["trajectory_id"]),
                    phase=str(identity["phase"]),
                    state_id=str(identity["state_id"]),
                    repeat_id=int(identity["repeat_id"]),
                    effect=float(value),
                )
            )
    return records, availability, unavailable, errors


def validate_confirmation_manifest(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_VERSION:
        errors.append("unsupported confirmation manifest schema")
    if manifest.get("status") != "FROZEN_BEFORE_CONFIRMATION":
        errors.append("confirmation manifest is not FROZEN_BEFORE_CONFIRMATION")
    if manifest.get("query_id") != "Q-R":
        errors.append("confirmation query must be Q-R")
    if manifest.get("trajectory_anchor") != "EAGER_TRAJECTORY":
        errors.append("confirmation anchor must be EAGER_TRAJECTORY")

    evaluator_link = manifest.get("evaluator", {})
    evaluator_path_value = evaluator_link.get("path") if isinstance(evaluator_link, dict) else None
    if not isinstance(evaluator_path_value, str):
        errors.append("confirmation evaluator path is uninstantiated")
    else:
        evaluator_path = resolve_from(manifest_path.parent, evaluator_path_value)
        if evaluator_path != Path(__file__).resolve():
            errors.append("confirmation evaluator path is not this frozen evaluator")
        elif sha256_file(evaluator_path) != evaluator_link.get("sha256"):
            errors.append("confirmation evaluator hash mismatch")

    analysis_code = manifest.get("analysis_code")
    if not isinstance(analysis_code, dict):
        errors.append("confirmation analysis code map is missing")
        analysis_code = {}
    if set(analysis_code) != set(ANALYSIS_CODE_PATHS):
        errors.append("confirmation analysis code dependency set drifted")
    for name, expected_path in ANALYSIS_CODE_PATHS.items():
        link = analysis_code.get(name, {})
        path_value = link.get("path") if isinstance(link, dict) else None
        if not isinstance(path_value, str):
            errors.append(f"analysis code {name} path is missing")
            continue
        path = resolve_from(manifest_path.parent, path_value)
        if path != expected_path.resolve():
            errors.append(f"analysis code {name} path drifted")
        elif sha256_file(path) != link.get("sha256"):
            errors.append(f"analysis code {name} hash mismatch")

    precision_link = manifest.get("precision_plan", {})
    precision: dict[str, Any] | None = None
    precision_path_value = precision_link.get("path") if isinstance(precision_link, dict) else None
    if not isinstance(precision_path_value, str):
        errors.append("precision plan path is uninstantiated")
    else:
        precision_path = resolve_from(manifest_path.parent, precision_path_value)
        if not precision_path.is_file():
            errors.append("precision plan file is missing")
        elif sha256_file(precision_path) != precision_link.get("sha256"):
            errors.append("precision plan hash mismatch")
        else:
            precision = load_json(precision_path)
            if (
                precision.get("schema_version") != PRECISION_VERSION
                or precision.get("valid") is not True
                or precision.get("verdict") != "VALID_FROZEN_PRECISION_PLAN"
            ):
                errors.append("linked precision plan is not valid/frozen")
            else:
                errors.extend(validate_precision_provenance(precision))

    bank_link = manifest.get("trajectory_bank", {})
    bank: dict[str, Any] | None = None
    bank_path_value = bank_link.get("path") if isinstance(bank_link, dict) else None
    if not isinstance(bank_path_value, str):
        errors.append("confirmation trajectory bank path is uninstantiated")
    else:
        bank_path = resolve_from(manifest_path.parent, bank_path_value)
        if not bank_path.is_file():
            errors.append("confirmation trajectory bank is missing")
        elif sha256_file(bank_path) != bank_link.get("sha256"):
            errors.append("confirmation trajectory bank hash mismatch")
        else:
            bank = load_json(bank_path)
            if (
                bank.get("schema_version") != BANK_VERSION
                or bank.get("valid") is not True
                or bank.get("verdict")
                != "VALID_FROZEN_CONFIRMATION_TRAJECTORY_BANK"
            ):
                errors.append("linked confirmation trajectory bank is invalid")
            bank_precision = bank.get("precision", {})
            if isinstance(precision_link, dict) and (
                bank_precision.get("sha256") != precision_link.get("sha256")
                or bank_precision.get("planned_confirmation_trajectories")
                != precision_link.get("planned_confirmation_trajectories")
            ):
                errors.append("trajectory bank and precision plan are not bound")
            design_link = bank.get("design", {})
            design_path_value = (
                design_link.get("path") if isinstance(design_link, dict) else None
            )
            if not isinstance(design_path_value, str):
                errors.append("trajectory bank design link is missing")
            else:
                design_path = Path(design_path_value).resolve()
                if not design_path.is_file() or sha256_file(design_path) != design_link.get(
                    "sha256"
                ):
                    errors.append("trajectory bank design identity failed")
                else:
                    design = load_json(design_path)
                    if (
                        design.get("schema_version") != DESIGN_VERSION
                        or design.get("status")
                        != "FROZEN_BEFORE_COMPLETE_CALIBRATION_RESULTS"
                    ):
                        errors.append("trajectory bank design was not frozen prospectively")

    inputs = manifest.get("trajectory_inputs")
    if not isinstance(inputs, list):
        errors.append("trajectory_inputs must be a list")
        inputs = []
    if bank is not None and inputs != bank.get("trajectory_specs"):
        errors.append("trajectory_inputs do not exactly match the frozen trajectory bank")
    planned = (
        precision.get("planned_confirmation_trajectories")
        if precision is not None
        else None
    )
    if not isinstance(planned, int) or planned < 8:
        errors.append("precision plan lacks a valid trajectory count")
    elif len(inputs) != planned:
        errors.append(f"trajectory input count {len(inputs)} does not match planned {planned}")
    if isinstance(precision_link, dict) and precision_link.get(
        "planned_confirmation_trajectories"
    ) != planned:
        errors.append("manifest precision trajectory count does not match linked plan")

    for index, row in enumerate(inputs):
        if not isinstance(row, dict):
            errors.append(f"trajectory_inputs[{index}] is not an object")
            continue
        missing = [field for field in TRAJECTORY_FIELDS if field not in row]
        if missing:
            errors.append(f"trajectory_inputs[{index}] missing {missing}")
    for field in ("trajectory_id", "trajectory_seed", "data_slice_id"):
        values = [row.get(field) for row in inputs if isinstance(row, dict)]
        if None in values or len(values) != len(set(values)):
            errors.append(f"confirmation {field} values must be present and unique")
    for field in (
        "source_config_path",
        "source_config_sha256",
        "capture_plan_path",
        "capture_plan_sha256",
        "results_root",
        "data_root",
    ):
        values = [row.get(field) for row in inputs if isinstance(row, dict)]
        if None in values or len(values) != len(set(values)):
            errors.append(f"confirmation {field} values must be present and unique")

    exclusion = manifest.get("calibration_exclusion", {})
    for field, exclusion_field in (
        ("trajectory_id", "trajectory_ids"),
        ("trajectory_seed", "trajectory_seeds"),
        ("data_slice_id", "data_slice_ids"),
    ):
        forbidden = set(exclusion.get(exclusion_field, [])) if isinstance(exclusion, dict) else set()
        if forbidden != EXPECTED_CALIBRATION_EXCLUSION[exclusion_field]:
            errors.append(f"calibration exclusion list {exclusion_field} is incomplete or altered")
        overlap = forbidden.intersection(
            row.get(field) for row in inputs if isinstance(row, dict)
        )
        if overlap:
            errors.append(f"confirmation reuses calibration {field}: {sorted(overlap, key=str)}")

    analysis = manifest.get("analysis", {})
    if not isinstance(analysis, dict):
        errors.append("analysis must be an object")
        analysis = {}
    if analysis.get("calibration_trajectories_in_confirmation_df") is not False:
        errors.append("calibration trajectories must be excluded from confirmation df")
    if analysis.get("correctness_authority") is not None:
        errors.append("this implementation-relative confirmation does not accept a correctness authority")
    if precision is not None:
        precision_multiplicity = precision.get("multiplicity", {})
        precision_family = precision_multiplicity.get("endpoint_family")
        supported_endpoints = set(endpoint_extractors()) | {U2_ENDPOINT}
        if not isinstance(precision_family, list) or not precision_family:
            errors.append("precision endpoint family is empty or invalid")
        else:
            unsupported = sorted(set(precision_family) - supported_endpoints)
            if unsupported:
                errors.append(f"precision endpoint family lacks extractors: {unsupported}")
            if set(precision.get("endpoints", {})) != set(precision_family):
                errors.append("precision endpoint plans do not exactly match endpoint family")
        precision_sensitivity = precision.get("sensitivity", {})
        if any(
            precision_sensitivity.get(key) != expected
            for key, expected in EXPECTED_SENSITIVITY.items()
        ):
            errors.append("precision sensitivity procedure drifted from the frozen evaluator")
        expected_analysis = {
            "endpoint_family": precision_family,
            "phase_conditioned_endpoint_family": precision_multiplicity.get(
                "phase_conditioned_endpoint_family"
            ),
            "confirmatory_comparisons": precision_multiplicity.get(
                "confirmatory_comparisons"
            ),
            "multiplicity": precision_multiplicity.get("method"),
            "per_interval_alpha": precision_multiplicity.get("per_interval_alpha"),
            "tail_scope": precision.get("tail", {}).get("scope"),
            "sensitivity": precision.get("sensitivity"),
        }
        for key, expected in expected_analysis.items():
            observed = analysis.get(key)
            if isinstance(expected, float) and isinstance(observed, (int, float)):
                matches = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
            else:
                matches = observed == expected
            if not matches:
                errors.append(f"analysis.{key} does not match precision plan")

    validated_inputs: list[dict[str, Any]] = []
    for index, row in enumerate(inputs):
        if not isinstance(row, dict) or any(field not in row for field in TRAJECTORY_FIELDS):
            continue
        plan_path = resolve_from(manifest_path.parent, str(row["capture_plan_path"]))
        config_path = resolve_from(
            manifest_path.parent, str(row["source_config_path"])
        )
        if not config_path.is_file():
            errors.append(f"trajectory_inputs[{index}] source config is missing")
            continue
        if sha256_file(config_path) != row["source_config_sha256"]:
            errors.append(f"trajectory_inputs[{index}] source config hash mismatch")
            continue
        if not plan_path.is_file():
            errors.append(f"trajectory_inputs[{index}] capture plan is missing")
            continue
        if sha256_file(plan_path) != row["capture_plan_sha256"]:
            errors.append(f"trajectory_inputs[{index}] capture plan hash mismatch")
            continue
        plan = load_json(plan_path)
        if plan.get("schema_version") != "forkcert.multi-transition-capture-plan.v0.1":
            errors.append(f"trajectory_inputs[{index}] capture plan schema is invalid")
        identity = plan.get("identity", {})
        for field in ("trajectory_id", "trajectory_seed", "data_slice_id"):
            if identity.get(field) != row[field]:
                errors.append(f"trajectory_inputs[{index}] {field} disagrees with capture plan")
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = config.get("dataset", {})
        training = config.get("training", {})
        offset = dataset.get("offset")
        count = dataset.get("max_prompts")
        config_slice = (
            f"{dataset.get('name')}[{offset}:{offset + count}]"
            if isinstance(offset, int) and isinstance(count, int)
            else None
        )
        if training.get("seed") != row["trajectory_seed"]:
            errors.append(f"trajectory_inputs[{index}] source config seed mismatch")
        if training.get("max_steps") != 300:
            errors.append(f"trajectory_inputs[{index}] source horizon is not 300")
        if config_slice != row["data_slice_id"]:
            errors.append(f"trajectory_inputs[{index}] source config data slice mismatch")
        if identity.get("query_id") != "Q-R" or identity.get("trajectory_anchor") != "EAGER_TRAJECTORY":
            errors.append(f"trajectory_inputs[{index}] capture plan has wrong query/anchor")
        targets = plan.get("targets", [])
        if len(targets) != 24 or Counter(target.get("phase") for target in targets) != Counter(
            {"early": 8, "middle": 8, "late": 8}
        ):
            errors.append(f"trajectory_inputs[{index}] capture plan lacks frozen 8x3 layout")
        expected_populations = {
            "early": "1:100",
            "middle": "101:200",
            "late": "201:300",
        }
        if any(
            target.get("eligible_step_population")
            != expected_populations.get(target.get("phase"))
            for target in targets
        ):
            errors.append(f"trajectory_inputs[{index}] eligible step populations drifted")
        if len({target.get("state_id") for target in targets}) != len(targets):
            errors.append(f"trajectory_inputs[{index}] duplicate state IDs")
        if len({target.get("optimizer_step") for target in targets}) != len(targets):
            errors.append(f"trajectory_inputs[{index}] duplicate optimizer steps")
        validated_inputs.append(
            {
                **row,
                "source_config_path": str(config_path),
                "capture_plan_path": str(plan_path),
                "results_root": str(
                    resolve_from(manifest_path.parent, str(row["results_root"]))
                ),
                "plan": plan,
            }
        )
    return precision, validated_inputs, errors


def apply_trajectory_sensitivity(
    estimate: dict[str, Any],
    endpoint_plan: dict[str, Any],
    interval_alpha: float,
    sensitivity_spec: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    shift_verdict = estimate["verdicts"]["shift_existence"]
    if shift_verdict != "REPRODUCIBLE_AVERAGE_SHIFT":
        return (
            {
                "status": "NOT_RUN_PRIMARY_INTERVAL_DID_NOT_ESTABLISH_SHIFT",
                "role": "sensitivity cannot promote a failed primary claim",
            },
            shift_verdict,
        )
    trajectory_effects = [
        float(row["mean_effect"]) for row in estimate["trajectory_rows"]
    ]
    point_estimate = float(estimate["B"]["estimate"])
    measurement_floor = float(endpoint_plan["shift_existence_floor"])
    null_center = measurement_floor if point_estimate > 0 else -measurement_floor
    sensitivity = trajectory_signflip_test(
        trajectory_effects,
        null_center=null_center,
        exact_max_trajectories=int(sensitivity_spec["exact_max_trajectories"]),
        monte_carlo_draws=int(sensitivity_spec["monte_carlo_draws"]),
        monte_carlo_seed=int(sensitivity_spec["monte_carlo_seed"]),
    )
    sensitivity["decision_alpha"] = float(interval_alpha)
    sensitivity["supports_primary_shift"] = (
        sensitivity["two_sided_p_value"] <= float(interval_alpha)
    )
    final_shift_verdict = (
        shift_verdict
        if sensitivity["supports_primary_shift"]
        else "INDETERMINATE_METHOD_SENSITIVITY"
    )
    return sensitivity, final_shift_verdict


def evaluate_realized_precision(
    estimate: dict[str, Any], endpoint_plan: dict[str, Any]
) -> dict[str, Any]:
    """Check the confirmation interval against the prospectively frozen width.

    Prospective sample sizing is not evidence that the realised sample actually
    achieved its promised precision: confirmation variance can exceed the
    calibration-based planning variance.  This gate is deliberately separate
    from shift existence and practical materiality.
    """
    interval = estimate.get("B", {}).get("trajectory_t_interval")
    planning_mode = endpoint_plan.get("planning_mode", "PRECISION_TARGETED")
    if planning_mode == FIXED_RESOURCE_EXISTENCE:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or interval[0] is None
            or interval[1] is None
        ):
            return {
                "verdict": "INDETERMINATE_REALIZED_PRECISION",
                "planning_mode": planning_mode,
                "desired_half_width": None,
                "realized_half_width": None,
                "reason": "fixed-resource mode still requires a finite reported interval",
            }
        lower, upper = float(interval[0]), float(interval[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            return {
                "verdict": "INDETERMINATE_REALIZED_PRECISION",
                "planning_mode": planning_mode,
                "desired_half_width": None,
                "realized_half_width": None,
                "reason": "confirmation interval is invalid",
            }
        return {
            "verdict": "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
            "planning_mode": planning_mode,
            "desired_half_width": None,
            "realized_half_width": (upper - lower) / 2.0,
            "reason": "fixed-resource existence mode reports interval width without inventing a target precision",
        }

    desired = float(endpoint_plan["desired_half_width"])
    if (
        not math.isfinite(desired)
        or desired <= 0.0
        or not isinstance(interval, list)
        or len(interval) != 2
        or interval[0] is None
        or interval[1] is None
    ):
        return {
            "verdict": "INDETERMINATE_REALIZED_PRECISION",
            "desired_half_width": desired,
            "realized_half_width": None,
            "reason": "finite confirmation interval and positive frozen width are required",
        }
    lower, upper = float(interval[0]), float(interval[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        return {
            "verdict": "INDETERMINATE_REALIZED_PRECISION",
            "desired_half_width": desired,
            "realized_half_width": None,
            "reason": "confirmation interval is invalid",
        }
    realized = (upper - lower) / 2.0
    adequate = realized <= desired * (1.0 + 1e-12)
    return {
        "verdict": (
            "ADEQUATE_REALIZED_PRECISION"
            if adequate
            else "INDETERMINATE_REALIZED_PRECISION"
        ),
        "desired_half_width": desired,
        "realized_half_width": realized,
        "reason": (
            "confirmation interval met the prospectively frozen width"
            if adequate
            else "confirmation variance exceeded the precision delivered by the frozen sample size"
        ),
    }


def evaluate_operator_attribution_eligibility(
    final_shift_verdict: str, realized_precision_verdict: str
) -> dict[str, Any]:
    blockers: list[str] = []
    if final_shift_verdict != "REPRODUCIBLE_AVERAGE_SHIFT":
        blockers.append(final_shift_verdict)
    if realized_precision_verdict not in {
        "ADEQUATE_REALIZED_PRECISION",
        "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
    }:
        blockers.append(realized_precision_verdict)
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "claim_scope": "contribution to an implementation-relative average shift B only",
    }


def realized_precision_decision_is_resolved(verdict: str) -> bool:
    """Whether the selected planning mode produced the interval report it promised."""
    return verdict in {
        "ADEQUATE_REALIZED_PRECISION",
        "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED",
    }


def classify_oracle_disposition(
    final_shift_verdict: str, realized_precision_verdict: str
) -> str:
    """Combine two decision axes without turning non-detection into equivalence."""
    adequate = realized_precision_verdict == "ADEQUATE_REALIZED_PRECISION"
    fixed_resource = (
        realized_precision_verdict
        == "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED"
    )
    if final_shift_verdict == "REPRODUCIBLE_AVERAGE_SHIFT":
        if fixed_resource:
            return "CONFIRMED_IMPLEMENTATION_RELATIVE_AVERAGE_SHIFT_FIXED_RESOURCE"
        return (
            "CONFIRMED_IMPLEMENTATION_RELATIVE_AVERAGE_SHIFT"
            if adequate
            else "AVERAGE_SHIFT_DETECTED_BUT_TARGET_PRECISION_MISSED"
        )
    if final_shift_verdict == "NO_STABLE_AVERAGE_DETECTED":
        if fixed_resource:
            return "NO_STABLE_AVERAGE_DETECTED_AT_FIXED_RESOURCE"
        return (
            "NO_AVERAGE_SHIFT_BEYOND_FLOOR_DETECTED_AT_TARGET_PRECISION"
            if adequate
            else "INDETERMINATE_NONDETECTION_WITH_INADEQUATE_PRECISION"
        )
    if final_shift_verdict == "INDETERMINATE_METHOD_SENSITIVITY":
        return "INDETERMINATE_METHOD_SENSITIVITY"
    return "INDETERMINATE_SHIFT_EXISTENCE"


def classify_practical_materiality(
    lower: float, upper: float, practical_tolerance: float | None, shift_verdict: str
) -> str:
    if practical_tolerance is None:
        return "UNINSTANTIATED_MATERIALITY"
    if lower > practical_tolerance or upper < -practical_tolerance:
        return "MATERIAL_AVERAGE_SHIFT"
    if lower >= -practical_tolerance and upper <= practical_tolerance:
        return "PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT"
    if shift_verdict == "REPRODUCIBLE_AVERAGE_SHIFT":
        return "DETECTED_BUT_MATERIALITY_INDETERMINATE"
    return "INDETERMINATE_MATERIALITY"


def evaluate_phase_conditioned_claims(
    estimate: dict[str, Any],
    endpoint_plan: dict[str, Any],
    interval_alpha: float,
    sensitivity_spec: dict[str, Any],
) -> dict[str, Any]:
    rows = estimate.get("conditional_B", {}).get("predeclared_phase_rows", [])
    by_phase = {row.get("phase"): row for row in rows}
    planned_phases = endpoint_plan.get("phase_conditioned_plans", {})
    if set(by_phase) != set(REQUIRED_PHASES) or set(planned_phases) != set(
        REQUIRED_PHASES
    ):
        return {
            "status": "INVALID_INCOMPLETE_PHASE_CONDITIONED_FAMILY",
            "claims": {},
            "errors": ["all three predeclared phases and precision plans are required"],
        }
    claims: dict[str, Any] = {}
    for phase in REQUIRED_PHASES:
        row = by_phase[phase]
        interval = row.get("trajectory_t_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            return {
                "status": "INVALID_INCOMPLETE_PHASE_CONDITIONED_FAMILY",
                "claims": claims,
                "errors": [f"{phase}: trajectory interval is missing"],
            }
        lower, upper = float(interval[0]), float(interval[1])
        trajectory_rows = row.get("trajectory_rows")
        if (
            not isinstance(trajectory_rows, list)
            or len(trajectory_rows) < 2
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("trajectory_id"), str)
                or not isinstance(item.get("mean_effect"), (int, float))
                for item in trajectory_rows
            )
            or len({item["trajectory_id"] for item in trajectory_rows})
            != len(trajectory_rows)
        ):
            return {
                "status": "INVALID_INCOMPLETE_PHASE_CONDITIONED_FAMILY",
                "claims": claims,
                "errors": [f"{phase}: trajectory identity/effect rows are invalid"],
            }
        floor = float(endpoint_plan["shift_existence_floor"])
        primary = (
            "REPRODUCIBLE_AVERAGE_SHIFT"
            if lower > floor or upper < -floor
            else "NO_STABLE_AVERAGE_DETECTED"
        )
        phase_estimate = {
            "verdicts": {"shift_existence": primary},
            "B": {
                "estimate": float(row["estimate"]),
                "trajectory_t_interval": [lower, upper],
            },
            "trajectory_rows": trajectory_rows,
        }
        sensitivity, final = apply_trajectory_sensitivity(
            phase_estimate,
            endpoint_plan,
            interval_alpha,
            sensitivity_spec,
        )
        precision = evaluate_realized_precision(phase_estimate, endpoint_plan)
        materiality = classify_practical_materiality(
            lower,
            upper,
            endpoint_plan.get("practical_tolerance"),
            final,
        )
        eligibility = evaluate_operator_attribution_eligibility(
            final, precision["verdict"]
        )
        claims[phase] = {
            "estimate": row,
            "sensitivity": sensitivity,
            "realized_precision": precision,
            "final_shift_verdict": final,
            "oracle_disposition": classify_oracle_disposition(
                final, precision["verdict"]
            ),
            "operator_attribution_eligibility": eligibility,
            "claim_scope": f"B under P(state | phase={phase})",
            "decision_axes": {
                "shift_existence": final,
                "realized_precision": precision["verdict"],
                "practical_materiality": materiality,
                "correctness": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
                "long_run_training_impact": "UNINSTANTIATED_ONE_STEP_ORACLE_ONLY",
            },
        }
    return {
        "status": "MEASURED_FROZEN_ALL_PHASE_CONFIRMATION_FAMILY",
        "claims": claims,
        "errors": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    precision, inputs, errors = validate_confirmation_manifest(manifest, manifest_path)

    all_bundles: list[tuple[dict[str, Any], dict[str, Any]]] = []
    state_evidence: list[dict[str, Any]] = []
    if not errors:
        for row in inputs:
            errors.extend(
                f"{row['trajectory_id']}: {error}"
                for error in validate_confirmation_source_audit(row)
            )
            bundles, evidence, current_errors = load_complete_state_bundles(
                row["plan"], Path(row["results_root"])
            )
            errors.extend(
                f"{row['trajectory_id']}: {error}" for error in current_errors
            )
            if len(bundles) != 24:
                errors.append(f"{row['trajectory_id']}: complete states {len(bundles)}/24")
            all_bundles.extend(bundles)
            state_evidence.extend(
                {"trajectory_id": row["trajectory_id"], **item} for item in evidence
            )

    planned = (
        precision.get("planned_confirmation_trajectories")
        if precision is not None
        else 0
    )
    construction_valid = not errors and len(all_bundles) == planned * 24
    endpoint_results: dict[str, Any] = {}
    endpoint_family = (
        precision.get("multiplicity", {}).get("endpoint_family", [])
        if precision is not None
        else []
    )
    phase_conditioned_family = set(
        precision.get("multiplicity", {}).get(
            "phase_conditioned_endpoint_family", []
        )
        if precision is not None
        else []
    )
    interval_alpha = (
        precision.get("multiplicity", {}).get("per_interval_alpha")
        if precision is not None
        else None
    )
    extractors = endpoint_extractors()
    all_family_endpoints_available = construction_valid
    all_family_endpoint_decisions_resolved = construction_valid
    operator_ready_endpoints: list[str] = []
    operator_blocked_endpoints: dict[str, list[str]] = {}
    for name in endpoint_family:
        if name not in extractors and name != U2_ENDPOINT:
            endpoint_results[name] = {
                "status": "UNINSTANTIATED_EXTRACTOR",
                "population_shift_inference_allowed": False,
            }
            all_family_endpoints_available = False
            all_family_endpoint_decisions_resolved = False
            continue
        if name == U2_ENDPOINT:
            endpoint_plan = precision["endpoints"][name]
            endpoint_class = endpoint_plan.get(
                "endpoint_class", "SIGNED_CALIBRATION_DIRECTION_UPDATE_ENDPOINT"
            )
            records, availability, unavailable, projection_errors = (
                collect_u2_direction_endpoint(
                    all_bundles, endpoint_plan.get("direction", {})
                )
            )
            errors.extend(f"{name}: {error}" for error in projection_errors)
        else:
            endpoint_class, extractor = extractors[name]
            if endpoint_class not in SIGNED_B_ENDPOINT_CLASSES:
                endpoint_results[name] = {
                    "status": "INVALID_UNSIGNED_BIAS_ENDPOINT",
                    "endpoint_class": endpoint_class,
                    "population_shift_inference_allowed": False,
                }
                errors.append(
                    f"{name}: endpoint class {endpoint_class!r} is not a signed Bias endpoint"
                )
                all_family_endpoints_available = False
                all_family_endpoint_decisions_resolved = False
                continue
            records, availability, unavailable = collect_endpoint(
                all_bundles, extractor
            )
        complete = construction_valid and len(records) == planned * 24 * 2
        if not complete:
            endpoint_results[name] = {
                "status": "UNAVAILABLE_FULL_FROZEN_CONFIRMATION",
                "endpoint_class": endpoint_class,
                "availability": dict(availability),
                "unavailable_records": unavailable,
                "population_shift_inference_allowed": False,
            }
            all_family_endpoints_available = False
            all_family_endpoint_decisions_resolved = False
            continue
        endpoint_plan = precision["endpoints"][name]
        tail = precision.get("tail", {})
        explicit_tail = tail.get("scope") == "EXPLICIT_PREVALENCE_COVERAGE"
        estimate = estimate_scalar_population(
            records,
            required_phases=REQUIRED_PHASES,
            min_confirmation_trajectories=planned,
            measurement_floor=float(endpoint_plan["shift_existence_floor"]),
            practical_tolerance=(
                float(endpoint_plan["practical_tolerance"])
                if endpoint_plan.get("practical_tolerance") is not None
                else None
            ),
            interval_alpha=float(interval_alpha),
            tail_prevalence=(float(tail["minimum_prevalence"]) if explicit_tail else None),
            tail_alpha=(float(tail["alpha"]) if explicit_tail else 0.05),
        )
        sensitivity, final_shift_verdict = apply_trajectory_sensitivity(
            estimate,
            endpoint_plan,
            float(interval_alpha),
            precision["sensitivity"],
        )
        realized_precision = evaluate_realized_precision(estimate, endpoint_plan)
        if final_shift_verdict == "INDETERMINATE_METHOD_SENSITIVITY":
            all_family_endpoint_decisions_resolved = False
        if not realized_precision_decision_is_resolved(
            realized_precision["verdict"]
        ):
            all_family_endpoint_decisions_resolved = False
        attribution_eligibility = evaluate_operator_attribution_eligibility(
            final_shift_verdict, realized_precision["verdict"]
        )
        if attribution_eligibility["eligible"]:
            operator_ready_endpoints.append(name)
        else:
            operator_blocked_endpoints[name] = attribution_eligibility["blockers"]
        phase_confirmation: dict[str, Any] | None = None
        if name in phase_conditioned_family:
            phase_confirmation = evaluate_phase_conditioned_claims(
                estimate,
                endpoint_plan,
                float(interval_alpha),
                precision["sensitivity"],
            )
            if phase_confirmation["status"] != "MEASURED_FROZEN_ALL_PHASE_CONFIRMATION_FAMILY":
                errors.extend(
                    f"{name}/phase-family: {error}"
                    for error in phase_confirmation["errors"]
                )
                all_family_endpoints_available = False
                all_family_endpoint_decisions_resolved = False
            else:
                for phase, claim in phase_confirmation["claims"].items():
                    claim_id = f"{name}::phase={phase}"
                    eligibility = claim["operator_attribution_eligibility"]
                    if eligibility["eligible"]:
                        operator_ready_endpoints.append(claim_id)
                    else:
                        operator_blocked_endpoints[claim_id] = eligibility["blockers"]
                    if (
                        claim["final_shift_verdict"]
                        == "INDETERMINATE_METHOD_SENSITIVITY"
                        or not realized_precision_decision_is_resolved(
                            claim["realized_precision"]["verdict"]
                        )
                    ):
                        all_family_endpoint_decisions_resolved = False
        endpoint_results[name] = {
            "status": "MEASURED_CONFIRMATION_ENDPOINT",
            "endpoint_class": endpoint_class,
            "availability": dict(availability),
            "unavailable_records": unavailable,
            "estimate": estimate,
            "sensitivity": sensitivity,
            "realized_precision": realized_precision,
            "final_shift_verdict": final_shift_verdict,
            "oracle_disposition": classify_oracle_disposition(
                final_shift_verdict, realized_precision["verdict"]
            ),
            "decision_axes": {
                "shift_existence": final_shift_verdict,
                "realized_precision": realized_precision["verdict"],
                "practical_materiality": estimate["verdicts"]["materiality"],
                "correctness": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
                "long_run_training_impact": "UNINSTANTIATED_ONE_STEP_ORACLE_ONLY",
            },
            "operator_attribution_eligibility": attribution_eligibility,
            "phase_conditioned_confirmation": phase_confirmation,
            "population_shift_inference_allowed": True,
            "correctness_claim_allowed": False,
        }
        if name == U2_ENDPOINT:
            endpoint_results[name]["frozen_direction"] = endpoint_plan.get(
                "direction"
            )

    construction_valid = construction_valid and not errors
    if not construction_valid:
        all_family_endpoints_available = False
        all_family_endpoint_decisions_resolved = False
    valid = construction_valid
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": (
            "VALID_CONFIRMATION_CONSTRUCTION"
            if valid
            else "INVALID_CONFIRMATION_CONSTRUCTION"
        ),
        "construction": {
            "errors": errors,
            "independent_confirmation_trajectories": planned if valid else None,
            "states": len(all_bundles),
            "top_level_df": planned - 1 if valid else None,
            "calibration_trajectories_in_df": False,
            "weighting": "equal trajectory; equal phase; equal state within phase",
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "endpoints": endpoint_results,
        "joint_family_verdict_allowed": (
            all_family_endpoints_available
            and bool(
                precision.get("multiplicity", {}).get("joint_claim_allowed")
                if precision is not None
                else False
            )
        ),
        "joint_family_construction_complete": all_family_endpoints_available,
        "all_shift_and_precision_decisions_resolved": all_family_endpoint_decisions_resolved,
        "operator_attribution_gate": {
            "ready_endpoints": operator_ready_endpoints,
            "blocked_endpoints": operator_blocked_endpoints,
            "automatic_operator_launch": False,
            "effect_family_scope": "GLOBAL_OR_PREDECLARED_PHASE_CONDITIONED_SIGNED_B_ONLY",
            "rule": "within this average-shift effect family, only a named global-B or predeclared phase-conditioned-B endpoint with independently confirmed REPRODUCIBLE_AVERAGE_SHIFT and a planning-mode-resolved interval report may enter its separately frozen contributor study",
            "other_effect_family_routes": {
                "reference_boundary_conditioned_signed_shift": "SEPARATE_BOUNDARY_CONFIRMATION_AND_ATTRIBUTION_GATE",
                "signed_semantic_event_shift": "SEPARATE_BOUNDARY_SEMANTIC_CONFIRMATION_AND_ATTRIBUTION_GATE",
                "nonnegative_semantic_disagreement": "SEPARATE_SEMANTIC_IMPACT_GATE_NOT_B",
            },
            "global_B_is_not_a_universal_operator_gate": True,
            "claim_scope": "contribution to the exact confirmed global-B or phase-conditioned-B estimand handled by this evaluator; not boundary-conditioned or semantic effects, correctness, harm, practical materiality, or an unrestricted root-cause claim",
        },
        "correctness": {
            "status": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
            "eager_is_truth": False,
        },
        "state_evidence": state_evidence,
        "nonclaims": [
            "implementation-relative shift is not correctness error",
            "confirmation does not establish long-run training impact",
            "operator readiness is not operator causality or root-cause proof",
            "NO_AVERAGE_SHIFT_BEYOND_FLOOR_DETECTED_AT_TARGET_PRECISION is not equivalence or proof of zero effect",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "construction": result["construction"], "operator_attribution_gate": result["operator_attribution_gate"]}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
