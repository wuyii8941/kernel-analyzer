#!/usr/bin/env python3
"""Evaluate one frozen independent boundary repair/injection family member."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    REQUIRED_PHASES,
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    FIXED_RESOURCE_EXISTENCE,
)
from theory_oracle.bias_oracle_population_v0_2 import (
    EffectRecord,
    estimate_scalar_population,
)
from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (
    EXPECTED_SENSITIVITY,
    apply_trajectory_sensitivity,
    evaluate_realized_precision,
)
from theory_oracle.freeze_qwen3_boundary_independent_contributor_bank_v0_1 import (
    SCHEMA_VERSION as INDEPENDENT_BANK_SCHEMA_VERSION,
    sha256_file,
)
from theory_oracle.validate_qwen3_boundary_independent_contributor_records_v0_1 import (
    SCHEMA_VERSION as RECORD_SCHEMA_VERSION,
    validate_and_collect_independent,
)


PLAN_SCHEMA_VERSION = (
    "forkcert.qwen3-boundary-independent-contributor-inference-plan.v0.1"
)
SCHEMA_VERSION = "forkcert.qwen3-boundary-independent-contributor-evaluation.v0.1"


def validate_inference_plan(
    plan: dict[str, Any],
    independent_bank: dict[str, Any],
    independent_bank_path: Path,
    records_bank: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        errors.append("unsupported independent contributor inference plan")
    if plan.get("status") != "FROZEN_BEFORE_OPERATOR_OUTCOMES":
        errors.append("inference plan was not frozen before operator outcomes")
    link = plan.get("independent_contributor_bank", {})
    if (
        Path(str(link.get("path", ""))).resolve() != independent_bank_path.resolve()
        or link.get("sha256") != sha256_file(independent_bank_path)
    ):
        errors.append("inference plan is not bound to the independent bank")
    if plan.get("operator_outcomes_used_for_planning") is not False:
        errors.append("inference planning used operator outcomes")
    if plan.get("planning_mode") != FIXED_RESOURCE_EXISTENCE:
        errors.append("v0.1 independent contribution supports fixed-resource mode only")
    if plan.get("target_endpoint") != independent_bank.get("target_endpoint"):
        errors.append("inference-plan endpoint mismatch")
    target_effect_role = independent_bank.get("target_effect_role")
    if (
        target_effect_role
        not in {
            "SIGNED_BOUNDARY_CONDITIONAL_EFFECT",
            "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B",
        }
        or plan.get("target_effect_role") != target_effect_role
    ):
        errors.append("inference-plan target effect role drifted")
    if plan.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("inference-plan weighting contract drifted")
    direction = independent_bank.get("frozen_effect_direction")
    if direction not in {-1, 1} or plan.get("frozen_effect_direction") != direction:
        errors.append("inference-plan endpoint direction drifted")
    planned = plan.get("planned_trajectories")
    observed = len(independent_bank.get("trajectory_inputs", []))
    if not isinstance(planned, int) or planned < 8 or planned != observed:
        errors.append("planned trajectory count differs from independent bank")
    intervention_plan = records_bank.get("operator_intervention_plan", {})
    if (
        plan.get("family_id") != intervention_plan.get("family_id")
        or plan.get("family_member_id") != intervention_plan.get("family_member_id")
        or plan.get("candidate_id") != records_bank.get("candidate_id")
        or plan.get("intervention_kind") != records_bank.get("intervention_kind")
    ):
        errors.append("inference plan and intervention family identity differ")
    members = plan.get("family_members")
    family_alpha = plan.get("family_alpha")
    interval_alpha = plan.get("per_member_interval_alpha")
    if (
        not isinstance(members, list)
        or not members
        or len(members) != len(set(members))
        or plan.get("family_member_id") not in members
        or not isinstance(family_alpha, (int, float))
        or not 0 < float(family_alpha) < 1
        or not isinstance(interval_alpha, (int, float))
        or not math.isclose(
            float(interval_alpha), float(family_alpha) / len(members), rel_tol=1e-12
        )
    ):
        errors.append("contributor family multiplicity contract is invalid")
        interval_alpha = math.nan
    floors: dict[str, float] = {}
    for name in ("baseline_transport_floor", "directional_contribution_floor"):
        value = plan.get(name)
        source = plan.get(f"{name}_source", {})
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or not isinstance(source, dict)
            or source.get("kind") not in {"EXACT_ZERO_NULL", "NEGATIVE_CONTROL_ENVELOPE"}
            or source.get("uses_operator_outcomes") is not False
            or not isinstance(source.get("description"), str)
            or not source.get("description")
            or not isinstance(source.get("selection_rule"), str)
            or not source.get("selection_rule")
        ):
            errors.append(f"{name} or its prospective source is invalid")
        else:
            floors[name] = float(value)
    if plan.get("sensitivity") != EXPECTED_SENSITIVITY:
        errors.append("trajectory sensitivity protocol drifted")
    return {
        "planned_trajectories": planned,
        "interval_alpha": float(interval_alpha),
        "direction": direction,
        "target_effect_role": target_effect_role,
        **floors,
    }, errors


def aligned_profile_records(
    rows: list[dict[str, Any]], direction: int
) -> tuple[list[EffectRecord], list[EffectRecord]]:
    baseline: list[EffectRecord] = []
    contribution: list[EffectRecord] = []
    for row in rows:
        common = {
            "trajectory_id": str(row["trajectory_id"]),
            "phase": str(row["phase"]),
            "state_id": str(row["state_id"]),
            "repeat_id": int(row["repeat_id"]),
        }
        baseline.append(
            EffectRecord(
                **common,
                effect=direction
                * (float(row["candidate_value"]) - float(row["reference_value"])),
            )
        )
        contribution.append(
            EffectRecord(
                **common,
                effect=direction * float(row["attribution_effect"]),
            )
        )
    return baseline, contribution


def repair_diagnostic_records(
    rows: list[dict[str, Any]], direction: int
) -> tuple[list[EffectRecord], list[EffectRecord]]:
    residual: list[EffectRecord] = []
    absolute_reduction: list[EffectRecord] = []
    for row in rows:
        common = {
            "trajectory_id": str(row["trajectory_id"]),
            "phase": str(row["phase"]),
            "state_id": str(row["state_id"]),
            "repeat_id": int(row["repeat_id"]),
        }
        baseline_value = float(row["candidate_value"]) - float(
            row["reference_value"]
        )
        residual_value = float(row["intervention_value"]) - float(
            row["reference_value"]
        )
        residual.append(
            EffectRecord(**common, effect=direction * residual_value)
        )
        absolute_reduction.append(
            EffectRecord(
                **common,
                effect=abs(baseline_value) - abs(residual_value),
            )
        )
    return residual, absolute_reduction


def semantic_impact_view(
    estimate: dict[str, Any],
    *,
    population_key: str,
    phase_key: str,
    effect_role: str,
) -> dict[str, Any]:
    """Rename an internal scalar estimate so disagreement is never exposed as B."""
    population_mean = dict(estimate["B"])
    population_mean["semantic_impact_existence_floor"] = population_mean.pop(
        "measurement_floor"
    )
    return {
        "schema_version": "forkcert.semantic-impact-population-view.v0.1",
        "construction": estimate["construction"],
        population_key: population_mean,
        phase_key: estimate["conditional_B"][
            "predeclared_phase_rows"
        ],
        "state_and_phase_heterogeneity": estimate["H"],
        "same_state_runtime_variability": estimate["N"],
        "trajectory_sampling_uncertainty": estimate["U"],
        "trajectory_rows": estimate["trajectory_rows"],
        "phase_rows": estimate["phase_rows"],
        "state_rows": estimate["state_rows"],
        "semantic_impact_is_not_B": True,
        "effect_role": effect_role,
    }


def evaluate(
    plan: dict[str, Any],
    independent_bank: dict[str, Any],
    independent_bank_path: Path,
    records_bank: dict[str, Any],
) -> dict[str, Any]:
    if (
        independent_bank.get("schema_version") != INDEPENDENT_BANK_SCHEMA_VERSION
        or records_bank.get("schema_version") != RECORD_SCHEMA_VERSION
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "errors": ["unsupported bank or record schema"],
            "population_operator_contribution_claim_allowed": False,
        }
    _, record_construction, record_errors = validate_and_collect_independent(
        records_bank, independent_bank, independent_bank_path
    )
    frozen, plan_errors = validate_inference_plan(
        plan, independent_bank, independent_bank_path, records_bank
    )
    errors = [*record_errors, *plan_errors]
    if errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "errors": errors,
            "record_construction": record_construction,
            "population_operator_contribution_claim_allowed": False,
        }
    baseline_records, contribution_records = aligned_profile_records(
        records_bank["rows"], int(frozen["direction"])
    )
    semantic_impact = (
        frozen["target_effect_role"] == "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
    )
    common = {
        "required_phases": REQUIRED_PHASES,
        "min_confirmation_trajectories": int(frozen["planned_trajectories"]),
        "interval_alpha": float(frozen["interval_alpha"]),
    }
    baseline = estimate_scalar_population(
        baseline_records,
        measurement_floor=float(frozen["baseline_transport_floor"]),
        **common,
    )
    contribution = estimate_scalar_population(
        contribution_records,
        measurement_floor=float(frozen["directional_contribution_floor"]),
        **common,
    )
    repair_diagnostics: dict[str, Any] | None = None
    if records_bank.get("intervention_kind") == "REPAIR_REMOVAL":
        residual_records, absolute_reduction_records = repair_diagnostic_records(
            records_bank["rows"], int(frozen["direction"])
        )
        repair_diagnostics = {
            "inference_role": "DESCRIPTIVE_NOT_IN_FROZEN_PRIMARY_FAMILY",
            "repair_minus_reference_aligned_residual": estimate_scalar_population(
                residual_records,
                measurement_floor=0.0,
                **common,
            ),
            "absolute_discrepancy_reduction": estimate_scalar_population(
                absolute_reduction_records,
                measurement_floor=0.0,
                **common,
            ),
            "absolute_reduction_claim_allowed": False,
            "explained_fraction": {
                "status": "UNINSTANTIATED_RATIO_REQUIRES_SEPARATE_ESTIMAND_AND_INFERENCE",
                "reason": (
                    "directional contribution can overshoot or reverse the residual; "
                    "a ratio to baseline is unstable near zero and is not the primary estimand"
                ),
            },
        }
    baseline_plan = {
        "planning_mode": FIXED_RESOURCE_EXISTENCE,
        "shift_existence_floor": float(frozen["baseline_transport_floor"]),
    }
    contribution_plan = {
        "planning_mode": FIXED_RESOURCE_EXISTENCE,
        "shift_existence_floor": float(frozen["directional_contribution_floor"]),
    }
    baseline_sensitivity, baseline_verdict = apply_trajectory_sensitivity(
        baseline,
        baseline_plan,
        float(frozen["interval_alpha"]),
        EXPECTED_SENSITIVITY,
    )
    contribution_sensitivity, contribution_verdict = apply_trajectory_sensitivity(
        contribution,
        contribution_plan,
        float(frozen["interval_alpha"]),
        EXPECTED_SENSITIVITY,
    )
    baseline_precision = evaluate_realized_precision(baseline, baseline_plan)
    contribution_precision = evaluate_realized_precision(
        contribution, contribution_plan
    )
    baseline_transported = baseline_verdict == "REPRODUCIBLE_AVERAGE_SHIFT"
    contribution_confirmed = (
        contribution_verdict == "REPRODUCIBLE_AVERAGE_SHIFT"
    )
    precision_reported = all(
        profile.get("verdict")
        == "REPORTED_FIXED_RESOURCE_PRECISION_NOT_GATED"
        for profile in (baseline_precision, contribution_precision)
    )
    claim_allowed = (
        baseline_transported and contribution_confirmed and precision_reported
    )
    if semantic_impact:
        final = (
            "REPRODUCIBLE_SEMANTIC_IMPACT_INTERVENTION_CONTRIBUTION"
            if claim_allowed
            else "INDETERMINATE_SEMANTIC_DISAGREEMENT_DID_NOT_TRANSPORT"
            if not baseline_transported
            else "NO_REPRODUCIBLE_SEMANTIC_IMPACT_CONTRIBUTION"
        )
        profiles = {
            "baseline_reference_candidate_disagreement": semantic_impact_view(
                baseline,
                population_key="population_mean_reference_candidate_disagreement",
                phase_key="predeclared_phase_mean_reference_candidate_disagreement",
                effect_role="BASELINE_SEMANTIC_DISAGREEMENT_TRANSPORT",
            ),
            "intervention_disagreement_contribution": semantic_impact_view(
                contribution,
                population_key="population_mean_intervention_disagreement_contribution",
                phase_key="predeclared_phase_mean_intervention_disagreement_contribution",
                effect_role="REPAIR_REDUCTION_OR_INJECTION_CREATION",
            ),
            "repair_diagnostics": (
                {
                    "inference_role": repair_diagnostics["inference_role"],
                    "reference_intervention_residual_disagreement": semantic_impact_view(
                        repair_diagnostics["repair_minus_reference_aligned_residual"],
                        population_key="population_mean_reference_intervention_residual_disagreement",
                        phase_key="predeclared_phase_mean_reference_intervention_residual_disagreement",
                        effect_role="REPAIR_RESIDUAL_SEMANTIC_DISAGREEMENT",
                    ),
                    "absolute_disagreement_reduction": semantic_impact_view(
                        repair_diagnostics["absolute_discrepancy_reduction"],
                        population_key="population_mean_absolute_disagreement_reduction",
                        phase_key="predeclared_phase_mean_absolute_disagreement_reduction",
                        effect_role="DESCRIPTIVE_ABSOLUTE_DISAGREEMENT_REDUCTION",
                    ),
                    "absolute_reduction_claim_allowed": repair_diagnostics[
                        "absolute_reduction_claim_allowed"
                    ],
                    "explained_fraction": repair_diagnostics["explained_fraction"],
                    "interpretation": (
                        "for semantic repair, candidate_value is D(reference,candidate), "
                        "intervention_value is D(reference,intervention), and their "
                        "difference is disagreement reduction"
                    ),
                }
                if repair_diagnostics is not None
                else None
            ),
        }
    else:
        final = (
            "REPRODUCIBLE_DIRECTIONAL_INTERVENTION_CONTRIBUTION"
            if claim_allowed
            else "INDETERMINATE_BASELINE_DID_NOT_TRANSPORT"
            if not baseline_transported
            else "NO_REPRODUCIBLE_DIRECTIONAL_CONTRIBUTION"
        )
        profiles = {
            "baseline_candidate_minus_reference_aligned": baseline,
            "intervention_contribution_aligned": contribution,
            "repair_diagnostics": repair_diagnostics,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "errors": [],
        "construction": {
            "state_bank_role": "INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK",
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "planned_trajectories": frozen["planned_trajectories"],
            "top_level_df": int(frozen["planned_trajectories"]) - 1,
            "interval_alpha": frozen["interval_alpha"],
            "effects_aligned_to_confirmed_endpoint_direction": frozen["direction"],
            "target_effect_role": frozen["target_effect_role"],
        },
        "profiles": profiles,
        "sensitivity": {
            "baseline_transport": baseline_sensitivity,
            "contribution": contribution_sensitivity,
        },
        "realized_precision": {
            "baseline_transport": baseline_precision,
            "contribution": contribution_precision,
        },
        "baseline_transport_verdict": baseline_verdict,
        "contribution_verdict": contribution_verdict,
        "fixed_resource_precision_reported": precision_reported,
        "final_verdict": final,
        "population_operator_contribution_claim_allowed": claim_allowed,
        "claim_scope": (
            "intervention-dependent contribution to the exact confirmed "
            + (
                "boundary-conditioned semantic-disagreement estimand"
                if semantic_impact
                else "signed boundary-conditional estimand"
            )
            + " under the independent contributor-bank distribution"
        ),
        "nonclaims": [
            "confirmed contribution is not unrestricted root cause, necessity or sufficiency",
            "repair and injection effects are not assumed symmetric",
            "operator contribution is not a correctness or long-run harm verdict",
            "directional contribution is not an explained fraction or absolute discrepancy reduction",
            "semantic disagreement contribution is not contribution to Bias",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--independent-bank", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    independent_path = Path(args.independent_bank).resolve()
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    result = evaluate(plan, independent, independent_path, records)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "final_verdict": result.get("final_verdict"), "errors": result["errors"]}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
