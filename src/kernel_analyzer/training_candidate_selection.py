"""Fail-closed selection for human review before expensive training.

This module never chooses a modification or launches training.  It checks a
predeclared evidence manifest and separates new operator families from follow-up
work on already studied families.  Numerical magnitude is one required policy
gate, never the sole ranking rule.
"""

from __future__ import annotations

import math


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def assess_case(case, policy):
    reasons = []
    required = {
        "measurement", "warm_state", "mechanism", "operator_family_id",
        "implementation_kind", "family_novelty",
    }
    missing = sorted(required - set(case))
    if missing:
        return {
            "case_id": case.get("case_id", "MISSING_CASE_ID"),
            "status": "NOT_ELIGIBLE",
            "reasons": ["MISSING_REQUIRED_FIELDS:" + ",".join(missing)],
            "queue": "NONE",
        }
    measurement = case["measurement"]
    warm = case["warm_state"]
    mechanism = case["mechanism"]
    if measurement.get("status") != "VALID":
        reasons.append("MEASUREMENT_NOT_VALID")
    if measurement.get("contrast_id") != "NATURAL_IMPLEMENTATION":
        reasons.append("NOT_A_NATURAL_IMPLEMENTATION_COMPARISON")
    if measurement.get("primary_endpoint") != "PARAMETER_WRITE":
        reasons.append("ACTUAL_PARAMETER_WRITE_NOT_PRIMARY")
    reference = measurement.get("reference_scope", {})
    if not reference.get("same_local_operands"):
        reasons.append("REFERENCE_DOES_NOT_USE_SAME_LOCAL_OPERANDS")
    if reference.get("includes_possible_upstream_differences") is not False:
        reasons.append("REFERENCE_SCOPE_MAY_INCLUDE_UPSTREAM_DIFFERENCES")
    if reference.get("source_or_call_identity_verified") is not True:
        reasons.append("IMPLEMENTATION_IDENTITY_NOT_VERIFIED")
    rms = warm.get("fixed_suite_total_rms")
    if warm.get("status") != "VALID" or warm.get("natural_optimizer_state") is not True:
        reasons.append("NATURAL_WARM_STATE_EVIDENCE_MISSING")
    if warm.get("actual_parameter_write") is not True:
        reasons.append("WARM_EVIDENCE_IS_NOT_ACTUAL_PARAMETER_WRITE")
    if not _finite_nonnegative(rms):
        reasons.append("WARM_EFFECT_SIZE_INVALID")
    elif rms < policy["minimum_warm_update_rms"]:
        reasons.append("WARM_EFFECT_BELOW_PREDECLARED_TRAINING_RELEVANCE_FLOOR")
    if mechanism.get("prediction_status") not in {
        "FROZEN_BEFORE_CONFIRMATION", "CONFIRMED_ON_SEPARATE_STATES",
    }:
        reasons.append("MECHANISM_PREDICTION_NOT_FROZEN")
    if mechanism.get("implementation_location_verified") is not True:
        reasons.append("MECHANISM_NOT_LOCATED_IN_REAL_IMPLEMENTATION")
    variants = mechanism.get("legal_modification_ids")
    if not isinstance(variants, list) or not variants or not all(
            isinstance(value, str) and value for value in variants):
        reasons.append("NO_REVIEWED_LEGAL_MODIFICATION")
    kind = case["implementation_kind"]
    if kind not in {"TRITON", "ORDINARY", "MIXED"}:
        reasons.append("UNKNOWN_IMPLEMENTATION_KIND")
    novelty = case["family_novelty"]
    if novelty not in {"NEW_OPERATOR_FAMILY", "ALREADY_DEEPLY_STUDIED"}:
        reasons.append("UNKNOWN_FAMILY_NOVELTY")
    if reasons:
        queue = "NONE"
        status = "NOT_ELIGIBLE"
    else:
        status = "ELIGIBLE_FOR_HUMAN_TRAINING_REVIEW"
        if novelty == "NEW_OPERATOR_FAMILY" and kind == "TRITON":
            queue = "NEW_TRITON_OPERATOR_FAMILY"
        elif novelty == "NEW_OPERATOR_FAMILY":
            queue = "NEW_NONTRITON_OPERATOR_FAMILY"
        else:
            queue = "EXISTING_FAMILY_MECHANISM_FOLLOWUP"
    return {
        "case_id": case.get("case_id", "MISSING_CASE_ID"),
        "operator_family_id": case.get("operator_family_id"),
        "implementation_kind": kind,
        "family_novelty": novelty,
        "status": status,
        "reasons": reasons,
        "queue": queue,
        "warm_update_rms": rms,
    }


def select_candidates(manifest):
    if manifest.get("schema") != "training-validation-candidate-manifest-v1":
        raise ValueError("Unknown training candidate manifest")
    policy = manifest.get("policy", {})
    if policy.get("fixed_before_candidate_measurement") is not True:
        raise ValueError("Training relevance policy must be fixed before candidate measurement")
    floor = policy.get("minimum_warm_update_rms")
    if not _finite_nonnegative(floor) or floor == 0:
        raise ValueError("A positive finite warm-update relevance floor is required")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Nonempty candidate list required")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty case IDs required")
    rows = [assess_case(case, policy) for case in cases]
    queue_order = {
        "NEW_TRITON_OPERATOR_FAMILY": 0,
        "NEW_NONTRITON_OPERATOR_FAMILY": 1,
        "EXISTING_FAMILY_MECHANISM_FOLLOWUP": 2,
        "NONE": 3,
    }
    rows.sort(key=lambda row: (queue_order[row["queue"]], row.get("operator_family_id") or "",
                               row["case_id"]))
    return {
        "schema": "training-validation-candidate-selection-v1",
        "policy": policy,
        "selection_uses_effect_size_as_only_criterion": False,
        "automatic_training_launch": False,
        "eligible_cases": sum(row["status"] == "ELIGIBLE_FOR_HUMAN_TRAINING_REVIEW"
                              for row in rows),
        "rows": rows,
        "scope": "Eligibility for human review; not a causal verdict or training launch decision",
    }
