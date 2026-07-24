#!/usr/bin/env python3
"""Freeze a same-confirmation-bank boundary mechanism-pilot bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.evaluate_qwen3_boundary_conditional_confirmation_v0_1 import (
    SCHEMA_VERSION as CONFIRMATION_SCHEMA,
    SEMANTIC_IMPACT_KINDS,
    parse_endpoint,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-attribution-bridge.v0.1"
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_bridge(
    confirmation: dict[str, Any], confirmation_path: Path, endpoint: str
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        kind, tau = parse_endpoint(endpoint)
    except ValueError as error:
        kind, tau = "INVALID", None
        errors.append(str(error))
    if (
        confirmation.get("schema_version") != CONFIRMATION_SCHEMA
        or confirmation.get("valid") is not True
        or confirmation.get("verdict") != "VALID_BOUNDARY_CONFIRMATION_CONSTRUCTION"
    ):
        errors.append("linked boundary confirmation is not valid")
    gate = confirmation.get("operator_attribution_gate", {})
    if gate.get("automatic_operator_launch") is not False:
        errors.append("boundary confirmation must not auto-launch attribution")
    if endpoint not in gate.get("ready_endpoints", []):
        errors.append("endpoint is absent from boundary confirmation ready gate")
    result = confirmation.get("endpoints", {}).get(endpoint, {})
    semantic_impact = kind in SEMANTIC_IMPACT_KINDS
    if semantic_impact:
        if (
            result.get("final_semantic_impact_verdict")
            != "REPRODUCIBLE_SEMANTIC_DISAGREEMENT"
        ):
            errors.append("endpoint lacks reproducible semantic disagreement")
        confirmed_effect = (
            result.get("semantic_impact_estimate", {})
            .get("population_mean_disagreement", {})
            .get("estimate")
        )
    else:
        if result.get("final_shift_verdict") != "REPRODUCIBLE_AVERAGE_SHIFT":
            errors.append("endpoint lacks a reproducible conditional shift")
        confirmed_effect = result.get("estimate", {}).get("B", {}).get("estimate")
    if not result.get("operator_attribution_eligibility", {}).get("eligible"):
        errors.append("endpoint attribution eligibility is false")
    if (
        not isinstance(confirmed_effect, (int, float))
        or not math.isfinite(float(confirmed_effect))
        or (float(confirmed_effect) <= 0.0 if semantic_impact else float(confirmed_effect) == 0.0)
    ):
        errors.append(
            "confirmed semantic endpoint lacks a finite positive impact"
            if semantic_impact
            else "confirmed conditional endpoint lacks a finite nonzero direction"
        )
        frozen_direction = None
    else:
        frozen_direction = 1 if semantic_impact or float(confirmed_effect) > 0 else -1
    support = result.get("support", {})
    if support.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("confirmed endpoint weighting contract drifted")
    state_groups = support.get("exposed_state_ids_by_trajectory_phase", {})
    mask_hashes = support.get(
        "reference_anchor_condition_mask_sha256_by_state", {}
    )
    state_ids = sorted(
        {state_id for values in state_groups.values() for state_id in values}
        if isinstance(state_groups, dict)
        else set()
    )
    if not state_ids or set(state_ids) != set(mask_hashes):
        errors.append("exposed state census and reference-anchor mask hashes differ")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in mask_hashes.values()
    ):
        errors.append("condition mask hashes must be SHA-256 identities")
    confirmation_state_ids = {
        row.get("state_id") for row in confirmation.get("state_evidence", [])
    }
    if not set(state_ids).issubset(confirmation_state_ids):
        errors.append("boundary exposed states are not a subset of confirmation evidence")

    valid = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": (
            "FROZEN_BEFORE_BOUNDARY_ATTRIBUTION_PILOT"
            if valid
            else "INVALID_BOUNDARY_ATTRIBUTION_BRIDGE"
        ),
        "endpoint_confirmation": {
            "path": str(confirmation_path),
            "sha256": sha256_file(confirmation_path),
        },
        "target_endpoint": endpoint,
        "endpoint_kind": kind,
        "tau": tau,
        "confirmed_conditional_effect_estimate": (
            None if semantic_impact else confirmed_effect
        ),
        "confirmed_semantic_impact_estimate": (
            confirmed_effect if semantic_impact else None
        ),
        "confirmed_target_effect_estimate": confirmed_effect,
        "frozen_effect_direction": frozen_direction,
        "target_effect_role": (
            "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
            if semantic_impact
            else "SIGNED_BOUNDARY_CONDITIONAL_EFFECT"
        ),
        "attribution_objective": (
            "REPAIR_REDUCES_OR_INJECTION_INCREASES_SEMANTIC_DISAGREEMENT"
            if semantic_impact
            else "INTERVENTION_CHANGES_EFFECT_IN_FROZEN_SIGNED_DIRECTION"
        ),
        "target_state_condition": {
            "kind": "REFERENCE_ANCHORED_ABSOLUTE_SIGNED_CLIP_MARGIN",
            "operator": "LE",
            "tau": tau,
            "anchor_repeat_id": 1,
        },
        "exposed_state_ids_by_trajectory_phase": state_groups,
        "reference_anchor_condition_mask_sha256_by_state": mask_hashes,
        "exposed_state_count": len(state_ids),
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "state_bank_role": "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT",
        "population_operator_contribution_claim_allowed": False,
        "intervention_contract": {
            "condition_mask_recomputed_after_intervention": False,
            "same_frozen_mask_required_for_reference_candidate_repair_and_injection": True,
            "states_outside_exposed_bank_must_not_enter_endpoint_estimate": True,
            "repair_or_injection_must_not_change_analysis_membership": True,
        },
        "claim_scope": (
            "same-endpoint-confirmation-bank mechanism response to the intervention "
            "for the exact signed or semantic-impact estimand; "
            "not a population operator-contribution estimate"
        ),
        "analysis_code": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "automatic_operator_launch": False,
        "next_gate": (
            "freeze a disjoint independent contributor trajectory bank and its "
            "reference masks before population operator-contribution inference"
        ),
        "errors": errors,
        "nonclaims": [
            "this bridge does not select an operator candidate",
            "this bridge does not prove root cause, necessity or sufficiency",
            "this bridge does not establish correctness or long-run harm",
            "semantic disagreement attribution is not contribution to Bias",
            "reusing the endpoint-confirmation state bank does not establish population operator contribution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    confirmation_path = Path(args.confirmation).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("boundary attribution bridge already exists; freeze is write-once")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    result = freeze_bridge(confirmation, confirmation_path, args.endpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "target_endpoint": result["target_endpoint"], "errors": result["errors"]}, indent=2))
    if result["valid"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
