#!/usr/bin/env python3
"""Validate same-bank mechanism-pilot records against one frozen boundary bridge.

This is deliberately a record-integrity gate, not an operator-effect verdict.
It prevents an intervention from changing the state/token population on which
the already-confirmed boundary-conditional estimand was defined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.bias_oracle_population_v0_2 import EffectRecord
from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-contributor-record-bank.v0.1"
ALLOWED_INTERVENTIONS = {
    "REPAIR_REMOVAL": lambda row: float(row["candidate_value"])
    - float(row["intervention_value"]),
    "INJECTION_CREATION": lambda row: float(row["intervention_value"])
    - float(row["reference_value"]),
}
ARM_NAMES = {"reference", "candidate", "intervention"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_state_scope(bridge: dict[str, Any]) -> dict[str, tuple[str, str]]:
    scope: dict[str, tuple[str, str]] = {}
    groups = bridge.get("exposed_state_ids_by_trajectory_phase", {})
    if not isinstance(groups, dict):
        return scope
    for group, state_ids in groups.items():
        try:
            trajectory, phase = group.split("::", 1)
        except ValueError:
            return {}
        if not isinstance(state_ids, list):
            return {}
        for state_id in state_ids:
            if not isinstance(state_id, str) or state_id in scope:
                return {}
            scope[state_id] = (trajectory, phase)
    return scope


def validate_effect_rows(
    bank: dict[str, Any],
    scope_artifact: dict[str, Any],
    *,
    state_bank_role: str,
    population_claim_allowed: bool,
    condition_anchor: str,
) -> tuple[list[EffectRecord], dict[str, Any], list[str]]:
    """Validate the common frozen-mask arm grid for either state-bank role."""
    errors: list[str] = []
    if not isinstance(bank.get("candidate_id"), str) or not bank["candidate_id"].strip():
        errors.append("candidate_id is uninstantiated")
    intervention_kind = bank.get("intervention_kind")
    if intervention_kind not in ALLOWED_INTERVENTIONS:
        errors.append("intervention_kind must be REPAIR_REMOVAL or INJECTION_CREATION")
    if bank.get("condition_membership_recomputed_after_intervention") is not False:
        errors.append("condition membership must not be recomputed after intervention")
    target_effect_role = scope_artifact.get(
        "target_effect_role", "SIGNED_BOUNDARY_CONDITIONAL_EFFECT"
    )
    semantic_impact = target_effect_role == "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
    if bank.get("target_effect_role", target_effect_role) != target_effect_role:
        errors.append("record-bank target effect role differs from frozen scope")

    scope = expected_state_scope(scope_artifact)
    masks = scope_artifact.get("reference_anchor_condition_mask_sha256_by_state", {})
    if not scope or set(scope) != set(masks):
        errors.append("frozen exposed-state scope and mask bank are inconsistent")
    rows = bank.get("rows")
    if not isinstance(rows, list):
        errors.append("record rows must be a list")
        rows = []
    observed_keys: list[tuple[str, int]] = []
    records: list[EffectRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row[{index}] is not an object")
            continue
        state_id = row.get("state_id")
        repeat_id = row.get("repeat_id")
        if state_id not in scope:
            errors.append(f"row[{index}] state is outside frozen exposed bank")
            continue
        if repeat_id not in {1, 2}:
            errors.append(f"row[{index}] repeat_id must be 1 or 2")
            continue
        observed_keys.append((state_id, int(repeat_id)))
        trajectory, phase = scope[state_id]
        if row.get("trajectory_id") != trajectory or row.get("phase") != phase:
            errors.append(f"row[{index}] trajectory/phase identity mismatch")
        arm_masks = row.get("arm_condition_mask_sha256")
        expected_mask = masks[state_id]
        if (
            not isinstance(arm_masks, dict)
            or set(arm_masks) != ARM_NAMES
            or any(value != expected_mask for value in arm_masks.values())
        ):
            errors.append(f"row[{index}] arm masks differ from frozen reference mask")
        cardinalities = row.get("arm_condition_cardinality")
        if (
            not isinstance(cardinalities, dict)
            or set(cardinalities) != ARM_NAMES
            or any(
                not isinstance(value, int) or value <= 0
                for value in cardinalities.values()
            )
            or len(set(cardinalities.values())) != 1
        ):
            errors.append(f"row[{index}] arm condition cardinalities differ or are invalid")
        scalar_fields = (
            "reference_value",
            "candidate_value",
            "intervention_value",
            "attribution_effect",
        )
        if any(
            not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            for field in scalar_fields
        ):
            errors.append(f"row[{index}] endpoint values/effect must be finite scalars")
            continue
        if semantic_impact:
            if (
                row.get("value_semantics")
                != "REFERENCE_ZERO_CANDIDATE_AND_INTERVENTION_ARE_REFERENCE_PAIRED_DISAGREEMENT_RATES"
            ):
                errors.append(
                    f"row[{index}] semantic disagreement lacks relational value semantics"
                )
                continue
            reference_value = float(row["reference_value"])
            candidate_value = float(row["candidate_value"])
            intervention_value = float(row["intervention_value"])
            if (
                reference_value != 0.0
                or not 0.0 <= candidate_value <= 1.0
                or not 0.0 <= intervention_value <= 1.0
            ):
                errors.append(
                    f"row[{index}] relational disagreement rates are outside their contract"
                )
                continue
        if intervention_kind in ALLOWED_INTERVENTIONS:
            expected_effect = ALLOWED_INTERVENTIONS[intervention_kind](row)
            observed_effect = float(row["attribution_effect"])
            if not math.isclose(
                observed_effect, expected_effect, rel_tol=1e-12, abs_tol=1e-15
            ):
                errors.append(f"row[{index}] attribution effect formula mismatch")
            else:
                records.append(
                    EffectRecord(
                        trajectory_id=trajectory,
                        phase=phase,
                        state_id=state_id,
                        repeat_id=int(repeat_id),
                        effect=observed_effect,
                    )
                )
    expected_keys = {
        (state_id, repeat_id) for state_id in scope for repeat_id in (1, 2)
    }
    if len(observed_keys) != len(set(observed_keys)):
        errors.append("duplicate state/repeat rows")
    if set(observed_keys) != expected_keys:
        errors.append("record bank must contain exactly two repeats for every exposed state")
    if errors:
        records = []
    return records, {
        "target_endpoint": scope_artifact.get("target_endpoint"),
        "candidate_id": bank.get("candidate_id"),
        "intervention_kind": intervention_kind,
        "expected_exposed_states": len(scope),
        "expected_rows": len(expected_keys),
        "condition_anchor": condition_anchor,
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "target_effect_role": target_effect_role,
        "value_semantics": (
            "RELATIONAL_DISAGREEMENT_RATES"
            if semantic_impact
            else "ARM_SPECIFIC_SIGNED_ENDPOINT_VALUES"
        ),
        "state_bank_role": state_bank_role,
        "population_operator_contribution_claim_allowed": population_claim_allowed,
    }, errors


def validate_and_collect(
    bank: dict[str, Any], bridge: dict[str, Any], bridge_path: Path
) -> tuple[list[EffectRecord], dict[str, Any], list[str]]:
    errors: list[str] = []
    if bank.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported boundary contributor record schema")
    if bank.get("status") != "COMPLETE_VALID_AFTER_INTERVENTION":
        errors.append("record bank is not marked complete after intervention")
    if (
        bridge.get("schema_version") != BRIDGE_SCHEMA_VERSION
        or bridge.get("valid") is not True
        or bridge.get("status") != "FROZEN_BEFORE_BOUNDARY_ATTRIBUTION_PILOT"
    ):
        errors.append("linked boundary attribution bridge is not valid/frozen")
    if bridge.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("boundary attribution bridge weighting contract drifted")
    if (
        bridge.get("state_bank_role")
        != "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT"
        or bridge.get("population_operator_contribution_claim_allowed") is not False
    ):
        errors.append("boundary bridge does not declare same-bank pilot-only scope")
    link = bank.get("boundary_attribution_bridge", {})
    try:
        linked_path = Path(str(link.get("path", ""))).resolve()
    except (OSError, RuntimeError):
        linked_path = Path()
    if linked_path != bridge_path.resolve() or link.get("sha256") != sha256_file(
        bridge_path
    ):
        errors.append("record bank is not bound to the exact boundary bridge")
    if bank.get("target_endpoint") != bridge.get("target_endpoint"):
        errors.append("record bank endpoint differs from boundary bridge")
    records, construction, row_errors = validate_effect_rows(
        bank,
        bridge,
        state_bank_role="ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT",
        population_claim_allowed=False,
        condition_anchor="FROZEN_REFERENCE_MASK_FROM_BOUNDARY_ATTRIBUTION_BRIDGE",
    )
    errors.extend(row_errors)
    construction["claim_scope"] = (
        "same-bank intervention response for the exact boundary-conditional "
        "estimand; no population contribution, global-B, root-cause, necessity, "
        "or sufficiency claim"
    )
    return ([] if errors else records), construction, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bridge_path = Path(args.bridge).resolve()
    bank_path = Path(args.records).resolve()
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    records, construction, errors = validate_and_collect(bank, bridge, bridge_path)
    result = {
        "schema_version": "forkcert.qwen3-boundary-contributor-record-validation.v0.1",
        "valid": not errors,
        "construction": construction,
        "effect_records": [record.__dict__ for record in records],
        "errors": errors,
        "nonclaims": [
            "record validity is not an operator-effect verdict",
            "repair and injection are separate intervention estimands",
            "the candidate is not thereby a root cause",
            "same-bank pilot records do not support population operator contribution",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
