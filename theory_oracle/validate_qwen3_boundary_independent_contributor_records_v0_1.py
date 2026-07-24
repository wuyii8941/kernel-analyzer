#!/usr/bin/env python3
"""Validate independent-bank boundary repair/injection records fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.freeze_qwen3_boundary_independent_contributor_bank_v0_1 import (
    SCHEMA_VERSION as INDEPENDENT_BANK_SCHEMA_VERSION,
    sha256_file,
)
from theory_oracle.validate_qwen3_boundary_contributor_records_v0_1 import (
    validate_effect_rows,
)


SCHEMA_VERSION = (
    "forkcert.qwen3-boundary-independent-contributor-record-bank.v0.1"
)
VALIDATION_SCHEMA_VERSION = (
    "forkcert.qwen3-boundary-independent-contributor-record-validation.v0.1"
)


def validate_and_collect_independent(
    records_bank: dict[str, Any],
    independent_bank: dict[str, Any],
    independent_bank_path: Path,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    if records_bank.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported independent contributor record schema")
    if records_bank.get("status") != "COMPLETE_VALID_AFTER_OPERATOR_INTERVENTIONS":
        errors.append("independent intervention record bank is not complete")
    if (
        independent_bank.get("schema_version") != INDEPENDENT_BANK_SCHEMA_VERSION
        or independent_bank.get("valid") is not True
        or independent_bank.get("status")
        != "FROZEN_BEFORE_BOUNDARY_OPERATOR_INTERVENTIONS"
        or independent_bank.get("state_bank_role")
        != "INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK"
        or independent_bank.get("eligible_for_independent_contributor_measurement")
        is not True
        or independent_bank.get("operator_intervention_outcomes_used_for_bank_freeze")
        is not False
    ):
        errors.append("linked independent contributor bank is not valid/frozen")
    if independent_bank.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("independent contributor weighting contract drifted")
    link = records_bank.get("independent_contributor_bank", {})
    linked_path = Path(str(link.get("path", ""))).resolve()
    if (
        linked_path != independent_bank_path.resolve()
        or link.get("sha256") != sha256_file(independent_bank_path)
    ):
        errors.append("records are not bound to the exact independent bank")
    if records_bank.get("target_endpoint") != independent_bank.get("target_endpoint"):
        errors.append("independent record endpoint differs from frozen bank")
    if records_bank.get("target_effect_role") != independent_bank.get(
        "target_effect_role"
    ):
        errors.append("independent records target effect role drifted")

    intervention_plan = records_bank.get("operator_intervention_plan", {})
    if (
        not isinstance(intervention_plan, dict)
        or intervention_plan.get("status") != "FROZEN_BEFORE_OPERATOR_OUTCOMES"
        or intervention_plan.get("candidate_id") != records_bank.get("candidate_id")
        or intervention_plan.get("intervention_kind")
        != records_bank.get("intervention_kind")
        or not isinstance(intervention_plan.get("family_id"), str)
        or not intervention_plan.get("family_id")
        or not isinstance(intervention_plan.get("family_member_id"), str)
        or not intervention_plan.get("family_member_id")
        or intervention_plan.get("candidate_selection_uses_independent_bank_outcomes")
        is not False
    ):
        errors.append("operator intervention identity/family was not prospectively frozen")

    records, construction, row_errors = validate_effect_rows(
        records_bank,
        independent_bank,
        state_bank_role="INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK",
        population_claim_allowed=False,
        condition_anchor="FROZEN_REFERENCE_MASK_FROM_INDEPENDENT_CONTRIBUTOR_BANK",
    )
    errors.extend(row_errors)
    construction.update(
        {
            "intervention_family_id": intervention_plan.get("family_id"),
            "intervention_family_member_id": intervention_plan.get(
                "family_member_id"
            ),
            "eligible_for_trajectory_level_evaluation": not errors,
            "claim_scope": (
                "validated independent-bank intervention records for the exact "
                "boundary-conditional estimand; no operator-effect verdict yet"
            ),
        }
    )
    return ([] if errors else records), construction, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-bank", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    independent_path = Path(args.independent_bank).resolve()
    records_path = Path(args.records).resolve()
    independent_bank = json.loads(independent_path.read_text(encoding="utf-8"))
    records_bank = json.loads(records_path.read_text(encoding="utf-8"))
    records, construction, errors = validate_and_collect_independent(
        records_bank, independent_bank, independent_path
    )
    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "valid": not errors,
        "construction": construction,
        "effect_records": [record.__dict__ for record in records],
        "errors": errors,
        "population_operator_contribution_claim_allowed": False,
        "next_gate": (
            "apply the prospectively frozen multiplicity-adjusted trajectory-level "
            "contribution estimator"
        ),
        "nonclaims": [
            "record validity is not an operator-effect verdict",
            "repair and injection are separate intervention families",
            "independence of the state bank does not prove root cause",
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
