#!/usr/bin/env python
"""Assemble a fail-closed localization certificate from audited evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-audit", required=True)
    parser.add_argument("--local-result", required=True)
    parser.add_argument("--intervention-audit", required=True)
    parser.add_argument("--intervention-result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    local_audit = load(args.local_audit)
    local_result = load(args.local_result)
    intervention_audit = load(args.intervention_audit)
    intervention_result = load(args.intervention_result)
    errors: list[str] = []
    if not local_audit.get("valid"):
        errors.append("local replay audit is invalid")
    if not intervention_audit.get("valid"):
        errors.append("intervention audit is invalid")
    if local_result.get("case_id") != intervention_result.get("case_id"):
        errors.append("local and intervention case ids differ")
    if not local_result["local_replay"]["production_observed"]:
        errors.append("same-input production evidence is absent")
    if local_result["local_replay"]["mediation_observed"]:
        errors.append("unexpected mediation result for this producer case")
    if intervention_result["claim"]["allowed_claim_level"] != "INTERVENTION_DEPENDENT_ATTRIBUTION":
        errors.append("intervention claim level is not the declared maximum")
    if not intervention_result["context"]["summary_equal_except_target_text"]:
        errors.append("non-target context gate failed")

    witness = local_result["complete_witness"]
    local = local_result["local_replay"]
    intervention = intervention_result["intervention"]
    outputs = intervention_result["outputs"]
    certificate = {
        "schema_version": "forkcert.localization_certificate.v0.1",
        "case_id": local_result["case_id"],
        "status": "VALID" if not errors else "INVALID",
        "failure_witness": {
            "reference_output": witness["eager"],
            "compiled_output": witness["compiled"],
            "max_abs_error": witness["eager_vs_compiled"]["max_abs"],
            "compiled_repeat_exact": witness["compiled_repeat_exact"],
            "reference_artifact_match": witness.get("eager_matches_reference_artifact", False),
        },
        "validated_discrepancy_stage": {
            "boundary": "pool output -> flatten+sum suffix",
            "source_nodes": ["aten.view", "aten.sum"],
            "same_input_production": local["production_observed"],
            "production_delta": local["production"],
            "boundary_mediation": local["mediation_observed"],
            "pool_only_control_exact": local["compiled_pool_exact"],
            "interpretation": "reduction suffix is a validated local producer candidate; mediation is negative for the pool boundary",
        },
        "region": {
            "generated_kernel": intervention["target_kernel"],
            "intervention_type": intervention["type"],
            "provenance_completeness": local_result["gates"]["provenance_complete"],
        },
        "mechanism_hypothesis": {
            "old_expression": intervention["old_expression"],
            "tested_expression": intervention["new_expression"],
            "statement": "the target reduction kernel's batch-stride expression is an endpoint-relevant numerical mechanism hypothesis",
            "not_claimed_as": ["historical patch", "unique root cause", "compiler-stage proof"],
        },
        "intervention_evidence": {
            "original_max_abs_error": outputs["reference_vs_original"]["max_abs"],
            "intervention_residual_max_abs": outputs["reference_vs_intervention"]["max_abs"],
            "within_control_residual_tolerance": outputs["intervention_within_control_residual_tolerance"],
            "target_code_changed_only": intervention_result["context"]["non_target_text_signature_equal"],
            "non_target_context_invariant": intervention_result["context"]["summary_equal_except_target_text"],
            "independent_repeat_audit": intervention_audit["valid"],
        },
        "allowed_max_claim_level": (
            "INTERVENTION_DEPENDENT_ATTRIBUTION" if not errors else "INVALID"
        ),
        "limitations": [
            "the blind package has not been scored against the hidden historical patch",
            "the intervention is a generated-code hypothesis rather than a developer-confirmed repair",
            "runtime/autotuning and compiler-pass provenance are not fully observed",
            "producer, propagator, and mediator are reported separately; no unique root cause is assigned",
        ],
        "inputs": {
            "local_audit": str(Path(args.local_audit).resolve()),
            "intervention_audit": str(Path(args.intervention_audit).resolve()),
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
