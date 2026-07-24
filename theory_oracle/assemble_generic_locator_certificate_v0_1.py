"""Assemble a pre-reveal certificate from bug-agnostic locator reports.

This certificate deliberately stops at observation.  It records the generic
endpoint Oracle, repeatability, and IR inventory, but never upgrades an IR
symbol to a source or root-cause claim without local replay and intervention
evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.operator_evidence import EvidenceGates, allowed_claim_level, validate_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_a = json.loads(Path(args.run_a).read_text())
    run_b = json.loads(Path(args.run_b).read_text())
    if run_a["case_identity"]["case_id"] != run_b["case_identity"]["case_id"]:
        raise SystemExit("repeat runs have different case IDs")
    for run in (run_a, run_b):
        contract = run["automation_contract"]
        if any(contract[key] for key in ("bug_specific_region_input", "bug_specific_repair_input", "bug_specific_semantic_rule")):
            raise SystemExit("locator report violates bug-agnostic contract")

    endpoint_a = run_a["oracle"]["endpoint"]
    endpoint_b = run_b["oracle"]["endpoint"]
    repeatable_endpoint = (
        endpoint_a["exact_match"] == endpoint_b["exact_match"]
        and endpoint_a["shape_match"] == endpoint_b["shape_match"]
        and endpoint_a["finite_match"] == endpoint_b["finite_match"]
        and endpoint_a["disagreement_count"] == endpoint_b["disagreement_count"]
        and abs(endpoint_a["max_abs_delta"] - endpoint_b["max_abs_delta"]) <= 1e-12
    )
    # A witness and its repeatability are different estimands.  Keep a valid
    # mismatch as an observation even when repeated execution changes its
    # magnitude; the latter is precisely the runtime-variability result that
    # the Oracle must expose.
    complete_witness = bool(
        endpoint_a["shape_match"]
        and not endpoint_a["exact_match"]
        and endpoint_a.get("finite_match", False)
    )
    runtime_a = run_a["oracle"]["runtime_repeatability"]
    runtime_b = run_b["oracle"]["runtime_repeatability"]
    runtime_variability_observed = bool(
        not runtime_a.get("exact_all_repeats", False)
        or not runtime_b.get("exact_all_repeats", False)
    )
    null_controls_valid = bool(
        runtime_a.get("instantiated")
        and runtime_b.get("instantiated")
        and runtime_a.get("finite_all_repeats", False)
        and runtime_b.get("finite_all_repeats", False)
    )
    gates = EvidenceGates(
        complete_witness=complete_witness,
        same_input_local_replay=False,
        local_discrepancy_reproducible=False,
        provenance_complete=False,
        candidate_realization_preserved=False,
        intervention_executed=False,
        oracle_recomputed=False,
        non_target_context_invariant=False,
        null_controls_valid=null_controls_valid,
    )
    report = {
        "schema_version": "forkcert.generic-locator-certificate.v0.1",
        "case_identity": run_a["case_identity"],
        "locator_runs": {"a": args.run_a, "b": args.run_b},
        "region_inventory": run_a["region_inventory"],
        "local_replay": {"status": "UNINSTANTIATED", "reason": "generic locator does not execute arbitrary region replay"},
        "provenance": {"status": "PARTIAL", "evidence": "generic IR symbols and call_tir reachability only"},
        "intervention": {"status": "UNINSTANTIATED", "reason": "no repair or injection supplied to blind locator"},
        "oracle": {
            "run_a": run_a["oracle"],
            "run_b": run_b["oracle"],
            "cross_run_endpoint_reproducible": repeatable_endpoint,
            "runtime_variability_observed": runtime_variability_observed,
        },
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "limitations": [
            "the endpoint reference is a declared contract, not an inferred mathematical truth",
            "IR symbol inventory is not source-line or unique-kernel provenance",
            "no local production or endpoint mediation claim is licensed",
            "this is pre-reveal evidence; fixed-version comparison belongs to a separate evaluator",
        ],
    }
    errors = validate_evidence_report(report)
    if errors:
        raise SystemExit("invalid certificate: " + "; ".join(errors))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "claim": report["allowed_claim_level"], "complete_witness": complete_witness}, sort_keys=True))


if __name__ == "__main__":
    main()
