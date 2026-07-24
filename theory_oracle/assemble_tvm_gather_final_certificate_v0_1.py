"""Assemble the post-reveal localization certificate without upgrading claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.operator_evidence import EvidenceGates, allowed_claim_level, validate_evidence_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--repair", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    pre = json.loads(Path(args.pre).read_text())
    post = json.loads(Path(args.post).read_text())
    repair = json.loads(Path(args.repair).read_text())
    gates = EvidenceGates(
        complete_witness=True,
        same_input_local_replay=bool(repair["same_input_local_replay"]["boundary_inputs_identical"]),
        local_discrepancy_reproducible=bool(repair["same_input_local_replay"]["raw_vs_repaired_different"]),
        provenance_complete=False,
        candidate_realization_preserved=False,
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=bool(repair["non_target_context_invariant"]),
        lower_level_replay=False,
        first_bad_stage_isolated=False,
        null_controls_valid=True,
    )
    report = {
        "schema_version": "forkcert.historical-localization-certificate.v0.1",
        "case_identity": pre["case_identity"],
        "failure_witness": pre["oracle"]["buggy"],
        "reference_contract": pre["oracle"]["declared_endpoint"],
        "first_verified_difference_stage": "ONNX frontend -> Relax negative-index normalization boundary (post-reveal stage comparison)",
        "region": {
            "id": "onnx_gather_to_relax_take",
            "compiled_region": "Relax take -> TIR take",
            "provenance": pre["provenance"],
        },
        "region_inventory": [
            {"region_id": "onnx_gather_to_relax_take", "compiled_region": "Relax take -> TIR take", "provenance_complete": False}
        ],
        "mechanism_hypothesis": {
            "text": "negative indices are passed directly to Relax/TIR take instead of being normalized according to ONNX semantics",
            "evidence": post["external_validation"],
        },
        "production_evidence": repair["same_input_local_replay"],
        "local_replay": repair["same_input_local_replay"],
        "intervention_evidence": {
            "type": repair["intervention"],
            "oracle_before": pre["oracle"]["buggy"],
            "oracle_after": repair,
            "non_target_context_invariant": repair["non_target_context_invariant"],
        },
        "intervention": {
            "type": repair["intervention"],
            "before": pre["oracle"]["buggy"],
            "after": repair,
            "non_target_context_invariant": repair["non_target_context_invariant"],
        },
        "provenance": pre["provenance"],
        "oracle": {
            "declared_endpoint": pre["oracle"]["declared_endpoint"],
            "before": pre["oracle"]["buggy"],
            "after": repair,
        },
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "external_validation": post["external_validation"],
        "limitations": [
            "pre-reveal locator claim remains OBSERVATION",
            "fixed-suffix mediation is uninstantiated in this one-region witness",
            "repair rebuilds Relax/TIR and changes non-target context",
            "kernel identity/source-line provenance is incomplete",
            "therefore this is not a unique root-cause or correctness proof beyond the declared ONNX contract",
        ],
    }
    errors = validate_evidence_report(report)
    if errors:
        raise SystemExit("invalid certificate: " + "; ".join(errors))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "allowed_claim_level": report["allowed_claim_level"]}, sort_keys=True))


if __name__ == "__main__":
    main()
