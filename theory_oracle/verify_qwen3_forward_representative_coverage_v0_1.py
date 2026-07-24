#!/usr/bin/env python
"""Cross-experiment audit of the frozen-forward representative denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equivalence", required=True)
    parser.add_argument("--operator-audit", required=True)
    parser.add_argument("--named-audit", required=True)
    parser.add_argument("--functional-audit", required=True)
    parser.add_argument("--sdpa-audit", required=True)
    parser.add_argument("--mask-audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    equivalence = json.loads(Path(args.equivalence).read_text())
    operator = json.loads(Path(args.operator_audit).read_text())
    named = json.loads(Path(args.named_audit).read_text())
    functional = json.loads(Path(args.functional_audit).read_text())
    sdpa = json.loads(Path(args.sdpa_audit).read_text())
    mask = json.loads(Path(args.mask_audit).read_text())
    metrics = equivalence["metrics"]

    gates = {
        "original_operator_audit_valid": operator["verdict"] == "VALID" and all(subject["valid"] for subject in operator["subjects"]),
        "named_batch_audit_valid": named["status"] == "VALID_BARRIER_CONDITIONED_BATCH_EVIDENCE",
        "functional_batch_audit_valid": functional["status"] == "VALID_BARRIER_CONDITIONED_BATCH_EVIDENCE",
        "sdpa_invalidation_audit_valid": sdpa["status"] == "VALID_INVALIDATION_AUDIT",
        "mask_invalid_treatment_audit_valid": mask["status"] == "VALID_INVALID_TREATMENT_AUDIT",
        "representative_denominator_62": metrics["minimum_representative_invocations_if_transport_succeeds"] == 62,
        "all_representatives_attempted": metrics["representative_attempts"] == 62,
        "usable_partition_exact": metrics["existing_valid_invocations"] == 2 and metrics["barrier_conditioned_invocations"] == 50,
        "invalid_partition_exact": metrics["invalid_representative_treatments"] == 10,
        "partition_sums_to_denominator": (
            metrics["existing_valid_invocations"]
            + metrics["barrier_conditioned_invocations"]
            + metrics["invalid_representative_treatments"]
            == metrics["minimum_representative_invocations_if_transport_succeeds"]
        ),
        "no_generated_family_fully_covered": metrics["fully_covered_treatment_families"] == 0,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-forward-representative-coverage-audit.v0.1",
        "status": "VALID_COMPLETE_ATTEMPT_COVERAGE" if passed else "INVALID_AUDIT",
        "gates": gates,
        "denominators": {
            "representatives": 62,
            "attempted": metrics["representative_attempts"],
            "usable_original_candidate": metrics["existing_valid_invocations"],
            "usable_barrier_conditioned": metrics["barrier_conditioned_invocations"],
            "invalid_treatments": metrics["invalid_representative_treatments"],
            "composite_sdpa_overlapping_evidence": metrics["composite_sdpa_barrier_invocations"],
            "generated_treatment_families_fully_covered": metrics["fully_covered_treatment_families"],
        },
        "claim": "every declared frozen-forward representative was attempted",
        "claims_not_supported": [
            "all representatives have usable causal effects",
            "barrier-conditioned effects transport to the original candidate",
            "all 536 invocations are directly intervened",
            "generated kernel families are fully covered",
            "backward or full training-step operator coverage",
            "population transport across matched states",
            "compiler correctness",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
