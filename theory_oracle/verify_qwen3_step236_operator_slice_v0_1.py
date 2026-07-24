#!/usr/bin/env python
"""Independently verify a step236 operator evidence report and linked files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.operator_evidence import canonical_json_sha256, sha256_file, validate_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text())
    errors = validate_evidence_report(report)
    if report.get("evaluation_errors"):
        errors.extend(
            f"transition evaluation: {message}"
            for message in report["evaluation_errors"]
        )
    identity = report.get("case_identity", {})
    for path_key, hash_key in (
        ("manifest", "manifest_sha256"),
    ):
        path = Path(str(identity.get(path_key, "")))
        if not path.is_file() or sha256_file(path) != identity.get(hash_key):
            errors.append(f"case identity mismatch: {path_key}")
    expected_content = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    if report.get("content_sha256") != expected_content:
        errors.append("evidence report content digest mismatch")
    intervention = report.get("intervention", {})
    if intervention.get("candidate_anchor_exact") is not True:
        errors.append("partitioned candidate did not reproduce original candidate")
    if intervention.get("reference_anchor_exact") is not True:
        errors.append("partitioned reference did not reproduce reference")
    if intervention.get("non_target_context", {}).get("exact") is not True:
        errors.append("non-target context is not invariant")
    transition_status = intervention.get("one_step_transition_status")
    if transition_status and not str(transition_status).startswith("PENDING_"):
        if transition_status != "VALID":
            errors.append("one-step transition intervention is not valid")
        transition_context = intervention.get("transition_non_target_context", [])
        if not transition_context or not all(row.get("exact") for row in transition_context):
            errors.append("transition non-target context is not invariant")
    payload = {
        "schema_version": "forkcert.operator-evidence-audit.v0.1",
        "valid": not errors,
        "verdict": "VALID" if not errors else "INVALID",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "allowed_claim_level": report.get("allowed_claim_level"),
        "errors": errors,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["valid"] else 2)


if __name__ == "__main__":
    main()
