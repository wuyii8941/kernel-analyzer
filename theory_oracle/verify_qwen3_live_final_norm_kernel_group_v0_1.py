#!/usr/bin/env python
"""Independently verify the live original final-RMSNorm kernel-group evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    report: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    observability_gate: dict[str, Any],
) -> list[str]:
    errors = []
    if report.get("valid") is not True or report.get("status") != "VALID":
        errors.append("runner report is not valid")
    failed_gates = [name for name, passed in report.get("gates", {}).items() if not passed]
    if failed_gates:
        errors.append(f"failed runner gates: {failed_gates}")
    if observability_gate.get("forward_kernel_inventory_eligible") is not True:
        errors.append("forward observability/provenance gate is not eligible")
    contract = json.loads(Path(manifest["realization_contract"]).read_text())
    anchors = report.get("anchors", {})
    if anchors.get("eager") != [contract["reference_scorer_sha256"]] * 2:
        errors.append("eager anchor mismatch")
    if anchors.get("candidate") != [contract["candidate_scorer_sha256"]] * 2:
        errors.append("candidate anchor mismatch")
    if anchors.get("noop") != anchors.get("candidate"):
        errors.append("no-op proxy changed candidate")
    if anchors.get("restored") != anchors.get("candidate"):
        errors.append("kernel restoration mismatch")
    if anchors.get("repair", [None, None])[0] != anchors.get("repair", [None, None])[1]:
        errors.append("repair repeats differ")

    provenance = report.get("kernel_group", {}).get("provenance", {})
    expected_rows = [
        row
        for row in inventory.get("kernels", [])
        if row.get("generated_symbol") == manifest["pointwise_kernel"]
    ]
    if len(expected_rows) != 1 or provenance.get("kernel_id") != expected_rows[0].get(
        "kernel_id"
    ):
        errors.append("kernel provenance mismatch")
    if not provenance.get("fx_node_metadata"):
        errors.append("kernel lacks FX/source metadata")

    production = report.get("same_input_production", {})
    records = production.get("records", [])
    if not (
        production.get("observed") is True
        and production.get("repeat_exact") is True
        and len(records) == 2
        and all(len(row) == 1 for row in records)
        and records[0][0] == records[1][0]
    ):
        errors.append("same-input production evidence is incomplete or non-repeatable")
    elif records[0][0]["compiled_to_reference_output"].get("nonzero", 0) <= 0:
        errors.append("reported production output has no discrepancy")

    mediation = report.get("fixed_original_suffix_mediation", {})
    continuous = mediation.get("candidate_to_repair", {})
    if mediation.get("observed_continuous") is not True or continuous.get("nonzero", 0) <= 0:
        errors.append("continuous endpoint mediation was not observed")
    if anchors.get("repair", [None])[0] == anchors.get("candidate", [None])[0]:
        errors.append("repair hash does not support continuous mediation")
    event_count = int(mediation.get("off_to_on", 0)) + int(mediation.get("on_to_off", 0))
    if (mediation.get("semantic_disagreement", 0.0) > 0) != (event_count > 0):
        errors.append("semantic mediation counts and rate disagree")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--observability-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "report": args.report,
            "manifest": args.manifest,
            "inventory": args.inventory,
            "observability_gate": args.observability_gate,
        }.items()
    }
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    errors = validate(
        values["report"],
        values["manifest"],
        values["inventory"],
        values["observability_gate"],
    )
    mediation = values["report"].get("fixed_original_suffix_mediation", {})
    payload = {
        "schema_version": "forkcert.live-original-kernel-group-audit.v0.1",
        "valid": not errors,
        "verdict": "VALID" if not errors else "INVALID",
        "errors": errors,
        "evidence_level": (
            "ORIGINAL_GENERATED_KERNEL_GROUP_PRODUCTION_AND_CONTINUOUS_MEDIATION"
            if not errors
            else "INVALID"
        ),
        "semantic_mediation_observed": mediation.get("semantic_disagreement", 0.0) > 0,
        "backward_update_claim_allowed": False,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "claim_limits": [
            "generated fused kernel group, not a unique constituent source op",
            "continuous forward mediation does not imply clipping mediation",
            "no backward/update claim",
            "one selected state and no correctness claim",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["valid"] else 2)


if __name__ == "__main__":
    main()
