#!/usr/bin/env python3
"""Attach valid signature measurements to the full observed-position inventory.

This is status-preserving bookkeeping. It verifies saved analysis artifacts
and exact source-release/task identities, but does not turn coverage into a
bias, mechanism, population, or training-outcome claim.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(
    inventory: dict[str, Any], summaries: list[dict[str, Any]], *, verify_artifacts: bool
) -> dict[str, Any]:
    result = copy.deepcopy(inventory)
    row_key = "positions" if "positions" in result else "records"
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result.get(row_key, []):
        identity = (str(Path(row["release"]).resolve()), str(row["task_id"]))
        if identity in by_identity:
            raise ValueError("Duplicate inventory identity: " + repr(identity))
        by_identity[identity] = row

    evidence_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    ignored_status_counts: dict[str, int] = {}
    for summary in summaries:
        if summary.get("schema") not in {
            "triton-signature-runtime-batch-summary-v1",
            "family-first-campaign-summary-v1",
        }:
            raise ValueError("Unexpected signature-summary schema")
        for row in summary.get("rows", []):
            status = str(row.get("measurement_status", "NOT_ASSESSED"))
            if status != "VALID":
                ignored_status_counts[status] = ignored_status_counts.get(status, 0) + 1
                continue
            source_release = row.get("source_release")
            if not source_release:
                raise ValueError("Valid summary row lacks source_release")
            identity = (str(Path(source_release).resolve()), str(row["task_id"]))
            if identity not in by_identity:
                raise ValueError("Measured position is absent from inventory: " + repr(identity))
            artifact_text = row.get("analysis_artifact")
            artifact_sha = row.get("analysis_sha256")
            if not artifact_text or not artifact_sha:
                raise ValueError("Valid summary row lacks analysis provenance")
            artifact = Path(artifact_text)
            if verify_artifacts and (not artifact.is_file() or _sha(artifact) != artifact_sha):
                raise ValueError("Analysis artifact changed: " + str(artifact))
            previous = evidence_by_identity.get(identity)
            if previous is not None and previous["analysis_sha256"] != artifact_sha:
                raise ValueError("Conflicting valid measurements: " + repr(identity))
            evidence_by_identity[identity] = row

    updated = 0
    already_valid = 0
    for identity, evidence in evidence_by_identity.items():
        target = by_identity[identity]
        if target.get("runtime_measurement_status") in {
            "VERIFIED", "RECORDED_MEASUREMENT_CHECKED"
        }:
            already_valid += 1
        else:
            target["runtime_measurement_status"] = "VERIFIED"
            updated += 1
        if row_key == "positions":
            target["support_status"] = "VALID_MEASUREMENT_COMPLETED"
            target["unsupported_reason"] = None
        target["triton_signature_measurement_evidence"] = {
            "case_id": evidence["case_id"],
            "analysis_artifact": evidence["analysis_artifact"],
            "analysis_sha256": evidence["analysis_sha256"],
            "claim_scope": "FIXED_SUITE_PARAMETER_WRITE",
            "coverage_only": True,
        }
    if row_key == "positions" and "summary" in result:
        rows = result["positions"]
        support = Counter(row["support_status"] for row in rows)
        family_support: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            family_support[row["operator_family"]][row["support_status"]] += 1
        distinct: dict[tuple[str, str], dict[str, Any]] = {}
        rank = {"IDENTIFIED": 0, "REFERENCE_AVAILABLE": 1,
                "READY_FOR_MEASUREMENT": 2, "VALID_MEASUREMENT_COMPLETED": 3}
        for row in rows:
            identity = (str(row.get("release_task_package_sha256", row["release"])),
                        str(row["task_id"]))
            old = distinct.get(identity)
            if old is None or rank.get(row["support_status"], -1) > rank.get(old["support_status"], -1):
                distinct[identity] = row
        distinct_support = Counter(row["support_status"] for row in distinct.values())
        distinct_family_support: dict[str, Counter[str]] = defaultdict(Counter)
        for row in distinct.values():
            distinct_family_support[row["operator_family"]][row["support_status"]] += 1
        result["summary"]["position_support_status_counts"] = dict(support)
        result["summary"]["distinct_position_support_status_counts"] = dict(distinct_support)
        result["summary"]["family_support_status_counts"] = {
            key: dict(value) for key, value in family_support.items()
        }
        result["summary"]["distinct_family_support_status_counts"] = {
            key: dict(value) for key, value in distinct_family_support.items()
        }
    result["triton_signature_measurement_merge"] = {
        "updated_positions": updated,
        "already_valid_positions": already_valid,
        "merged_valid_positions": len(evidence_by_identity),
        "ignored_nonvalid_status_counts": ignored_status_counts,
        "scope": (
            "Valid fixed-suite measurements attached to exact inventory positions; "
            "not a population, mechanism, root-cause, or training-outcome claim"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    outputs = [output] + ([args.summary_output.resolve()] if args.summary_output else [])
    if any(path.exists() or not path.is_relative_to(Path("/data1/tzh")) for path in outputs):
        parser.error("Choose new outputs under /data1/tzh")
    result = merge(
        _read(args.inventory), [_read(path) for path in args.summary], verify_artifacts=True,
    )
    result["triton_signature_measurement_merge_input_sha256"] = {
        str(args.inventory.resolve()): _sha(args.inventory),
        **{str(path.resolve()): _sha(path) for path in args.summary},
        str(Path(__file__).resolve()): _sha(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".gz":
        with gzip.open(output, "wt", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    else:
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(result["triton_signature_measurement_merge"], ensure_ascii=False))


if __name__ == "__main__":
    main()
