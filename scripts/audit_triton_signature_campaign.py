#!/usr/bin/env python3
"""Reconcile a frozen Triton-signature selection across versioned executions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(master: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    frozen: dict[tuple[str, str], dict[str, Any]] = {}
    for campaign in master.get("campaigns", []):
        identity = (str(Path(campaign.get("original_release", campaign["release"])).resolve()),
                    str(campaign["task_id"]))
        if identity in frozen:
            raise ValueError("duplicate frozen identity")
        frozen[identity] = campaign
    attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if summary.get("schema") not in {
            "triton-signature-runtime-batch-summary-v1", "family-first-campaign-summary-v1"
        }:
            raise ValueError("unexpected summary schema")
        for row in summary.get("rows", []):
            source_release = row.get("source_release")
            if not source_release:
                continue
            identity = (str(Path(source_release).resolve()), str(row["task_id"]))
            if identity in frozen:
                attempts[identity].append(row)
    rows = []
    for identity, campaign in frozen.items():
        evidence = attempts.get(identity, [])
        valid = [row for row in evidence if row.get("measurement_status") == "VALID"]
        if valid:
            status = "VALID"
        else:
            reasons = "\n".join(str(row.get("failure_reason") or "") for row in evidence)
            executions = {str(row.get("execution_status")) for row in evidence}
            if "Live recurrence source differs" in reasons:
                status = "RUNTIME_PATH_MISMATCH_NOT_MEASURED"
            elif any("TIMEOUT" in value for value in executions):
                status = "EXECUTION_TIMEOUT_NOT_MEASURED"
            elif executions - {"NOT_STARTED", "None"}:
                status = "EXECUTION_FAILED_NOT_MEASURED"
            else:
                status = "NOT_STARTED"
        rows.append({
            "source_release": identity[0], "task_id": identity[1],
            "case_id": campaign["case"]["case_id"], "adapter": campaign.get("adapter"),
            "operator_family": campaign["operator_family"], "final_status": status,
            "attempt_count": len(evidence),
        })
    counts = Counter(row["final_status"] for row in rows)
    return {
        "schema": "triton-signature-campaign-audit-v1",
        "selection_uses_numerical_outcomes": False,
        "frozen_task_count": len(rows),
        "final_status_counts": dict(counts),
        "all_frozen_tasks_accounted": not counts.get("NOT_STARTED"),
        "valid_measurements_are_not_root_causes": True,
        "timeouts_and_path_mismatches_are_not_negative_results": True,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    result = audit(_read(args.master_manifest), [_read(path) for path in args.summary])
    result["input_sha256"] = {
        str(args.master_manifest.resolve()): _sha(args.master_manifest),
        **{str(path.resolve()): _sha(path) for path in args.summary},
        str(Path(__file__).resolve()): _sha(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "frozen_task_count": result["frozen_task_count"],
        "final_status_counts": result["final_status_counts"],
        "all_frozen_tasks_accounted": result["all_frozen_tasks_accounted"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
