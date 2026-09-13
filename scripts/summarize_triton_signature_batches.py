#!/usr/bin/env python3
"""Summarize runtime-batched Triton signature measurements per position."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in manifest.get("groups", []):
        output = Path(group["campaign_output"])
        timeout = output / "execution_timeout.json"
        for source in group["source_campaigns"]:
            task_id = source["task_id"]
            run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
            run = output / "runs" / run_id
            status_path = run / "status.json"
            if timeout.exists() and not status_path.exists():
                execution = "EXECUTION_TIMEOUT_NOT_MEASURED"
            elif status_path.exists():
                execution = str(_read(status_path).get("status", "UNDECLARED"))
            else:
                execution = "INCOMPLETE_ATTEMPT" if run.exists() else "NOT_STARTED"
            analysis_path = run / "analysis.json"
            analysis = _read(analysis_path) if analysis_path.exists() else {}
            bias = analysis.get("bias_analysis", {})
            rows.append({
                "group_id": group["group_id"],
                "case_id": source["case_id"],
                "task_id": task_id,
                "operator_family": source["operator_family"],
                "signature_key": source.get("signature_key"),
                "source_release": source.get("source_release", group.get("release")),
                "analysis_artifact": str(analysis_path.resolve()) if analysis_path.exists() else None,
                "analysis_sha256": _sha(analysis_path) if analysis_path.exists() else None,
                "execution_status": execution,
                "measurement_status": analysis.get("measurement_status", "NOT_ASSESSED"),
                "equivalence_decision": analysis.get("equivalence_decision", "NOT_ASSESSED"),
                "fixed_suite_parameter_write_total_rms": bias.get("fixed_suite_total_rms"),
                "fixed_suite_parameter_write_aligned_ratio": bias.get(
                    "fixed_suite_aligned_ratio_of_sums"
                ),
                "population_guarantee": False,
                "bias_or_root_cause_confirmed_by_this_row": False,
                "training_outcome_measured": False,
            })
    return {
        "schema": "triton-signature-runtime-batch-summary-v1",
        "selection_uses_numerical_outcomes": False,
        "shared_capture_does_not_pool_case_statistics": True,
        "position_count": len(rows),
        "execution_status_counts": dict(Counter(row["execution_status"] for row in rows)),
        "valid_measurement_count": sum(row["measurement_status"] == "VALID" for row in rows),
        "scope": "STRUCTURAL_SIGNATURE_COVERAGE; NOT_A_ROOT_CAUSE_OR_TRAINING_OUTCOME_COUNT",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    result = summarize(_read(args.manifest))
    result["manifest"] = str(args.manifest.resolve())
    result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "position_count": result["position_count"],
        "execution_status_counts": result["execution_status_counts"],
        "valid_measurement_count": result["valid_measurement_count"],
    }))


if __name__ == "__main__":
    main()
