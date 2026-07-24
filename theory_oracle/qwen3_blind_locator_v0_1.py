#!/usr/bin/env python
"""Bug-agnostic Qwen3 historical locator using only Oracle and local traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from forkcert.relational_oracle import compute_endpoint_oracle
from theory_oracle.qwen3_historical_case_runner_v0_1 import execute_case


def first_trace_mismatch(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left.get("op") != right.get("op"):
            return {"index": index, "reference": left.get("op"), "candidate": right.get("op"), "kind": "operation_sequence"}
        left_outputs = left.get("outputs", [])
        right_outputs = right.get("outputs", [])
        if len(left_outputs) != len(right_outputs):
            return {"index": index, "reference_outputs": len(left_outputs), "candidate_outputs": len(right_outputs), "kind": "output_arity"}
        for output_index, (lout, rout) in enumerate(zip(left_outputs, right_outputs)):
            if lout.get("sha256") != rout.get("sha256") or lout.get("shape") != rout.get("shape"):
                return {"index": index, "op": left.get("op"), "output_index": output_index, "reference": lout, "candidate": rout, "kind": "operation_value"}
    if len(reference) != len(candidate):
        return {"index": min(len(reference), len(candidate)), "reference_remaining": len(reference), "candidate_remaining": len(candidate), "kind": "operation_length"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--candidate-source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    manifest = json.loads((case_dir / "case_manifest.json").read_text())
    run_dir = args.out.resolve().parent / (args.out.stem + "_run")
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    run = execute_case(args.candidate_source_root, case_dir, run_dir)
    reference_endpoint = torch.load(case_dir / manifest["reference"]["endpoint"]["path"], map_location="cpu", weights_only=True).numpy()
    candidate_endpoint = torch.load(run_dir / run["endpoint"]["path"], map_location="cpu", weights_only=True).numpy()
    endpoint = compute_endpoint_oracle(reference_endpoint, candidate_endpoint).as_dict()
    region_rows: list[dict[str, Any]] = []
    for name, reference_row in manifest["reference"]["regions"].items():
        ref = torch.load(case_dir / reference_row["path"], map_location="cpu", weights_only=True).numpy()
        cand_row = run["regions"].get(name)
        if cand_row is None:
            region_rows.append({"region": name, "present": False, "oracle": None})
            continue
        cand = torch.load(run_dir / cand_row["path"], map_location="cpu", weights_only=True).numpy()
        region_rows.append({"region": name, "present": True, "oracle": compute_endpoint_oracle(ref, cand).as_dict()})
    reference_trace = json.loads((case_dir / manifest["reference"]["trace"]["path"]).read_text())["records"]
    candidate_trace = json.loads((run_dir / run["trace"]["path"]).read_text())["records"]
    region_by_name = {row["region"]: row for row in region_rows}
    declared_order = list(manifest["subject"]["regions"])
    mismatched_regions = [
        name for name in declared_order
        if name in region_by_name
        and region_by_name[name]["present"]
        and not region_by_name[name]["oracle"]["exact_match"]
    ]
    report = {
        "schema_version": "forkcert.qwen3-blind-locator.v0.1",
        "case_id": manifest["case_id"],
        "candidate_run": str((run_dir / "run.json").resolve()),
        "endpoint_oracle": endpoint,
        "region_oracle": region_rows,
        "first_changed_region": mismatched_regions[0] if mismatched_regions else None,
        "trace_alignment": {"reference_records": len(reference_trace), "candidate_records": len(candidate_trace), "first_mismatch": first_trace_mismatch(reference_trace, candidate_trace)},
        "repeatability": run["endpoint"]["repeatability"],
        "claim": {
            "level": "OPERATION_INTERVAL_CANDIDATE" if endpoint["exact_match"] is False and mismatched_regions else "OBSERVATION",
            "statement": "generic trace and boundary comparison identifies a candidate interval; no historical patch or unique root cause is inferred",
        },
        "nonclaims": ["no issue/patch metadata was consumed", "first changed region is not automatically the root cause", "Oracle discrepancy is implementation/reference-relative"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
