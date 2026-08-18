#!/usr/bin/env python3
"""Join immutable discovery candidates to completed live follow-up results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    queue = json.loads((COVERAGE / "bias_candidate_queue.json").read_text())
    observed = {}
    for path in sorted((COVERAGE / "live_contrasts").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "COMPLETE_LIVE_FULL_COORDINATE_CONTRASTS":
            continue
        for row in payload["results"]:
            bound = dict(row); bound["live_result_sha256"] = payload["result_sha256"]
            observed.setdefault(row["candidate_id"], {})[row["contrast_axis"]] = bound
    cases = {}
    complete_cases = {}
    for name in ("phi4_seq64_lmhead_dx.json", "qwen64_vproj.json", "qwen128_vproj.json",
                 "mamba_seq64_input_proj.json"):
        case_path = COVERAGE / "cases" / name
        if not case_path.exists():
            continue
        case = json.loads(case_path.read_text())
        cases[case["candidate_id"]] = case
        if case["status"].startswith("COMPLETE_"):
            complete_cases[case["candidate_id"]] = case
    terminal_trajectories = {}
    for name in ("qwen64_vproj_trajectory.json", "qwen128_vproj_trajectory.json",
                 "mamba_seq64_input_proj_trajectory.json"):
        path = COVERAGE / "cases" / name
        if path.exists():
            trajectory = json.loads(path.read_text())
            terminal_trajectories[trajectory["candidate_id"]] = trajectory
    rows = []
    for candidate in queue["candidates"]:
        cid = candidate["candidate_id"]
        arms = observed.get(cid)
        if candidate["claim"] != "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING":
            disposition = candidate["claim"]
        elif cid in terminal_trajectories and terminal_trajectories[cid]["status"] \
                == "PASS_STRICT_FLASH_STYLE_CASE":
            disposition = "COMPLETE_BOUNDED_FLASH_STYLE_FB_BIAS_CASE"
        elif cid in terminal_trajectories and terminal_trajectories[cid]["status"] \
                == "FAIL_DIRECTIONAL_ACCUMULATION":
            disposition = "COMPLETE_FB_CASE_REJECTED_DIRECTIONAL_ACCUMULATION"
        elif cid in complete_cases:
            disposition = "COMPLETE_BOUNDED_FLASH_STYLE_FB_BIAS_CASE"
        elif arms and not any(row["t1_eligible"] for key, row in arms.items()
                              if key != "TOTAL"):
            disposition = "REJECTED_BY_CORRECTED_FULL_COORDINATE_DIRECTION"
        elif arms:
            disposition = "T1_POSITIVE_PENDING_COMPLETE_CASE_GATES"
        else:
            disposition = "PENDING_LIVE_FULL_COORDINATE_FOLLOWUP"
        rows.append({
            "candidate_id": cid, "architecture": candidate["architecture"],
            "sequence_length": candidate["sequence_length"],
            "disposition": disposition,
            "live_result_sha256": next((row.get("live_result_sha256")
                                         for row in (arms or {}).values()), None),
            "case_result_sha256": cases.get(cid, {}).get("result_sha256"),
            "trajectory_result_sha256": terminal_trajectories.get(cid, {})
            .get("result_sha256"),
        })
    counts = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    payload = {
        "schema": "kernel-analyzer-live-contrast-dispositions-v1",
        "status": "COMPLETE_LIVE_FULL_COORDINATE_FOLLOWUP"
        if all(row["disposition"] != "PENDING_LIVE_FULL_COORDINATE_FOLLOWUP"
               for row in rows) else "PARTIAL_LIVE_FOLLOWUP",
        "queue_sha256": queue["result_sha256"], "candidate_count": len(rows),
        "counts": counts, "rows": rows,
        "claim_boundary": "Untouched candidates remain in the denominator; sampled discovery positives are not cases.",
    }
    payload["result_sha256"] = canonical(payload)
    output = COVERAGE / "live_contrast_dispositions.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
