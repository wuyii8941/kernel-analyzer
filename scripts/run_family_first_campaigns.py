#!/usr/bin/env python3
"""Freeze, resume, and report family-first generic coverage campaigns."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import hashlib


ROOT = Path(__file__).resolve().parents[1]


def command(campaign: dict, action: str, *, device: str) -> list[str]:
    base = [sys.executable, str(ROOT / "scripts/run_training_numerical_analysis.py"),
            "coverage", action, "--output", campaign["campaign_output"]]
    if action == "freeze":
        runtime = campaign["runtime"]
        result = [*base, "--release", campaign["release"], "--plans", campaign["case_plan"],
                "--architecture", runtime["architecture"], "--model", runtime["model"],
                "--input-bank", runtime["input_bank"], "--batch-size", "1"]
        if runtime.get("allow_graph_breaks"):
            result.append("--allow-graph-breaks")
        return result
    if action == "run":
        return [*base, "--device", device, "--limit", "1"]
    return base


def observed_status(campaign: dict, action: str, returncode: int) -> str:
    if returncode != 0:
        return "COMMAND_FAILED"
    output = Path(campaign["campaign_output"])
    if action == "freeze":
        return "FROZEN" if (output / "protocol.json").exists() else "FREEZE_OUTPUT_MISSING"
    if action == "run":
        run_id = hashlib.sha256(campaign["task_id"].encode()).hexdigest()[:20]
        status_path = output / "runs" / run_id / "status.json"
        if not status_path.exists():
            return "MEASUREMENT_INCOMPLETE_OR_RUNNING"
        task_status = json.loads(status_path.read_text()).get("status", "UNDECLARED")
        return "MEASUREMENT_VALID" if task_status == "VALID" else "MEASUREMENT_" + task_status
    reports = sorted((output / "reports").glob("*.json"))
    if not reports:
        return "REPORT_OUTPUT_MISSING"
    report = json.loads(reports[-1].read_text())
    return "REPORT_COMPLETE" if report.get("complete_all_measurements") else "REPORT_INCOMPLETE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "report"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-campaigns", type=int)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Explicit campaign offset for parallel workers; never selected from outcomes")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative")
    campaigns = payload["campaigns"][args.start_index:]
    if args.limit_campaigns is not None:
        campaigns = campaigns[:args.limit_campaigns]
    snapshot_dir = args.manifest.parent / "orchestration"
    snapshot_dir.mkdir(exist_ok=True)
    rows = []
    for campaign in campaigns:
        output = Path(campaign["campaign_output"])
        if args.action == "freeze" and (output / "protocol.json").exists():
            rows.append({"case_id": campaign["case"]["case_id"], "status": "ALREADY_FROZEN"})
            continue
        cmd = command(campaign, args.action, device=args.device)
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        status = observed_status(campaign, args.action, completed.returncode)
        rows.append({
            "case_id": campaign["case"]["case_id"],
            "operator_family": campaign["operator_family"],
            "returncode": completed.returncode,
            "status": status,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "command": cmd,
        })
        # Execution freezes and runs stop at the first unfinished family so a
        # later invocation resumes deterministically without selecting a
        # substitute.  Reporting is different: it must inspect every frozen
        # family and retain all failures in one snapshot.
        if (args.action != "report"
                and status not in {"FROZEN", "MEASUREMENT_VALID", "REPORT_COMPLETE"}):
            break
    result = {
        "schema": "family-first-campaign-orchestration-v1",
        "action": args.action,
        "device": args.device if args.action == "run" else None,
        "manifest": str(args.manifest.resolve()),
        "start_index": args.start_index,
        "rows": rows,
        "all_commands_completed": len(rows) == len(campaigns) and all(
            row["status"] in {"FROZEN", "ALREADY_FROZEN", "MEASUREMENT_VALID", "REPORT_COMPLETE"}
            for row in rows),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = snapshot_dir / f"{args.action}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "snapshot": str(path), "campaigns_attempted": len(rows),
        "all_commands_completed": result["all_commands_completed"],
        "last_status": rows[-1]["status"] if rows else "NO_CAMPAIGNS",
    }))
    if rows and rows[-1]["status"] not in {
        "FROZEN", "ALREADY_FROZEN", "MEASUREMENT_VALID", "REPORT_COMPLETE"
    }:
        raise SystemExit(rows[-1]["returncode"] or 1)


if __name__ == "__main__":
    main()
