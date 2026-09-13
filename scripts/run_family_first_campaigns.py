#!/usr/bin/env python3
"""Freeze, resume, and report family-first coverage campaigns."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import os


ROOT = Path(__file__).resolve().parents[1]


def subprocess_environment() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    required = f"{ROOT / 'src'}:{ROOT}"
    return {
        **os.environ,
        "PYTHONPATH": required if not existing else required + ":" + existing,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def command(campaign: dict, action: str, *, device: str) -> list[str]:
    if campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}:
        if action in {"freeze", "report"}:
            # These actions are handled by the orchestrator itself for
            # family-specific runners; never launch a GPU process here.
            return [sys.executable, "-c", "pass"]
        output = Path(campaign["campaign_output"])
        return [
            sys.executable, str(ROOT / campaign["capture_script"]),
            "--" + campaign["specialized_plan_argument"], campaign["specialized_plan"],
            "--case-plan", campaign["case_plan"],
            "--states", "32",
            "--training-bias-profile-v2-output-dir", str(output / "raw"),
            "--output-dir", str(output / "legacy"),
            "--spool-dir", str(output / "spool"),
            "--input-bank", campaign["runtime"]["input_bank"],
            "--architecture", campaign["runtime"]["architecture"],
            "--release-dir", campaign["release"],
            "--model", campaign["runtime"]["model"],
            "--device", device,
        ]
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
    if campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}:
        if action == "freeze":
            return "FROZEN" if (output / "protocol.json").exists() else "FREEZE_OUTPUT_MISSING"
        if action == "run":
            completion = output / "completion_verification.json"
            if not completion.exists():
                return "MEASUREMENT_INCOMPLETE_OR_RUNNING"
            report = json.loads(completion.read_text())
            complete = report.get("measurement_complete") or report.get("recorded_measurement_complete")
            return "MEASUREMENT_VALID" if complete else "MEASUREMENT_INCOMPLETE"
        return "REPORT_COMPLETE" if list((output / "reports").glob("*.json")) else "REPORT_OUTPUT_MISSING"
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
        if (campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}
                and args.action == "freeze"):
            if output.exists():
                raise SystemExit("Specialized campaign output already exists: " + str(output))
            output.mkdir(parents=True)
            freeze = {
                "schema": "family-first-specialized-freeze-v1",
                "adapter": campaign["adapter"],
                "operator_family": campaign["operator_family"],
                "reference_method": campaign["case"]["reference_method"],
                "case": campaign["case"],
                "case_plan": campaign["case_plan"],
                "specialized_plan": campaign["specialized_plan"],
                "runtime": campaign["runtime"],
                "release": campaign["release"],
                "capture_script": campaign["capture_script"],
                "selection_uses_numerical_outcomes": False,
            }
            (output / "protocol.json").write_text(json.dumps(freeze, indent=2) + "\n")
            rows.append({"case_id": campaign["case"]["case_id"], "status": "FROZEN"})
            continue
        cmd = command(campaign, args.action, device=args.device)
        environment = subprocess_environment()
        completed = subprocess.run(
            cmd, cwd=ROOT, text=True, capture_output=True, env=environment
        )
        if (args.action == "run"
                and campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}
                and completed.returncode == 0):
            # Specialized capture scripts write the raw protocol but do not
            # share the generic coverage finalizer.  Audit immediately and
            # keep the result in the same campaign directory.
            finalizer = (
                ROOT / "scripts/finalize_residual_rms_forward.py"
                if campaign["adapter"] == "RESIDUAL_RMS_FORWARD"
                else ROOT / "scripts/finalize_decayed_recurrence.py"
            )
            audit = subprocess.run(
                [sys.executable, str(finalizer), "--root", str(output),
                 "--output", str(output / "completion_verification.json")],
                cwd=ROOT, text=True, capture_output=True, env=environment,
            )
            if audit.returncode != 0:
                completed = subprocess.CompletedProcess(
                    completed.args, audit.returncode,
                    completed.stdout + "\n" + audit.stdout,
                    completed.stderr + "\n" + audit.stderr,
                )
        status = observed_status(campaign, args.action, completed.returncode)
        if args.action == "run" and completed.returncode != 0:
            failure_path = output / "execution_failure.json"
            if not failure_path.exists():
                failure_path.write_text(json.dumps({
                    "schema": "family-first-execution-failure-v1",
                    "case_id": campaign["case"]["case_id"],
                    "task_id": campaign["task_id"],
                    "returncode": completed.returncode,
                    "command": cmd,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                    "not_a_numerical_measurement": True,
                }, indent=2, ensure_ascii=False) + "\n")
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
