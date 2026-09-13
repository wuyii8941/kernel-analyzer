#!/usr/bin/env python3
"""Build an execution-only continuation for unfinished frozen signature tasks.

The source batch manifest remains the scientific selection. This utility
never inspects numerical outcomes: it carries forward only tasks without a
VALID status, excludes tasks with an explicit terminal measurement status,
and splits large runtime groups so one execution timeout cannot strand an
entire model/shape group.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_triton_signature_batches import _task_status


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_continuation(
    source: dict[str, Any], *, output_root: Path, max_cases_per_group: int,
    terminal_retry_tasks: set[str] | None = None,
    release_overrides: dict[str, str] | None = None,
    only_tasks: set[str] | None = None,
) -> dict[str, Any]:
    if max_cases_per_group < 1:
        raise ValueError("max_cases_per_group must be positive")
    terminal_retry_tasks = terminal_retry_tasks or set()
    release_overrides = release_overrides or {}
    groups: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source_group in source.get("groups", []):
        pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        source_rows = {
            row["task_id"]: row for row in source_group.get("source_campaigns", [])
        }
        for task_id, case in zip(source_group["task_ids"], source_group["cases"]):
            status = _task_status(source_group, task_id)
            row = {
                "task_id": task_id,
                "case_id": case["case_id"],
                "source_status": status,
                "source_group_id": source_group["group_id"],
            }
            if only_tasks is not None and task_id not in only_tasks:
                excluded.append({**row, "reason": "NOT_SELECTED_FOR_THIS_EXECUTION_CONTINUATION"})
            elif status == "VALID":
                excluded.append({**row, "reason": "ALREADY_VALID"})
            elif status not in {"NOT_STARTED", "INCOMPLETE_ATTEMPT"} and task_id not in terminal_retry_tasks:
                excluded.append({**row, "reason": "TERMINAL_TASK_STATUS_NOT_RETRIED"})
            else:
                pending.append((task_id, case, source_rows.get(task_id, {})))
        for offset in range(0, len(pending), max_cases_per_group):
            chunk = pending[offset:offset + max_cases_per_group]
            task_ids = [row[0] for row in chunk]
            identity = source_group["group_id"] + "\0" + "\0".join(task_ids)
            group_id = (
                "continuation-" + source_group["runtime"]["architecture"] + "-"
                + hashlib.sha256(identity.encode()).hexdigest()[:16]
            )
            old_release = str(Path(source_group["release"]).resolve())
            release = release_overrides.get(old_release, old_release)
            retried = [task for task in task_ids if task in terminal_retry_tasks]
            if retried and release == old_release:
                raise ValueError("Terminal retry requires a different runtime release")
            groups.append({
                "group_id": group_id,
                "release": release,
                "runtime": source_group["runtime"],
                "campaign_output": str((output_root / group_id).resolve()),
                "case_count": len(chunk),
                "task_ids": task_ids,
                "cases": [row[1] for row in chunk],
                "source_campaigns": [
                    {
                        **row[2],
                        "task_id": row[0],
                        "case_id": row[1]["case_id"],
                        "continuation_of_group": source_group["group_id"],
                        "continuation_of_output": source_group["campaign_output"],
                        "continuation_reason": _task_status(source_group, row[0]),
                        "explicit_terminal_retry": row[0] in terminal_retry_tasks,
                    }
                    for row in chunk
                ],
            })
    return {
        "schema": "triton-signature-runtime-continuation-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_selection_unchanged": True,
        "selection_uses_numerical_outcomes": False,
        "continuation_uses_execution_status_only": True,
        "prior_valid_measurements_are_not_repeated": True,
        "terminal_task_failures_are_not_retried_unless_explicit": True,
        "explicit_terminal_retry_tasks": sorted(terminal_retry_tasks),
        "release_overrides": dict(sorted(release_overrides.items())),
        "execution_continuation_task_filter": sorted(only_tasks) if only_tasks is not None else None,
        "max_cases_per_group": max_cases_per_group,
        "batch_size": min(int(source.get("batch_size", 1)), max_cases_per_group),
        "group_count": len(groups),
        "case_count": sum(group["case_count"] for group in groups),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-cases-per-group", type=int, default=4)
    parser.add_argument("--retry-terminal-task", action="append", default=[])
    parser.add_argument("--only-task", action="append", default=[])
    parser.add_argument("--release-override", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() or not output_root.is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    source_path = args.source_manifest.resolve()
    overrides: dict[str, str] = {}
    for item in args.release_override:
        if "=" not in item:
            parser.error("--release-override must be OLD=NEW")
        old, new = item.split("=", 1)
        old, new = str(Path(old).resolve()), str(Path(new).resolve())
        if old in overrides or not Path(new).exists():
            parser.error("Invalid, duplicate, or absent release override")
        overrides[old] = new
    result = build_continuation(
        json.loads(source_path.read_text()),
        output_root=output_root,
        max_cases_per_group=args.max_cases_per_group,
        terminal_retry_tasks=set(args.retry_terminal_task),
        release_overrides=overrides,
        only_tasks=set(args.only_task) if args.only_task else None,
    )
    result["source_manifest"] = str(source_path)
    result["source_manifest_sha256"] = _sha(source_path)
    output_root.mkdir(parents=True)
    for group in result["groups"]:
        plan = output_root / (group["group_id"] + ".plan.json")
        plan.write_text(json.dumps({"cases": group["cases"]}, indent=2) + "\n")
        group["case_plan"] = str(plan.resolve())
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest), "group_count": result["group_count"],
        "case_count": result["case_count"], "excluded_count": result["excluded_count"],
    }))


if __name__ == "__main__":
    main()
