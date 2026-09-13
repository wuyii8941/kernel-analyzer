#!/usr/bin/env python3
"""Freeze a versioned continuation for one specialized signature campaign.

The scientific task and selection rule are copied from an existing frozen
campaign. Only the specialized reference plan and output location change.
This supports source-version repairs before any valid measurement exists and
never selects a replacement based on numerical outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: dict[str, Any], *, specialized_plan: Path,
          output_root: Path, task_id: str | None = None) -> dict[str, Any]:
    campaigns = source.get("campaigns", [])
    if task_id is not None:
        campaigns = [row for row in campaigns if str(row.get("task_id")) == task_id]
    if len(campaigns) != 1:
        raise ValueError("source manifest must identify exactly one campaign")
    old = campaigns[0]
    if old.get("adapter") in {None, "GENERIC_COVERAGE"}:
        raise ValueError("source campaign is not specialized")
    plan = _read(specialized_plan)
    matches = [row for row in plan.get("cases", [])
               if str(row.get("task_id")) == str(old["task_id"])]
    if len(matches) != 1:
        raise ValueError("new specialized plan must contain the frozen task exactly once")
    old_output = Path(old["campaign_output"])
    completion = old_output / "completion_verification.json"
    if completion.exists():
        records = _read(completion).get("records", [])
        if any(str(row.get("task_id")) == str(old["task_id"])
               and row.get("status") in {"VERIFIED", "RECORDED_MEASUREMENT_CHECKED"}
               for row in records):
            raise ValueError("source task already has a valid measurement")
    case_id = str(old["case"]["case_id"]) + "-source-refresh"
    campaign_output = output_root / case_id
    case_plan = output_root / (case_id + ".plan.json")
    campaign = {
        **old,
        "case": {**old["case"], "case_id": case_id},
        "specialized_plan": str(specialized_plan.resolve()),
        "case_plan": str(case_plan.resolve()),
        "campaign_output": str(campaign_output.resolve()),
        "continuation_of_output": str(old_output.resolve()),
        "continuation_reason": "SPECIALIZED_REFERENCE_SOURCE_VERSION_REFRESH",
    }
    return {
        "schema": "specialized-signature-continuation-manifest-v1",
        "scientific_selection_unchanged": True,
        "selection_uses_numerical_outcomes": False,
        "replacement_task_selected": False,
        "source_campaign_task_had_valid_measurement": False,
        "campaigns": [campaign],
        "case_plan_payload": {"cases": matches},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--specialized-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() or not output_root.is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    source_path = args.source_manifest.resolve()
    specialized_plan = args.specialized_plan.resolve()
    result = build(_read(source_path), specialized_plan=specialized_plan,
                   output_root=output_root, task_id=args.task_id)
    result["source_manifest"] = str(source_path)
    result["source_manifest_sha256"] = _sha(source_path)
    result["specialized_plan_sha256"] = _sha(specialized_plan)
    output_root.mkdir(parents=True)
    campaign = result["campaigns"][0]
    Path(campaign["case_plan"]).write_text(
        json.dumps(result.pop("case_plan_payload"), indent=2) + "\n"
    )
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "task_id": campaign["task_id"]}))


if __name__ == "__main__":
    main()
