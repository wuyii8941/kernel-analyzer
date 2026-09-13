#!/usr/bin/env python3
"""Group frozen Triton-signature selections by executable runtime.

This changes execution cost, not scientific selection: every position comes
from an already frozen signature manifest, keeps its original case identity,
and is measured separately.  Compatible positions share model loading and
state replay through the existing multi-case capture implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_triton_signature_queue import campaign_status


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    source: dict[str, Any],
    *,
    output_root: Path,
    batch_size: int,
    release_overrides: dict[str, str] | None = None,
    retry_tasks: set[str] | None = None,
    graph_break_releases: set[str] | None = None,
) -> dict[str, Any]:
    release_overrides = release_overrides or {}
    retry_tasks = retry_tasks or set()
    graph_break_releases = graph_break_releases or set()
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for campaign in source.get("campaigns", []):
        status = campaign_status(campaign)
        if campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}:
            excluded.append({
                "case_id": campaign["case"]["case_id"],
                "task_id": campaign["task_id"],
                "reason": "SPECIALIZED_ADAPTER_RETAINS_ITS_EXISTING_EXECUTION_PATH",
                "source_status": status,
            })
            continue
        retry = str(campaign["task_id"]) in retry_tasks
        if status != "NOT_STARTED" and not retry:
            excluded.append({
                "case_id": campaign["case"]["case_id"],
                "task_id": campaign["task_id"],
                "reason": "SOURCE_CAMPAIGN_ALREADY_STARTED_OR_TERMINAL",
                "source_status": status,
            })
            continue
        if retry and not status.startswith("TERMINAL_"):
            raise ValueError(
                "Explicit retry requires a terminal source campaign: "
                + str(campaign["task_id"])
            )
        runtime = campaign["runtime"]
        original_release = str(Path(campaign["release"]).resolve())
        release = release_overrides.get(original_release, original_release)
        if retry and release == original_release:
            raise ValueError("Explicit retry requires a different runtime release")
        if release != original_release and not Path(release).exists():
            raise ValueError("Release override does not exist: " + release)
        key = (
            release,
            str(runtime["architecture"]),
            str(Path(runtime["model"]).resolve()),
            str(Path(runtime["input_bank"]).resolve()),
            str(bool(runtime.get("allow_graph_breaks", False) or original_release in graph_break_releases)),
        )
        grouped[key].append(campaign)

    groups: list[dict[str, Any]] = []
    for key, campaigns in sorted(grouped.items()):
        release, architecture, model, input_bank, allow_graph_breaks = key
        identity = "\0".join(key)
        group_id = "runtime-" + architecture + "-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        groups.append({
            "group_id": group_id,
            "release": release,
            "runtime": {
                "architecture": architecture,
                "model": model,
                "input_bank": input_bank,
                "allow_graph_breaks": allow_graph_breaks == "True",
            },
            "campaign_output": str((output_root / group_id).resolve()),
            "case_count": len(campaigns),
            "task_ids": [item["task_id"] for item in campaigns],
            "cases": [item["case"] for item in campaigns],
            "source_campaigns": [{
                "case_id": item["case"]["case_id"],
                "task_id": item["task_id"],
                "operator_family": item["operator_family"],
                "signature_key": item.get("signature_key"),
                "source_campaign_output": item["campaign_output"],
                "source_release": str(Path(item["release"]).resolve()),
                "explicit_retry": str(item["task_id"]) in retry_tasks,
            } for item in campaigns],
        })

    return {
        "schema": "triton-signature-runtime-batch-manifest-v1",
        "selection_source_is_frozen_signature_manifest": True,
        "selection_uses_numerical_outcomes": False,
        "grouping_definition": "release + architecture + model + input_bank + allow_graph_breaks",
        "shared_capture_does_not_pool_case_statistics": True,
        "batch_size": batch_size,
        "release_overrides": dict(sorted(release_overrides.items())),
        "explicit_retry_tasks": sorted(retry_tasks),
        "graph_break_release_overrides": sorted(graph_break_releases),
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--release-override", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--retry-task", action="append", default=[])
    parser.add_argument("--allow-graph-breaks-release", action="append", default=[])
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_root.exists() or not args.output_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    source = json.loads(args.source_manifest.read_text())
    overrides: dict[str, str] = {}
    for item in args.release_override:
        if "=" not in item:
            parser.error("--release-override must be OLD=NEW")
        old, new = item.split("=", 1)
        old = str(Path(old).resolve())
        new = str(Path(new).resolve())
        if old in overrides or not old or not new:
            parser.error("Invalid or duplicate --release-override")
        overrides[old] = new
    result = build(
        source,
        output_root=args.output_root,
        batch_size=args.batch_size,
        release_overrides=overrides,
        retry_tasks=set(args.retry_task),
        graph_break_releases={str(Path(path).resolve()) for path in args.allow_graph_breaks_release},
    )
    result["source_manifest"] = str(args.source_manifest.resolve())
    result["source_manifest_sha256"] = _sha(args.source_manifest)
    args.output_root.mkdir(parents=True)
    for group in result["groups"]:
        plan = args.output_root / (group["group_id"] + ".plan.json")
        plan.write_text(json.dumps({"cases": group["cases"]}, indent=2, ensure_ascii=False) + "\n")
        group["case_plan"] = str(plan.resolve())
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "group_count": result["group_count"],
        "case_count": result["case_count"],
        "excluded_count": result["excluded_count"],
    }))


if __name__ == "__main__":
    main()
