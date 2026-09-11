#!/usr/bin/env python3
"""Turn the family-first queue into resumable generic coverage campaigns.

Only reference methods already supported by the generic capture engine are
included.  Existing family-specific reference runners remain visible as
unsupported by this manifest rather than being silently replaced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


GENERIC_METHODS = {"AOT_REPLAY", "EXTERNAL_FP32_RECOMPUTE"}


def discover_runtime_configs(protocol_paths: list[Path]) -> dict[str, dict]:
    candidates = defaultdict(set)
    sources = defaultdict(list)
    for path in protocol_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError):
            continue
        release = payload.get("release")
        values = (
            payload.get("architecture"),
            payload.get("model"),
            payload.get("input_bank"),
            bool(payload.get("allow_graph_breaks", False)),
        )
        if release and all(values[:3]):
            candidates[str(Path(release).resolve())].add(values)
            sources[(str(Path(release).resolve()), values)].append(str(path.resolve()))
    result = {}
    for release, values in candidates.items():
        if len(values) != 1:
            continue
        architecture, model, input_bank, allow_graph_breaks = next(iter(values))
        result[release] = {
            "architecture": architecture,
            "model": model,
            "input_bank": input_bank,
            "allow_graph_breaks": allow_graph_breaks,
            "source_protocols": sorted(
                sources[(release, (architecture, model, input_bank, allow_graph_breaks))]
            ),
        }
    return result


def build(queue: dict, runtime_configs: dict[str, dict], *, output_root: Path) -> dict:
    selected = []
    skipped = []
    seen_families = set()
    for row in queue["rows"]:
        if row["wave"] != "NEW_FAMILY_FIRST":
            continue
        family = row["operator_family"]
        if family in seen_families:
            continue
        bindings = row.get("reference_candidates", [])
        methods = {binding.get("reference_method") for binding in bindings}
        usable = sorted(methods & GENERIC_METHODS)
        if len(usable) != 1:
            skipped.append({
                "operator_family": family, "release": row["release"],
                "task_id": row["task_id"],
                "reason": "GENERIC_REFERENCE_METHOD_UNAVAILABLE",
                "declared_methods": sorted(str(value) for value in methods),
            })
            seen_families.add(family)
            continue
        config = runtime_configs.get(str(Path(row["release"]).resolve()))
        if config is None:
            skipped.append({
                "operator_family": family, "release": row["release"],
                "task_id": row["task_id"], "reason": "RUNTIME_CONFIG_NOT_UNIQUELY_RECOVERED",
            })
            seen_families.add(family)
            continue
        method = usable[0]
        case_hash = hashlib.sha256((row["release"] + "\0" + row["task_id"]).encode()).hexdigest()[:16]
        case_id = f"family-{family.lower().replace('_', '-')}-{case_hash}"
        selected.append({
            "operator_family": family,
            "implementation_kind": row.get("implementation_kind", "UNDECLARED"),
            "release": row["release"],
            "task_id": row["task_id"],
            "case": {
                "case_id": case_id,
                "task_id": row["task_id"],
                "carrier": row["carrier"],
                "reference_method": method,
                "family": family,
                "selection_rule": "FAMILY_FIRST_QUEUE_WITHOUT_NUMERICAL_OUTCOMES",
            },
            "runtime": config,
            "campaign_output": str((output_root / case_id).resolve()),
        })
        seen_families.add(family)
    return {
        "schema": "family-first-generic-campaign-manifest-v1",
        "selection_uses_numerical_outcomes": False,
        "selection_scope": "FIRST_CANONICAL_POSITION_PER_FAMILY_WITH_GENERIC_REFERENCE_AND_RECOVERED_RUNTIME_CONFIG",
        "selected_campaign_count": len(selected),
        "selected_operator_families": [row["operator_family"] for row in selected],
        "execution_policy": {
            "mode": "BOUNDED_FAMILY_VALIDATION_NOT_POSITION_EXPANSION",
            "maximum_one_campaign_per_operator_family": True,
            "automatic_continuation_after_failure": False,
            "coverage_position_count_is_not_a_success_metric": True,
        },
        "skipped_new_family_rows": skipped,
        "campaigns": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_root / "manifest.json"
    if args.output_root.exists() or not args.output_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    queue = json.loads(args.queue.read_text())
    protocols = sorted(args.protocol_root.glob("**/protocol.json"))
    runtime_configs = discover_runtime_configs(protocols)
    result = build(queue, runtime_configs, output_root=args.output_root)
    result["queue_sha256"] = hashlib.sha256(args.queue.read_bytes()).hexdigest()
    result["runtime_config_protocol_count"] = len(protocols)
    args.output_root.mkdir(parents=True)
    for campaign in result["campaigns"]:
        path = args.output_root / (campaign["case"]["case_id"] + ".plan.json")
        path.write_text(json.dumps({"cases": [campaign["case"]]}, indent=2) + "\n")
        campaign["case_plan"] = str(path.resolve())
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "selected_campaign_count": result["selected_campaign_count"],
        "selected_operator_families": result["selected_operator_families"],
        "skipped": len(result["skipped_new_family_rows"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
