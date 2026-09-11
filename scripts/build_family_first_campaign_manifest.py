#!/usr/bin/env python3
"""Turn the family-first queue into resumable coverage campaigns.

The default keeps the original generic-only manifest for reproducibility.
``--include-specialized`` adds already-audited family adapters (normalization
and recurrence) without changing their reference definitions or selecting on
numerical outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


GENERIC_METHODS = {"AOT_REPLAY", "EXTERNAL_FP32_RECOMPUTE"}
SPECIALIZED_METHODS = {
    "RESIDUAL_RMS_FORWARD_COMMON_INPUT": {
        "adapter": "RESIDUAL_RMS_FORWARD",
        "script": "scripts/run_residual_rms_forward_capture.py",
        "plan_argument": "family-plan",
    },
    "DECAYED_RECURRENCE_COMMON_INPUT": {
        "adapter": "DECAYED_RECURRENCE",
        "script": "scripts/run_decayed_recurrence_capture.py",
        "plan_argument": "recurrence-plan",
    },
    "SEGMENTED_RECURRENCE_FIRST_COMMON_INPUT": {
        "adapter": "SEGMENTED_RECURRENCE_FIRST",
        "script": "scripts/run_decayed_recurrence_capture.py",
        "plan_argument": "recurrence-plan",
    },
    "CONTINUED_RECURRENCE_COMMON_INPUT": {
        "adapter": "CONTINUED_RECURRENCE",
        "script": "scripts/run_decayed_recurrence_capture.py",
        "plan_argument": "recurrence-plan",
    },
}


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


def build(
    queue: dict,
    runtime_configs: dict[str, dict],
    *,
    output_root: Path,
    include_specialized: bool = False,
    specialized_plan_overrides: dict[str, str] | None = None,
    frontier_families: set[str] | None = None,
    release_overrides: dict[str, str] | None = None,
    allow_graph_breaks_families: set[str] | None = None,
) -> dict:
    specialized_plan_overrides = specialized_plan_overrides or {}
    release_overrides = release_overrides or {}
    allow_graph_breaks_families = allow_graph_breaks_families or set()
    selected = []
    skipped = []
    seen_families = set()
    for row in queue["rows"]:
        if row["wave"] != "NEW_FAMILY_FIRST":
            continue
        family = row["operator_family"]
        if frontier_families is not None and family not in frontier_families:
            continue
        if family in seen_families:
            continue
        bindings = row.get("reference_candidates", [])
        methods = {binding.get("reference_method") for binding in bindings}
        usable = sorted(methods & (GENERIC_METHODS | (
            set(SPECIALIZED_METHODS) if include_specialized else set()
        )))
        if len(usable) != 1:
            skipped.append({
                "operator_family": family, "release": row["release"],
                "task_id": row["task_id"],
                "reason": (
                    "GENERIC_OR_SPECIALIZED_REFERENCE_METHOD_UNAVAILABLE"
                    if include_specialized else "GENERIC_REFERENCE_METHOD_UNAVAILABLE"
                ),
                "declared_methods": sorted(str(value) for value in methods),
            })
            seen_families.add(family)
            continue
        original_release = row["release"]
        config = runtime_configs.get(str(Path(original_release).resolve()))
        if config is None:
            skipped.append({
                "operator_family": family, "release": row["release"],
                "task_id": row["task_id"], "reason": "RUNTIME_CONFIG_NOT_UNIQUELY_RECOVERED",
            })
            seen_families.add(family)
            continue
        config = dict(config)
        if family in allow_graph_breaks_families:
            config["allow_graph_breaks"] = True
        method = usable[0]
        release = release_overrides.get(family, original_release)
        if family in release_overrides and not Path(release).exists():
            raise ValueError(f"release override does not exist: {release}")
        case_hash = hashlib.sha256((release + "\0" + row["task_id"]).encode()).hexdigest()[:16]
        case_id = f"family-{family.lower().replace('_', '-')}-{case_hash}"
        binding = next(
            item for item in bindings if item.get("reference_method") == method
        )
        specialized = SPECIALIZED_METHODS.get(method)
        specialized_plan = (
            specialized_plan_overrides.get(family, binding.get("bound_plan"))
            if specialized else None
        )
        selected.append({
            "operator_family": family,
            "implementation_kind": row.get("implementation_kind", "UNDECLARED"),
            "release": release,
            "original_release": original_release,
            "task_id": row["task_id"],
            "case": {
                "case_id": case_id,
                "task_id": row["task_id"],
                "carrier": row["carrier"],
                "reference_method": method,
                "family": family,
                "selection_rule": "FAMILY_FIRST_QUEUE_WITHOUT_NUMERICAL_OUTCOMES",
            },
            "adapter": specialized["adapter"] if specialized else "GENERIC_COVERAGE",
            "capture_script": specialized["script"] if specialized else None,
            "specialized_plan_argument": specialized["plan_argument"] if specialized else None,
            "specialized_plan": specialized_plan,
            "specialized_plan_override": (
                family in specialized_plan_overrides if specialized else False
            ),
            "release_override": family in release_overrides,
            "runtime": config,
            "campaign_output": str((output_root / case_id).resolve()),
        })
        seen_families.add(family)
    return {
        "schema": (
            "family-first-campaign-manifest-v2"
            if include_specialized else "family-first-generic-campaign-manifest-v1"
        ),
        "selection_uses_numerical_outcomes": False,
        "specialized_plan_overrides": specialized_plan_overrides,
        "selection_scope": (
            "STATIC_UNMEASURED_FRONTIER_FAMILY_ROWS_ONLY"
            if frontier_families is not None else
            "FIRST_CANONICAL_POSITION_PER_FAMILY_WITH_GENERIC_OR_AUDITED_SPECIALIZED_REFERENCE_AND_RECOVERED_RUNTIME_CONFIG"
            if include_specialized else
            "FIRST_CANONICAL_POSITION_PER_FAMILY_WITH_GENERIC_REFERENCE_AND_RECOVERED_RUNTIME_CONFIG"
        ),
        "frontier_families": sorted(frontier_families) if frontier_families is not None else None,
        "release_overrides": dict(sorted(release_overrides.items())),
        "allow_graph_breaks_families": sorted(allow_graph_breaks_families),
        "selected_campaign_count": len(selected),
        "selected_operator_families": [row["operator_family"] for row in selected],
        "execution_policy": {
            "mode": "BOUNDED_FAMILY_VALIDATION_NOT_POSITION_EXPANSION",
            "maximum_one_campaign_per_operator_family": True,
            "automatic_continuation_after_failure": False,
            "coverage_position_count_is_not_a_success_metric": True,
            "specialized_adapters_use_declared_reference_plans": True,
        },
        "skipped_new_family_rows": skipped,
        "campaigns": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--include-specialized", action="store_true",
        help="Include existing audited normalization/recurrence adapters.",
    )
    parser.add_argument(
        "--specialized-plan-override", action="append", default=[], metavar="FAMILY=PATH",
        help="Use a freshly rebound audited plan for one specialized family.",
    )
    parser.add_argument(
        "--frontier", type=Path,
        help="Use only families listed in an unmeasured-family frontier report.",
    )
    parser.add_argument(
        "--family", action="append", default=[], dest="selected_families",
        help="Explicitly select one or more audited family IDs (static selection only).",
    )
    parser.add_argument(
        "--release-override", action="append", default=[], metavar="FAMILY=PATH",
        help="Use a separately rebound runtime release for one selected family.",
    )
    parser.add_argument(
        "--allow-graph-breaks-family", action="append", default=[], dest="graph_break_families",
        help="Declare graph-break-compatible compilation for one selected family.",
    )
    args = parser.parse_args()
    manifest_path = args.output_root / "manifest.json"
    if args.output_root.exists() or not args.output_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    queue = json.loads(args.queue.read_text())
    protocols = sorted(args.protocol_root.glob("**/protocol.json"))
    runtime_configs = discover_runtime_configs(protocols)
    overrides = {}
    for item in args.specialized_plan_override:
        if "=" not in item:
            parser.error("--specialized-plan-override must be FAMILY=PATH")
        family, path = item.split("=", 1)
        if not family or not path or family in overrides:
            parser.error("Invalid or duplicate specialized plan override")
        overrides[family] = str(Path(path).resolve())
    frontier_families = None
    if args.frontier:
        frontier_payload = json.loads(args.frontier.read_text())
        frontier_families = {
            str(row["operator_family"])
            for row in frontier_payload.get("new_family_targets", [])
            if row.get("operator_family")
        }
    if args.selected_families:
        frontier_families = set(args.selected_families)
    release_overrides = {}
    for item in args.release_override:
        if "=" not in item:
            parser.error("--release-override must be FAMILY=PATH")
        family, path = item.split("=", 1)
        if not family or not path or family in release_overrides:
            parser.error("Invalid or duplicate release override")
        release_overrides[family] = str(Path(path).resolve())
    graph_break_families = set(args.graph_break_families)
    result = build(
        queue, runtime_configs, output_root=args.output_root,
        include_specialized=args.include_specialized,
        specialized_plan_overrides=overrides,
        frontier_families=frontier_families,
        release_overrides=release_overrides,
        allow_graph_breaks_families=graph_break_families,
    )
    result["queue_sha256"] = hashlib.sha256(args.queue.read_bytes()).hexdigest()
    if args.frontier:
        result["frontier_sha256"] = hashlib.sha256(args.frontier.read_bytes()).hexdigest()
    result["runtime_config_protocol_count"] = len(protocols)
    args.output_root.mkdir(parents=True)
    for campaign in result["campaigns"]:
        path = args.output_root / (campaign["case"]["case_id"] + ".plan.json")
        if campaign.get("specialized_plan"):
            source = Path(campaign["specialized_plan"])
            if not source.exists():
                raise SystemExit("Declared specialized plan is missing: " + str(source))
            payload = json.loads(source.read_text())
            matches = [
                case for case in payload.get("cases", [])
                if case.get("task_id") == campaign["task_id"]
            ]
            if len(matches) != 1:
                raise SystemExit(
                    "Specialized plan must contain exactly one selected task: "
                    + campaign["task_id"]
                )
            plan_payload = {"cases": matches}
        else:
            plan_payload = {"cases": [campaign["case"]]}
        path.write_text(json.dumps(plan_payload, indent=2) + "\n")
        campaign["case_plan"] = str(path.resolve())
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "selected_campaign_count": result["selected_campaign_count"],
        "selected_operator_families": result["selected_operator_families"],
        "skipped": len(result["skipped_new_family_rows"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
