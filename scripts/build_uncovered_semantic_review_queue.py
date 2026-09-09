#!/usr/bin/env python3
"""Attach compiler-carried training semantics to uncovered Triton structures.

This is a review queue, not an automatic operator-family classifier.  It reads
saved source/provenance only and never reads numerical measurements.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def release_root(source):
    source = Path(source).resolve()
    try:
        trace_index = source.parts.index("trace")
    except ValueError as exc:
        raise ValueError(f"Source is not inside a release trace: {source}") from exc
    return Path(*source.parts[:trace_index])


def trace_context(source):
    """Return release/trace coordinates when compiler provenance can exist."""
    source = Path(source).resolve()
    try:
        root = release_root(source)
    except ValueError:
        return None
    return root, relative_trace_source(source, root)


def relative_trace_source(source, root):
    return str(Path(source).resolve().relative_to((root / "trace").resolve()))


def load_release_semantics(root):
    bridge_path = root / "candidate_fb_bridge.json.gz"
    campaign_path = root / "campaign.json.gz"
    aot_paths = [
        root / "default_aot_capture.json.gz",
        root / "default_aot_capture_raw.json.gz",
    ]
    aot_path = next((path for path in aot_paths if path.exists()), None)
    if not campaign_path.exists():
        return {
            "bindings": {},
            "campaign_source_bindings": {},
            "fallback_bindings": {},
            "provenance_sha256": {
                "candidate_fb_bridge": None,
                "campaign": None,
                "default_aot_capture": None,
            },
            "aot_metadata_status": (
                "AVAILABLE" if aot_path is not None else "NOT_PRESENT_IN_RELEASE"
            ),
            "semantic_metadata_status": "MISSING_REQUIRED_PROVENANCE",
            "missing_required_provenance": [str(campaign_path)],
        }

    bridge_raw = bridge_path.read_bytes() if bridge_path.exists() else None
    campaign_raw = campaign_path.read_bytes()
    aot_raw = aot_path.read_bytes() if aot_path is not None else None
    bridge = read_json(bridge_path) if bridge_path.exists() else {"rows": []}
    campaign = read_json(campaign_path)
    aot = read_json(aot_path) if aot_path is not None else {"capture": {"graphs": []}}

    campaigns = {}
    campaign_source_bindings = {}
    for row in campaign.get("rows", []):
        key = (row.get("region_id"), row.get("symbol"))
        campaigns.setdefault(key, []).append(row)

    def campaign_record(row, *, binding_status, binding_method, aot_node_ids):
        source_nodes = sorted(set(row.get("source_nodes", [])))
        record = {
            "candidate_region_id": row.get("region_id"),
            "binding_status": binding_status,
            "binding_method": binding_method,
            "aot_node_ids": aot_node_ids,
            "original_aten": sorted(set(row.get("original_aten", []))),
            "source_nodes": source_nodes,
            "reference_kinds": ([row["reference_kind"]]
                                if row.get("reference_kind") else []),
        }
        module_paths = set()
        stack_traces = set()
        for node_name in source_nodes:
            for node in nodes.get(node_name, []):
                stack = node.get("fwd_nn_module_stack") or node.get("nn_module_stack") or {}
                for value in stack.values():
                    if isinstance(value, (list, tuple)) and value:
                        module_paths.add(str(value[0]))
                trace = node.get("stack_trace")
                if trace:
                    stack_traces.add(trace)
        record["module_paths"] = sorted(module_paths)
        record["stack_trace_count"] = len(stack_traces)
        return record

    nodes = {}
    for graph in aot.get("capture", {}).get("graphs", []):
        for node in graph.get("nodes", []):
            name = node.get("name")
            if name:
                nodes.setdefault(name, []).append(node)

    bindings = {}
    for row in bridge.get("rows", []):
        source_path = row.get("source_path")
        symbol = row.get("symbol")
        if not source_path or not symbol:
            continue
        key = (source_path, symbol)
        campaign_rows = campaigns.get((row.get("candidate_region_id"), symbol), [])
        for campaign_row in campaign_rows:
            bindings.setdefault(key, []).append(campaign_record(
                campaign_row,
                binding_status=row.get("status"),
                binding_method=row.get("method"),
                aot_node_ids=row.get("aot_node_ids", []),
            ))

    # Some historical releases predate candidate_fb_bridge.json.gz but their
    # campaign rows already carry the exact generated source path and symbol.
    # This establishes source-to-program semantics, though not the stronger
    # candidate-to-AOT-node identity supplied by the bridge.
    for row in campaign.get("rows", []):
        source_path = row.get("source_path")
        symbol = row.get("symbol")
        if not source_path or not symbol:
            continue
        campaign_source_bindings.setdefault((source_path, symbol), []).append(
            campaign_record(
                row,
                binding_status=row.get("status"),
                binding_method="CAMPAIGN_SOURCE_PATH_AND_SYMBOL",
                aot_node_ids=[],
            )
        )

    fallback_bindings = {}
    for row in campaign.get("rows", []):
        phase = str(row.get("phase", "")).lower()
        symbol = row.get("symbol")
        if phase in {"forward", "backward"} and symbol:
            fallback_bindings.setdefault((phase, symbol), []).append(campaign_record(
                row,
                binding_status="CAMPAIGN_SYMBOL_PHASE_ONLY",
                binding_method="UNIQUE_CAMPAIGN_PHASE_SYMBOL_FOR_REVIEW_ONLY",
                aot_node_ids=[],
            ))

    return {
        "bindings": bindings,
        "campaign_source_bindings": campaign_source_bindings,
        "fallback_bindings": fallback_bindings,
        "provenance_sha256": {
                "candidate_fb_bridge": (
                    hashlib.sha256(bridge_raw).hexdigest()
                    if bridge_raw is not None else None
                ),
            "campaign": hashlib.sha256(campaign_raw).hexdigest(),
            "default_aot_capture": (
                hashlib.sha256(aot_raw).hexdigest() if aot_raw is not None else None
            ),
        },
        "aot_metadata_status": (
            "AVAILABLE" if aot_path is not None else "NOT_PRESENT_IN_RELEASE"
        ),
        "semantic_metadata_status": (
            "AVAILABLE" if bridge_raw is not None
            else "CAMPAIGN_AVAILABLE_EXACT_SOURCE_BRIDGE_MISSING"
        ),
        "missing_required_provenance": (
            [] if bridge_raw is not None else [str(bridge_path)]
        ),
    }


def enrich(priority):
    releases = {}
    clusters = []
    binding_counts = Counter()
    for cluster in priority["clusters"]:
        members = []
        for member in cluster["members"]:
            source = Path(member["source"]).resolve()
            context = trace_context(source)
            if context is None:
                status = "NOT_A_RUNTIME_RELEASE_TRACE"
                binding_counts[status] += 1
                members.append({
                    **member,
                    "release_root": None,
                    "trace_source": None,
                    "semantic_binding_status": status,
                    "compiler_semantics": [],
                })
                continue
            root, relative = context
            if root not in releases:
                releases[root] = load_release_semantics(root)
            bindings = releases[root]["bindings"].get((relative, member["symbol"]), [])
            if bindings:
                status = "COMPILER_SEMANTICS_ATTACHED"
            else:
                bindings = releases[root]["campaign_source_bindings"].get(
                    (relative, member["symbol"]), [])
                if bindings:
                    status = "COMPILER_SEMANTICS_ATTACHED_FROM_CAMPAIGN_SOURCE_PATH"
                else:
                    status = None
            if status is None:
                phase = ("backward" if "backward" in relative.lower()
                         else "forward" if "forward" in relative.lower() else None)
                fallback = releases[root]["fallback_bindings"].get(
                    (phase, member["symbol"]), []) if phase else []
                if len(fallback) == 1:
                    bindings = fallback
                    status = "REVIEW_SEMANTICS_ATTACHED_WITHOUT_EXACT_SOURCE_BRIDGE"
                elif len(fallback) > 1:
                    status = "AMBIGUOUS_CAMPAIGN_SYMBOL_PHASE"
                else:
                    status = "NO_COMPILER_BINDING_FOUND"
            binding_counts[status] += 1
            members.append({
                **member,
                "release_root": str(root),
                "trace_source": relative,
                "semantic_binding_status": status,
                "compiler_semantics": bindings,
            })
        all_bindings = [binding for member in members for binding in member["compiler_semantics"]]
        clusters.append({
            **cluster,
            "members": members,
            "compiler_original_aten": sorted({
                value for binding in all_bindings for value in binding["original_aten"]
            }),
            "compiler_module_paths": sorted({
                value for binding in all_bindings for value in binding["module_paths"]
            }),
            "semantic_review_status": "REQUIRES_HUMAN_FAMILY_REVIEW",
        })
    return {
        "schema": "uncovered-triton-semantic-review-queue-v1",
        "selection_uses_numerical_results": False,
        "scope": priority["scope"],
        "warning": (
            "Compiler-carried operations and module paths aid review but do not by "
            "themselves establish one mathematical operator family."
        ),
        "uncovered_definition_count": priority["uncovered_definition_count"],
        "structural_cluster_count": priority["structural_cluster_count"],
        "semantic_binding_counts": dict(binding_counts),
        "release_provenance_sha256": {
            str(root): data["provenance_sha256"] for root, data in releases.items()
        },
        "release_aot_metadata_status": {
            str(root): data["aot_metadata_status"] for root, data in releases.items()
        },
        "release_semantic_metadata_status": {
            str(root): data["semantic_metadata_status"] for root, data in releases.items()
        },
        "release_missing_required_provenance": {
            str(root): data["missing_required_provenance"]
            for root, data in releases.items()
            if data["missing_required_provenance"]
        },
        "clusters": clusters,
    }


def compact(result):
    """Remove repeated compiler records while preserving auditable coverage.

    The cluster already stores the union of compiler operations and module
    paths.  Keeping the full compiler record again under every source member
    makes an all-release review file hundreds of megabytes without adding a
    new coverage state.  Source identities, binding outcomes, and release-level
    provenance remain in the compact document.
    """
    clusters = []
    for cluster in result["clusters"]:
        status_counts = Counter(
            member["semantic_binding_status"] for member in cluster["members"]
        )
        members = [
            {
                "source": member["source"],
                "symbol": member["symbol"],
                "release_root": member["release_root"],
                "trace_source": member["trace_source"],
                "semantic_binding_status": member["semantic_binding_status"],
            }
            for member in cluster["members"]
        ]
        clusters.append({
            **{key: value for key, value in cluster.items() if key != "members"},
            "member_binding_status_counts": dict(status_counts),
            "members": members,
        })
    return {
        **result,
        "schema": "uncovered-triton-semantic-review-queue-compact-v1",
        "representation": (
            "CLUSTER_LEVEL_COMPILER_SEMANTICS_WITH_MEMBER_SOURCE_AND_BINDING_STATUS"
        ),
        "clusters": clusters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compact", action="store_true",
        help="Store compiler semantics once per cluster instead of per member.",
    )
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.priority.read_bytes()
    priority = json.loads(raw)
    if priority.get("schema") != "uncovered-triton-structural-priority-v1":
        raise ValueError("Unsupported priority input")
    result = enrich(priority)
    if args.compact:
        result = compact(result)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        "clusters": result["structural_cluster_count"],
        "definitions": result["uncovered_definition_count"],
        "semantic_binding_counts": result["semantic_binding_counts"],
    }))


if __name__ == "__main__":
    main()
