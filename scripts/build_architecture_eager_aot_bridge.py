#!/usr/bin/env python3
"""Conservatively bind architecture eager invocations to captured AOT nodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.build_architecture_invocation_ledger import align_origin_witness


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def eager_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (
            tuple(str(dim) for dim in value["shape"]),
            value["dtype"],
            tuple(str(stride) for stride in value["stride"]),
        )
        for value in event["output_tensors"]
    )


def aot_signature(node: dict[str, Any]) -> tuple[Any, ...] | None:
    meta = node.get("tensor_meta")
    if (
        isinstance(meta, list) and len(meta) >= 4
        and isinstance(meta[0], list) and isinstance(meta[1], str)
    ):
        return ((tuple(str(value) for value in meta[0]), meta[1], tuple(str(value) for value in meta[3])),)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("qwen", "mamba", "moe", "phi", "deepseek8", "generic"),
        required=True,
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--origin-inventory", type=Path, required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weak = load(args.inventory)
    strong = load(args.origin_inventory)
    aot = load(args.aot)
    if weak["architecture"] != args.architecture or strong["architecture"] != args.architecture:
        raise ValueError("inventory architecture mismatch")
    if aot.get("architecture", "qwen") != args.architecture:
        raise ValueError("AOT architecture mismatch")
    if aot["status"] not in {
        "COMPLETE_AOT_FB_CAPTURE",
        "COMPLETE_EXECUTION_PARTIAL_CROSS_SEGMENT_BRIDGE",
        "COMPLETE_SEGMENT_LOCAL_CAPTURE_WITH_EXPLICIT_CROSS_SEGMENT_BOUNDARIES",
    }:
        raise ValueError("AOT capture invalid")

    weak_events = weak["trace"]["events"]
    strong_events = strong["trace"]["events"]
    origins, extras = align_origin_witness(weak_events, strong_events)

    aot_nodes = {}
    index: dict[tuple[str, int, str, tuple[Any, ...] | None], list[str]] = defaultdict(list)
    for graph in aot["capture"]["graphs"]:
        phase = graph["phase"]
        graph_index = int(graph.get("graph_index", 0))
        for node in graph["nodes"]:
            if node["op"] != "call_function":
                continue
            node_id = f"{phase.lower()}:graph{graph_index}:{node['name']}"
            aot_nodes[node_id] = node
            if node.get("seq_nr") is not None:
                index[(phase, int(node["seq_nr"]), node["target"], aot_signature(node))].append(node_id)

    used_nodes = set()
    rows = []
    for event in weak_events:
        witness = origins[event["invocation_id"]]
        sequence = (
            witness.get("forward_autograd_sequence_nr")
            if event["phase"] == "FORWARD"
            else witness.get("backward_autograd_sequence_nr")
        )
        candidates = [] if sequence is None else index.get(
            (event["phase"], int(sequence), event["overload"], eager_signature(event)), []
        )
        candidates = [node_id for node_id in candidates if node_id not in used_nodes]
        if len(candidates) == 1:
            status = "EXACT_AUTOGRAD_SEQUENCE_TARGET_AND_OUTPUT_SIGNATURE"
            node_ids = candidates
            used_nodes.add(candidates[0])
        else:
            status = (
                "UNRESOLVED_NO_AUTOGRAD_SEQUENCE"
                if sequence is None
                else "UNRESOLVED_NONUNIQUE_OR_ABSENT_EXACT_AOT_NODE"
            )
            node_ids = []
        row = {
            "invocation_id": event["invocation_id"],
            "phase": event["phase"],
            "overload": event["overload"],
            "autograd_sequence_nr": sequence,
            "status": status,
            "aot_node_ids": node_ids,
            "name_shape_or_ordinal_pairing_used": False,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    counts = Counter(row["status"] for row in rows)
    exact = counts["EXACT_AUTOGRAD_SEQUENCE_TARGET_AND_OUTPUT_SIGNATURE"]
    payload = {
        "schema": "kernel-analyzer-architecture-eager-aot-bridge-v1",
        "status": "COMPLETE_EXACT_BRIDGE" if exact == len(rows) else "PARTIAL_FAIL_CLOSED",
        "architecture": args.architecture,
        "denominator": {
            "eager_invocations": len(rows),
            "exact_eager_aot_bindings": exact,
            "unresolved_eager_invocations": len(rows) - exact,
            "aot_call_function_nodes": len(aot_nodes),
            "aot_nodes_used_by_exact_bindings": len(used_nodes),
            "observer_induced_strong_detaches_excluded": len(extras),
        },
        "status_counts": dict(sorted(counts.items())),
        "unresolved_overload_counts": dict(sorted(Counter(
            row["overload"] for row in rows if row["status"].startswith("UNRESOLVED")
        ).items())),
        "gates": {
            "all_eager_invocations_retained": len(rows) == len(weak_events),
            "weak_strong_origin_witness_exact": len(origins) == len(weak_events),
            "name_shape_or_ordinal_pairing_used": False,
            "all_eager_aot_bindings_exact": exact == len(rows),
        },
        "bindings": {
            "inventory_result_sha256": weak["result_sha256"],
            "origin_inventory_result_sha256": strong["result_sha256"],
            "aot_result_sha256": aot["result_sha256"],
        },
        "rows": rows,
        "claim_boundary": (
            "Only unique sequence-number + target + full output-signature matches are exact. "
            "Nested dispatch, elisions, functionalization and compiler rewrites remain unresolved."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=9) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "denominator": payload["denominator"], "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
