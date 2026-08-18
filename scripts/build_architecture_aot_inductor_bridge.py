#!/usr/bin/env python3
"""Build an exact-name-and-descriptor AOT-to-Inductor proof-ID bridge.

This generic bridge is intentionally conservative.  Compiler-added or
rewritten nodes remain explicit; architecture-specific semantic rewrite rules
may close them later, but cardinality or ordinal similarity never does.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_target(value: str) -> str:
    return value.replace("torch.ops.", "").replace(".default.default", ".default")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("qwen", "mamba", "moe"), required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--inductor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aot = load(args.aot)
    candidate = load(args.inductor)
    if aot.get("architecture", "qwen") != args.architecture:
        raise ValueError("AOT architecture mismatch")
    if candidate.get("architecture", "qwen") != args.architecture:
        raise ValueError("candidate architecture mismatch")
    if not aot.get("preserve_aot_aten"):
        raise ValueError("the mathematical AOT input must use --preserve-aot-aten")
    if aot["status"] != "COMPLETE_AOT_FB_CAPTURE":
        raise ValueError("AOT capture is invalid")
    if candidate["status"] != "COMPLETE_PROOF_ID_PROPAGATION_CAPTURE":
        raise ValueError("candidate proof capture is invalid")

    aot_nodes = {}
    for graph in aot["capture"]["graphs"]:
        graph_index = int(graph.get("graph_index", 0))
        for node in graph["nodes"]:
            if node["op"] != "call_function":
                continue
            node_id = f"{graph['phase'].lower()}:graph{graph_index}:{node['name']}"
            aot_nodes[node_id] = node
    candidate_nodes = {}
    for graph in candidate["proof_graphs"]:
        tag_to_id = {row["tagged_fx_name"]: row["proof_id"] for row in graph["rows"]}
        for row in graph["rows"]:
            candidate_nodes[row["proof_id"]] = {
                **row,
                "phase": graph["phase"],
                "input_proof_ids": sorted(
                    tag_to_id[name]
                    for name in row["input_tagged_fx_names"]
                    if name in tag_to_id
                ),
            }
    rows = []
    for node_id, node in sorted(aot_nodes.items()):
        other = candidate_nodes.get(node_id)
        if other is None:
            status = "UNRESOLVED_AOT_NODE_ABSENT_AFTER_COMPILER_REWRITE"
            checks = None
        else:
            phase = node_id.split(":", 1)[0]
            expected_inputs = sorted(
                f"{phase}:graph0:{name}" for name in node["input_nodes"]
                if f"{phase}:graph0:{name}" in aot_nodes
            )
            checks = {
                "proof_id_exact": True,
                "target_exact": normalize_target(str(other["target"])) == normalize_target(str(node["target"])),
                "call_function_input_edges_exact": other["input_proof_ids"] == expected_inputs,
            }
            status = (
                "EXACT_COMPILER_CARRIED_PROOF_ID_AND_TARGET"
                if all(checks.values())
                else "UNRESOLVED_PROOF_ID_TARGET_CHANGED"
            )
        row = {
            "aot_node_id": node_id,
            "aot_target": node["target"],
            "candidate_proof_id": node_id if other is not None else None,
            "candidate_target": other["target"] if other is not None else None,
            "status": status,
            "checks": checks,
            "ordinal_or_shape_pairing_used": False,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    exact_ids = {
        row["candidate_proof_id"] for row in rows
        if row["status"] == "EXACT_COMPILER_CARRIED_PROOF_ID_AND_TARGET"
    }
    added = sorted(set(candidate_nodes) - exact_ids)
    counts = Counter(row["status"] for row in rows)
    unresolved = len(rows) - counts["EXACT_COMPILER_CARRIED_PROOF_ID_AND_TARGET"]
    payload = {
        "schema": "kernel-analyzer-architecture-aot-inductor-bridge-v1",
        "status": "COMPLETE_EXACT_BRIDGE" if not unresolved and not added else "PARTIAL_FAIL_CLOSED",
        "architecture": args.architecture,
        "candidate_preserve_aot_aten": bool(candidate.get("preserve_aot_aten")),
        "denominator": {
            "aot_call_function_nodes": len(aot_nodes),
            "candidate_proof_nodes": len(candidate_nodes),
            "exact_aot_nodes": counts["EXACT_COMPILER_CARRIED_PROOF_ID_AND_TARGET"],
            "unresolved_aot_nodes": unresolved,
            "candidate_added_or_rewritten_nodes": len(added),
        },
        "gates": {
            "candidate_repeat_stable": bool(candidate["repeat_stable"]),
            "all_candidate_proof_tags_observed": candidate["proof_tag_summary"]["tags_not_observed"] == 0,
            "all_aot_nodes_exact": unresolved == 0,
            "all_candidate_added_nodes_semantically_closed": not added,
            "ordinal_name_similarity_or_shape_pairing_used": False,
        },
        "bindings": {
            "aot_result_sha256": aot["result_sha256"],
            "candidate_result_sha256": candidate["result_sha256"],
        },
        "status_counts": dict(sorted(counts.items())),
        "candidate_added_or_rewritten_node_ids": added,
        "rows": rows,
        "claim_boundary": (
            "Exact rows bind AOT nodes to compiler-carried proof IDs. Candidate-added or target-changed "
            "nodes remain unresolved until an explicit semantic rewrite proof is supplied."
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
