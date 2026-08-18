#!/usr/bin/env python3
"""Audit compiler-carried candidate roots against actual AOT VJP graphs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
AOT_PREFIX = {
    "qwen3_1p7b": "qwen", "mamba_130m": "mamba",
    "phi4_mini_3p8b": "phi4", "deepseek_r1_0528_qwen3_8b": "deepseek8b",
}
RELEASE_PREFIX = AOT_PREFIX
COMMENT = re.compile(
    r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]"
)


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def flatten_nodes(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if set(value) == {"node"}:
            yield str(value["node"])
        else:
            for item in value.values():
                yield from flatten_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_nodes(item)


def ancestors(name: str, nodes: dict[str, dict[str, Any]]) -> set[str]:
    found = set()
    stack = [name]
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        stack.extend(nodes.get(current, {}).get("input_nodes", ()))
    return found


def reaches_output(name: str, nodes: dict[str, dict[str, Any]], outputs: set[str]) -> bool:
    seen = set(); stack = [name]
    while stack:
        current = stack.pop()
        if current in outputs:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(nodes.get(current, {}).get("users", ()))
    return False


def expected_target(function: str) -> str:
    if function.endswith(".mm"):
        return "aten.mm.default"
    if function.endswith(".bmm"):
        return "aten.bmm.default"
    if function == "torch.ops.aten.convolution_backward.default":
        return "aten.convolution_backward.default"
    return "UNRESOLVED"


def tensor_shape(node: dict[str, Any] | None) -> tuple[Any, ...] | None:
    metadata = (node or {}).get("tensor_meta")
    if isinstance(metadata, list) and metadata and isinstance(metadata[0], list):
        return tuple(metadata[0])
    return None


def source_nodes(path: Path, line: int) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    for index in range(min(line - 1, len(lines) - 1), max(-1, line - 8), -1):
        match = COMMENT.search(lines[index])
        if match:
            return (
                [item.strip() for item in match.group(1).split(",") if item.strip()],
                [item.strip() for item in match.group(2).split(",") if item.strip()],
            )
    return [], []


def main() -> None:
    tasks = load(COVERAGE / "candidate_fb_proof_tasks.json")
    queue_rows = {
        row["candidate_id"]: row for row in load(COVERAGE / "bias_candidate_queue.json")["candidates"]
    }
    cached_aot_path: Path | None = None
    cached_aot: dict[str, Any] | None = None
    rows = []
    for task in tasks["rows"]:
        architecture = task["architecture"]; seq = int(task["sequence_length"])
        prefix = AOT_PREFIX[architecture]
        aot_path = COVERAGE / f"{prefix}_seq{seq}_aot.json.gz"
        if not aot_path.exists():
            rows.append({
                "candidate_id": task["candidate_id"], "status": "UNRESOLVED_MISSING_AOT_CAPTURE",
                "gates": {},
            })
            continue
        if aot_path != cached_aot_path:
            cached_aot = load(aot_path)
            cached_aot_path = aot_path
        assert cached_aot is not None
        aot = cached_aot
        call = queue_rows[task["candidate_id"]]["exact_generated_call"]
        release = COVERAGE / "runtime_releases" / f"{RELEASE_PREFIX[architecture]}_seq{seq}_r1"
        generated = release / "trace" / call["source_path"]
        names, original_aten = source_nodes(generated, int(call["source_line"]))
        phase = str(call["phase"]).upper()
        segment_match = re.search(rf"_{phase.lower()}_segment(\d+)_", call["source_path"])
        segment = int(segment_match.group(1)) if segment_match else 0
        phase_graphs = sorted(
            (graph for graph in aot["capture"]["graphs"] if graph["phase"] == phase),
            key=lambda graph: int(graph["graph_index"]),
        )
        if segment >= len(phase_graphs):
            rows.append({"candidate_id": task["candidate_id"],
                         "status": "UNRESOLVED_AOT_SEGMENT", "gates": {}})
            continue
        graph = phase_graphs[segment]
        graph_nodes = {str(node["name"]): node for node in graph["nodes"]}
        target = expected_target(str(call["function"]))
        roots = [graph_nodes[name] for name in names
                 if name in graph_nodes and graph_nodes[name].get("target") == target]
        root_exact = len(roots) == 1
        root = roots[0] if root_exact else None
        sequence_nr = root.get("seq_nr") if root else None
        all_graphs = aot["capture"]["graphs"]
        forward_nodes = {
            str(node["name"]): node for item in all_graphs if item["phase"] == "FORWARD"
            for node in item["nodes"]
        }
        backward_nodes = {
            str(node["name"]): node for item in all_graphs if item["phase"] == "BACKWARD"
            for node in item["nodes"]
        }
        sequence_forward = [node for node in forward_nodes.values()
                            if node.get("seq_nr") == sequence_nr and node.get("target") in {
                                "aten.mm.default", "aten.bmm.default", "aten.convolution.default"
                            }]
        sequence_backward = [node for node in backward_nodes.values()
                             if node.get("seq_nr") == sequence_nr]
        vjp_matmuls = [node for node in sequence_backward
                       if node.get("target") in {"aten.mm.default", "aten.bmm.default"}]
        transpose_nodes = [node for node in sequence_backward if node.get("target") in {
            "aten.t.default", "aten.transpose.int"
        }]
        tangent_names = {name for name, node in backward_nodes.items()
                         if node.get("op") == "placeholder" and name.startswith("tangents")}
        backward_outputs = set()
        for item in all_graphs:
            if item["phase"] == "BACKWARD":
                for node in item["nodes"]:
                    if node["op"] == "output":
                        backward_outputs.update(flatten_nodes(node["arguments"]))
        forward_saved = set()
        for item in all_graphs:
            if item["phase"] == "FORWARD":
                for node in item["nodes"]:
                    if node["op"] == "output":
                        forward_saved.update(flatten_nodes(node["arguments"]))
        saved_placeholders = forward_saved & {
            name for name, node in backward_nodes.items() if node.get("op") == "placeholder"
        }
        vjp_edges = []
        for node in vjp_matmuls:
            dependency = ancestors(str(node["name"]), backward_nodes)
            vjp_edges.append({
                "node": node["name"],
                "has_cotangent_ancestor": bool(dependency & tangent_names),
                "has_saved_tensor_ancestor": bool(dependency & saved_placeholders),
                "reaches_backward_output": reaches_output(
                    str(node["name"]), backward_nodes, backward_outputs
                ),
            })
        forward_input_shapes = sorted(
            (shape for shape in (
                tensor_shape(forward_nodes.get(name))
                for node in sequence_forward for name in node.get("input_nodes", ())
            ) if shape is not None), key=str,
        )
        vjp_output_shapes = sorted(
            (shape for shape in (tensor_shape(node) for node in vjp_matmuls)
             if shape is not None), key=str,
        )
        checks = {
            "compiler_carried_source_comment_present": bool(names and original_aten),
            "candidate_root_exact": root_exact,
            "sequence_number_present": sequence_nr is not None,
            "unique_forward_root_for_sequence": len(sequence_forward) == 1,
            "saved_tensor_placeholders_present": bool(saved_placeholders),
            "vjp_matmul_edges_present": bool(vjp_edges),
            "vjp_transposes_present": bool(vjp_edges)
            and len(transpose_nodes) >= len(vjp_edges),
            "vjp_output_shapes_match_forward_inputs": bool(vjp_output_shapes)
            and vjp_output_shapes == forward_input_shapes,
            "every_vjp_edge_has_cotangent": bool(vjp_edges) and all(
                row["has_cotangent_ancestor"] for row in vjp_edges
            ),
            "every_vjp_edge_has_saved_tensor": bool(vjp_edges) and all(
                row["has_saved_tensor_ancestor"] for row in vjp_edges
            ),
            "every_vjp_edge_reaches_backward_output": bool(vjp_edges) and all(
                row["reaches_backward_output"] for row in vjp_edges
            ),
        }
        aot_structure_exact = all(checks.values())
        row = {
            "candidate_id": task["candidate_id"], "aot_capture": str(aot_path.relative_to(ROOT)),
            "aot_capture_sha256": aot["result_sha256"], "generated_source_nodes": names,
            "original_aten": original_aten, "root": root["name"] if root else None,
            "sequence_nr": sequence_nr,
            "forward_roots": [node["name"] for node in sequence_forward],
            "vjp_edges": vjp_edges, "saved_placeholders_count": len(saved_placeholders),
            "forward_input_shapes": forward_input_shapes,
            "vjp_output_shapes": vjp_output_shapes,
            "vjp_transpose_nodes": [node["name"] for node in transpose_nodes],
            "gates": checks,
            "status": "EXACT_AOT_VJP_STRUCTURE" if aot_structure_exact else "UNRESOLVED_AOT_VJP_STRUCTURE",
            "claim_boundary": (
                "AOT structural proof does not by itself prove eager origin or candidate numerical correctness."
            ),
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    payload = {
        "schema": "kernel-analyzer-candidate-aot-vjp-audit-v1",
        "status": "PARTIAL_FAIL_CLOSED",
        "task_sha256": tasks["result_sha256"], "rows": rows,
        "counts": {
            "tasks": len(rows),
            "exact_aot_vjp_structure": sum(row["status"] == "EXACT_AOT_VJP_STRUCTURE" for row in rows),
            "unresolved": sum(row["status"] != "EXACT_AOT_VJP_STRUCTURE" for row in rows),
        },
    }
    if payload["counts"]["unresolved"] == 0:
        payload["status"] = "COMPLETE_AOT_VJP_STRUCTURE"
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "candidate_aot_vjp_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), **payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
