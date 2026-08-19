#!/usr/bin/env python3
"""Build exact AOT replay cuts for single-output backward semantic regions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from build_multishape_backward_hotspots import capture_path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_carriers.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/semantic_region_capture_plans"


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def cut_task(graph: dict[str, Any], node_ids: list[str], task_id: str,
             expected_graph_code_sha256: str) -> dict[str, Any]:
    nodes = {str(node["name"]): node for node in graph["nodes"]}
    names = [node_id.rsplit(":", 1)[-1] for node_id in node_ids]
    selected = set(names)
    runtime_name = {
        name: str((nodes[name].get("segmented_origin") or {}).get("original_name", name))
        for name in nodes
    }
    inputs = []
    outputs = []
    prefix = f'{str(graph["phase"]).lower()}:graph{graph["graph_index"]}:'
    for name in names:
        node = nodes[name]
        for edge in node.get("input_edges", []):
            source = str(edge["source_node"])
            if source not in selected:
                inputs.append({
                    "source_node": runtime_name[source],
                    "consumer_node_id": prefix + runtime_name[name],
                    "consumer_argument_path": edge["argument_path"],
                })
        for user_name in node.get("users", []):
            if str(user_name) in selected:
                continue
            user = nodes[str(user_name)]
            for edge in user.get("input_edges", []):
                if str(edge["source_node"]) == name:
                    outputs.append({
                        "source_node_id": prefix + runtime_name[name],
                        "consumer_node_id": prefix + runtime_name[str(user_name)],
                        "consumer_argument_path": edge["argument_path"],
                    })
    output_sources = sorted({row["source_node_id"] for row in outputs})
    if len(output_sources) != 1:
        raise ValueError(f"semantic region is not single-output: {task_id}: {output_sources}")
    return {
        "task_id": f"semantic-region-reference:{task_id}",
        "cut_id": f"semantic-region-reference:{task_id}",
        "phase": str(graph["phase"]),
        "graph_index": int(graph["graph_index"]),
        "expected_graph_code_sha256": expected_graph_code_sha256,
        "aot_node_ids": [prefix + runtime_name[name] for name in names],
        "aot_node_names": [runtime_name[name] for name in names],
        "semantic_endpoint_id": f"semantic-region:{task_id}",
        "expected_boundary_inputs": inputs,
        "expected_boundary_outputs": outputs,
        "required_extractor_schema": "forkcert.fx-reference-cut.v2",
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for model in ("qwen", "phi4", "deepseek8b", "mamba"):
        for sequence_length in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence_length}_r1"
            capture = load(capture_path(model, sequence_length, release))["capture"]
            graphs = {(str(g["phase"]), int(g["graph_index"])): g for g in capture["graphs"]}
            same_dtype = load(release / "same_dtype_tasks.json.gz")
            runtime_graph_hash = {}
            for cut in same_dtype["reference_cut_tasks"]:
                runtime_graph_hash[(str(cut["phase"]), int(cut["graph_index"]))] = str(
                    cut["expected_graph_code_sha256"]
                )
            cases = []
            blocked = []
            for cell in source["cells"]:
                if (cell["model"] != model or cell["sequence_length"] != sequence_length
                        or cell["capture_boundary"] != "COMPILER_BOUND_SEMANTIC_REGION"):
                    continue
                representative = cell["representative"]
                internal = list(representative.get("internal_aot_endpoint_candidates", []))
                if len(internal) == 1:
                    node_ids = internal
                    boundary = "COMPILER_BOUND_INTERNAL_AOT_OUTPUT"
                    reference_method = "AOT_REPLAY"
                elif (representative.get("implementation_kind") == "EXTERN"
                      and representative.get("region_symbol") in {"mm", "bmm", "addmm"}):
                    node_ids = []
                    boundary = "COMPILER_BOUND_EXTERNAL_OUTPUT"
                    reference_method = "EXTERNAL_FP32_RECOMPUTE"
                elif (cell["family"] == "NORMALIZATION_BACKWARD"
                      and str(representative.get("region_symbol", "")).startswith(
                          "triton_red_fused_sum_")):
                    node_ids = []
                    boundary = "COMPILER_PARTIAL_REDUCTION_BUFFER"
                    reference_method = "PARTIAL_REDUCTION_FROM_BOUND_INPUT"
                else:
                    blocked.append({
                        "case_id": cell["cell_id"],
                        "task_id": representative["task_id"],
                        "family": cell["family"],
                        "reason": "BLOCKED_NO_EXACT_INTERNAL_PORT_SEMANTICS",
                    })
                    continue
                case = {
                    "case_id": cell["cell_id"],
                    "task_id": representative["task_id"],
                    "family": cell["family"],
                    "carrier": (
                        "model.norm.weight"
                        if model == "phi4"
                        and representative["task_id"] == "backward:497:output_0"
                        else cell["nearest_carrier"]["name"]
                    ),
                    "carrier_binding": (
                        "KNOWN_STRICT_SENSITIVITY_ANCHOR"
                        if model == "phi4"
                        and representative["task_id"] == "backward:497:output_0"
                        else "STATIC_AOT_REACHABILITY_REQUIRES_DYNAMIC_CHECK"
                    ),
                    "capture_boundary": boundary,
                    "reference_method": reference_method,
                }
                if reference_method == "AOT_REPLAY":
                    first = node_ids[0].split(":", 2)
                    graph = graphs[(first[0].upper(), int(first[1].removeprefix("graph")))]
                    case["reference_cut_task"] = cut_task(
                            graph, node_ids, representative["task_id"],
                            runtime_graph_hash[(first[0].upper(), int(first[1].removeprefix("graph")))],
                        )
                cases.append(case)
            payload = {
                "schema": "kernel-analyzer-semantic-region-capture-plan-v1",
                "model": model,
                "sequence_length": sequence_length,
                "complete_fb_unit_required": True,
                "cases": cases,
                "blocked": blocked,
            }
            target = OUTPUT / f"{model}_seq{sequence_length}.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            total += len(cases)
            print(json.dumps({"output": str(target), "cases": len(cases)}))
    print(json.dumps({"single_output_semantic_regions": total}))


if __name__ == "__main__":
    main()
