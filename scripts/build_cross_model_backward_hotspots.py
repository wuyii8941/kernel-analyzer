#!/usr/bin/env python3
"""Inventory executable backward semantic bottlenecks across four models.

Selection uses compiler-carried AOT semantics and module topology only.  It
does not read T1/T3/T4, SEUP, error magnitude, or historical verdicts.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    "qwen": (
        ROOT / "results/coverage/runtime_releases/qwen_seq64_r1",
        ROOT / "results/coverage/standard_aot/qwen_seq64_capture.json.gz",
    ),
    "phi4": (
        ROOT / "results/coverage/runtime_releases/phi4_seq64_r1",
        ROOT / "results/coverage/runtime_releases/phi4_seq64_r1/default_aot_capture.json.gz",
    ),
    "deepseek8b": (
        ROOT / "results/coverage/runtime_releases/deepseek8b_seq64_r1",
        ROOT / "results/coverage/runtime_releases/deepseek8b_seq64_r1/default_aot_capture_raw.json.gz",
    ),
    "mamba": (
        ROOT / "results/coverage/runtime_releases/mamba_seq64_r1",
        ROOT / "results/coverage/runtime_releases/mamba_seq64_r1/default_aot_capture.json.gz",
    ),
}


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def classify(nodes: list[dict[str, Any]], model: str) -> tuple[str | None, int, list[str], list[str]]:
    originals = [str(node.get("original_aten", node.get("target", ""))).lower() for node in nodes]
    paths = []
    for node in nodes:
        stack = node.get("nn_module_stack") or node.get("fwd_nn_module_stack") or {}
        paths.extend(str(value[0]).lower() for value in stack.values()
                     if isinstance(value, (list, tuple)) and value)
    paths = sorted(set(paths))
    joined = " ".join(paths); ops = " ".join(originals)
    if any(key in ops for key in ("nll_loss", "log_softmax", "cross_entropy")):
        return "LOSS_CE_BACKWARD", 100, originals, paths
    if "lm_head" in joined or "embed_out" in joined:
        return "LOSS_HEAD_BACKWARD", 95, originals, paths
    if any(key in joined for key in ("layernorm", "rmsnorm", "input_layernorm",
                                     "post_attention_layernorm", "q_norm", "k_norm", ".norm")):
        return "NORMALIZATION_BACKWARD", 90, originals, paths
    if "self_attn" in joined or "attention" in joined:
        if any(key in ops for key in ("softmax", "bmm", "scaled_dot_product")):
            return "ATTENTION_STATE_OR_TRANSPORT_BACKWARD", 90, originals, paths
        return "ATTENTION_PROJECTION_BACKWARD", 75, originals, paths
    if model == "mamba" and any(key in joined + " " + ops for key in (
        "mixer", "selective", "scan", "ssm", "conv1d", "state")):
        return "STATE_SPACE_RECURRENT_BACKWARD", 95, originals, paths
    return None, 0, originals, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_formation/hotspot_search/cross_model_backward_hotspots.json")
    args = parser.parse_args()
    rows = []
    for model, (release, capture_path) in CELLS.items():
        capture = load(capture_path)["capture"]
        nodes = {
            f'{str(node["phase"]).lower()}:graph{graph["graph_index"]}:{node["name"]}': node
            for graph in capture["graphs"] for node in graph["nodes"]
        }
        bridge_path = release / "candidate_fb_bridge_v2.json.gz"
        if not bridge_path.exists():
            bridge_path = release / "candidate_fb_bridge.json.gz"
        bridge = load(bridge_path)
        plan = load(release / "same_dtype_tasks.json.gz")
        regions = {str(row["candidate_region_id"]): row for row in bridge["rows"]}
        for task in plan["rows"]:
            if str(task["phase"]).upper() != "BACKWARD":
                continue
            region = regions.get(str(task["candidate_region_id"]))
            if region is None:
                continue
            semantic_nodes = [nodes[node_id] for node_id in region.get("aot_node_ids", [])
                              if node_id in nodes and nodes[node_id].get("op") == "call_function"]
            family, score, operations, module_paths = classify(semantic_nodes, model)
            if family is None:
                continue
            endpoint = task.get("exact_aot_endpoint_id")
            rows.append({
                "model": model, "sequence_length": 64, "family": family,
                "priority": score, "task_id": task["task_id"],
                "candidate_region_id": task["candidate_region_id"],
                "exact_aot_endpoint_id": endpoint,
                "implementation_kind": task["implementation_kind"],
                "semantic_operations": sorted(set(operations)),
                "module_paths": module_paths,
                "layer_indices": sorted({
                    int(value) for path in module_paths
                    for value in re.findall(r"(?:layers|layer|blocks|h)\.(\d+)", path)
                }),
                "exact_endpoint_executable": endpoint is not None,
                "formation_capture_status": (
                    "NEEDS_PARAMETER_CARRIER_BINDING" if endpoint is not None
                    else "BLOCKED_NO_EXACT_ENDPOINT"
                ),
                "selection_uses_historical_verdict": False,
            })
        del capture, nodes, bridge, plan, regions
    rows.sort(key=lambda row: (-int(row["priority"]), row["model"], str(row["task_id"])))
    counts = collections.Counter((row["model"], row["family"]) for row in rows)
    output = {
        "schema": "kernel-analyzer-cross-model-backward-hotspot-inventory-v1",
        "status": "SEMANTIC_ROSTER_READY_FOR_CARRIER_BINDING",
        "selection": "AOT_SEMANTICS_AND_MODULE_TOPOLOGY_ONLY",
        "models": list(CELLS), "sequence_length": 64,
        "row_count": len(rows),
        "counts": {f"{model}:{family}": count for (model, family), count in sorted(counts.items())},
        "rows": rows,
        "excluded_selection_inputs": ["T1", "T3", "T4", "SEUP", "ERROR_MAGNITUDE"],
        "next_step": "Bind a reachable declared parameter carrier, then run local-gradient-update formation without requiring local bias.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "counts": output["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
