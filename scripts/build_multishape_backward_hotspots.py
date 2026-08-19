#!/usr/bin/env python3
"""Inventory training-semantic backward bottlenecks for all 4×3 model cells."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from build_cross_model_backward_hotspots import classify, load


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen", "phi4", "deepseek8b", "mamba")
SHAPES = (64, 128, 256)


def capture_path(model: str, sequence_length: int, release: Path) -> Path:
    candidates = []
    if model == "qwen":
        candidates.append(ROOT / f"results/coverage/standard_aot/qwen_seq{sequence_length}_capture.json.gz")
    candidates.extend((release / "default_aot_capture.json.gz",
                       release / "default_aot_capture_raw.json.gz"))
    return next(path for path in candidates if path.exists())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_hotspots.json")
    args = parser.parse_args()
    rows = []
    for model in MODELS:
        for sequence_length in SHAPES:
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence_length}_r1"
            capture = load(capture_path(model, sequence_length, release))["capture"]
            nodes = {
                f'{str(node["phase"]).lower()}:graph{graph["graph_index"]}:{node["name"]}': node
                for graph in capture["graphs"] for node in graph["nodes"]
            }
            bridge_path = release / "candidate_fb_bridge_v2.json.gz"
            if not bridge_path.exists():
                bridge_path = release / "candidate_fb_bridge.json.gz"
            bridge = load(bridge_path); plan = load(release / "same_dtype_tasks.json.gz")
            regions = {str(row["candidate_region_id"]): row for row in bridge["rows"]}
            for task in plan["rows"]:
                if str(task["phase"]).upper() != "BACKWARD":
                    continue
                region = regions.get(str(task["candidate_region_id"]));
                if region is None:
                    continue
                semantic_nodes = [nodes[node_id] for node_id in region.get("aot_node_ids", [])
                                  if node_id in nodes and nodes[node_id].get("op") == "call_function"]
                internal_candidates = []
                region_node_ids = set(region.get("aot_node_ids", []))
                for origin in task.get("compiler_origin_rows", []):
                    name = origin.get("exact_origin_node")
                    if not name:
                        continue
                    phase = str(origin.get("phase", task["phase"])).lower()
                    match = re.match(r"ka_[fb]_g\d+_\d+_(.+)$", str(name))
                    original_name = match.group(1) if match else str(name)
                    # The normalized capture uses graph0 for the original
                    # monolithic AOT graph.  Match by exact original node name
                    # against the already compiler-bound region, never by
                    # tensor shape or runtime ordinal.
                    matches = sorted(node_id for node_id in region_node_ids
                                     if node_id.startswith(f"{phase}:")
                                     and (
                                         node_id.rsplit(":", 1)[-1] == original_name
                                         or node_id.rsplit(":", 1)[-1].split("__", 1)[-1] == original_name
                                     ))
                    if not matches:
                        matches = sorted(node_id for node_id in region_node_ids
                                         if node_id.endswith(f"__{original_name}"))
                    internal_candidates.extend(matches)
                family, priority, operations, module_paths = classify(semantic_nodes, model)
                if family is None:
                    continue
                rows.append({
                    "model": model, "sequence_length": sequence_length,
                    "family": family, "priority": priority,
                    "task_id": task["task_id"], "candidate_region_id": task["candidate_region_id"],
                    "exact_aot_endpoint_id": task.get("exact_aot_endpoint_id"),
                    "exact_endpoint_executable": task.get("exact_aot_endpoint_id") is not None,
                    # A missing leaf endpoint does not make the generated semantic
                    # region unobservable.  In particular, Phi lm_head dX is an
                    # exact compiler-bound region whose public task output is a
                    # view of the internal MM.  Keep those regions in the F+B
                    # denominator; leaf-only screening silently dropped a known
                    # positive sensitivity control.
                    "semantic_region_executable": region.get("status") == "BOUND_TO_PROVED_FB_MATHEMATICS",
                    "region_source_path": region.get("source_path"),
                    "region_source_line": region.get("source_line"),
                    "region_symbol": region.get("symbol"),
                    "region_aot_node_ids": region.get("aot_node_ids", []),
                    "internal_aot_endpoint_candidates": sorted(set(internal_candidates)),
                    "implementation_kind": task["implementation_kind"],
                    "semantic_operations": sorted(set(operations)),
                    "module_paths": module_paths,
                    "layer_indices": sorted({int(value) for path in module_paths
                                             for value in re.findall(
                                                 r"(?:layers|layer|blocks|h)\.(\d+)", path)}),
                })
            del capture, nodes, bridge, plan, regions
    counts = collections.Counter((row["model"], row["sequence_length"], row["family"])
                                 for row in rows)
    result = {
        "schema": "kernel-analyzer-multishape-backward-hotspot-inventory-v1",
        "status": "COMPLETE_4_MODELS_X_3_SHAPES",
        "row_count": len(rows),
        "executable_count": sum(row["exact_endpoint_executable"] for row in rows),
        "semantic_region_executable_count": sum(row["semantic_region_executable"] for row in rows),
        "selection": "AOT_SEMANTICS_AND_MODULE_TOPOLOGY_ONLY",
        "excluded_selection_inputs": ["T1", "T2", "T3", "T4", "SEUP", "ERROR_MAGNITUDE"],
        "counts": {f"{model}:seq{shape}:{family}": count
                   for (model, shape, family), count in sorted(counts.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "executable": result["executable_count"]}))


if __name__ == "__main__":
    main()
