#!/usr/bin/env python3
"""Bind all 4×3 semantic cells to nearest exact AOT parameter carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bind_backward_rescreen_carriers import (
    MODELS,
    bind_forward_parameters,
    endpoint_reachability,
    load,
)
from build_multishape_backward_hotspots import capture_path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_equivalence.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_carriers.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    bound = []
    summaries = {}
    for model, model_path in MODELS.items():
        for sequence_length in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence_length}_r1"
            capture = load(capture_path(model, sequence_length, release)).get("capture")
            forward = next(row for row in capture["graphs"] if row["phase"] == "FORWARD")
            parameter_rows, by_primal = bind_forward_parameters(forward, model_path)
            cells = [row for row in source["cells"] if row["model"] == model
                     and row["sequence_length"] == sequence_length]
            graph_nodes = {
                f'{str(node["phase"]).lower()}:graph{graph["graph_index"]}:{node["name"]}': node
                for graph in capture["graphs"] for node in graph["nodes"]
            }
            cell_endpoints = {}
            for row in cells:
                representative = row["representative"]
                exact = representative.get("exact_aot_endpoint_id")
                if exact is not None:
                    cell_endpoints[row["cell_id"]] = [str(exact)]
                    continue
                region_ids = set(representative.get("region_aot_node_ids", []))
                boundary = []
                for node_id in region_ids:
                    node = graph_nodes.get(node_id)
                    if node is None:
                        continue
                    prefix = node_id.rsplit(":", 1)[0]
                    if any(f"{prefix}:{user}" not in region_ids for user in node.get("users", [])):
                        boundary.append(node_id)
                cell_endpoints[row["cell_id"]] = sorted(set(boundary))
            endpoints = {endpoint for values in cell_endpoints.values() for endpoint in values}
            reach = endpoint_reachability(capture, by_primal, endpoints)
            for row in cells:
                candidates = cell_endpoints[row["cell_id"]]
                reached = [
                    (endpoint, parameter)
                    for endpoint in candidates for parameter in reach[endpoint]["parameters"]
                ]
                reached.sort(key=lambda item: (item[1]["aot_distance"], item[1]["name"], item[0]))
                entry = dict(row)
                entry["capture_aot_boundary_endpoints"] = candidates
                entry["capture_boundary_cardinality"] = len(candidates)
                entry["reachability"] = {
                    "status": "EXACT_AOT_PARAMETER_REACHABILITY" if reached
                              else "UNRESOLVED_NO_PARAMETER_REACH",
                    "per_endpoint": {endpoint: reach[endpoint] for endpoint in candidates},
                }
                entry["nearest_carrier"] = reached[0][1] if reached else None
                bound.append(entry)
            summaries[f"{model}:seq{sequence_length}"] = {
                "cells": len(cells),
                "cells_with_carrier": sum(any(
                    reach[endpoint]["status"] == "EXACT_AOT_PARAMETER_REACHABILITY"
                    for endpoint in cell_endpoints[row["cell_id"]]
                ) for row in cells),
                "single_output_semantic_regions": sum(
                    row["capture_boundary"] == "COMPILER_BOUND_SEMANTIC_REGION"
                    and len(cell_endpoints[row["cell_id"]]) == 1 for row in cells),
                "multi_output_semantic_regions": sum(
                    row["capture_boundary"] == "COMPILER_BOUND_SEMANTIC_REGION"
                    and len(cell_endpoints[row["cell_id"]]) > 1 for row in cells),
                "parameter_placeholders": len(parameter_rows),
                "unresolved_parameter_bindings": sum(
                    row["status"] != "EXACT_MODULE_STACK_PARAMETER_BINDING" for row in parameter_rows),
            }
            del capture, forward, parameter_rows, by_primal, reach
    result = {
        "schema": "kernel-analyzer-multishape-backward-carriers-v1",
        "status": "BOUND" if all(row["nearest_carrier"] for row in bound) else "PARTIAL",
        "cell_count": len(bound),
        "cells_with_carrier": sum(row["nearest_carrier"] is not None for row in bound),
        "selection_uses_candidate_values_or_historical_verdict": False,
        "summaries": summaries, "cells": bound,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "cells": len(bound), "bound": result["cells_with_carrier"]}))


if __name__ == "__main__":
    main()
