#!/usr/bin/env python3
"""Bind a forward endpoint through a unique recorded parameter input.

This is for forward-only segmented regions whose old capture lacks a complete
forward-to-backward path.  It uses the existing module-stack parameter binding
and the exact AOT input edge.  It does not claim nonzero gradient influence;
the runtime three-stage measurement must establish that separately.
"""

import argparse
import re
from pathlib import Path

from scripts.bind_backward_rescreen_carriers import bind_forward_parameters
from scripts.run_numerical_coverage import read, save, sha


def _endpoint_node(forward, endpoint):
    match = re.fullmatch(r"forward:graph(\d+):(.+)", endpoint)
    if not match:
        raise ValueError("Explicit forward graph endpoint required")
    graph_index, name = int(match.group(1)), match.group(2)
    direct = [node for node in forward["nodes"] if node["name"] == name]
    if len(direct) == 1:
        return direct[0]
    by_origin = [
        node for node in forward["nodes"]
        if int((node.get("segmented_origin") or {}).get("graph_index", -1)) == graph_index
        and str((node.get("segmented_origin") or {}).get("original_name", "")) == name
    ]
    if len(by_origin) != 1:
        raise ValueError("Unique endpoint node required: " + endpoint)
    return by_origin[0]


def map_direct_operands(forward, by_primal, endpoints):
    rows = []
    for endpoint in sorted(endpoints):
        node = _endpoint_node(forward, endpoint)
        candidates = {}
        for edge in node.get("input_edges", []):
            binding = by_primal.get(str(edge.get("source_node")))
            if binding is not None:
                candidates[binding["name"]] = {
                    "name": binding["name"],
                    "aliases": binding["aliases"],
                    "shape": binding["shape"],
                    "aot_distance": 1,
                    "parameter_operand_node": edge["source_node"],
                    "binding_scope": "UNIQUE_DIRECT_PARAMETER_OPERAND_NOT_GRADIENT_REACH_PROOF",
                }
        parameters = list(candidates.values()) if len(candidates) == 1 else []
        rows.append({
            "endpoint": endpoint,
            "parameters": parameters,
            "status": (
                "UNIQUE_DIRECT_PARAMETER_OPERAND"
                if parameters else "UNRESOLVED_DIRECT_PARAMETER_OPERAND"
            ),
            "candidate_parameter_names": sorted(candidates),
        })
    return {
        "rows": rows,
        "unresolved_identity": [],
        "runtime_measurement_complete": False,
        "claim_scope": (
            "DIRECT_PARAMETER_OPERAND_ONLY; NOT_NONZERO_INFLUENCE_OR_COMPLETE_DERIVATIVE"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "tasks", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    outer = read(args.capture)
    capture = outer.get("capture", outer)
    forward = next(graph for graph in capture["graphs"] if graph["phase"] == "FORWARD")
    tasks = [
        task for task in read(args.tasks)["rows"]
        if task.get("status") == "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT"
        and str(task.get("exact_aot_endpoint_id", "")).startswith("forward:")
    ]
    bindings, by_primal = bind_forward_parameters(forward, args.model)
    result = map_direct_operands(
        forward, by_primal, {task["exact_aot_endpoint_id"] for task in tasks}
    )
    result.update({
        "schema": "forward-parameter-paths-v1",
        "mapping_method": "UNIQUE_DIRECT_PARAMETER_OPERAND_V1",
        "tasks": tasks,
        "parameter_binding_records": bindings,
        "numerical_results_read": False,
    })
    dependencies = [args.capture, args.tasks, args.model / "config.json", Path(__file__),
                    Path(__file__).with_name("bind_backward_rescreen_carriers.py")]
    result["source_sha256"] = {str(path.resolve()): sha(path) for path in dependencies}
    save(args.output, result)
    print({
        "endpoints": len(result["rows"]),
        "mapped": sum(bool(row["parameters"]) for row in result["rows"]),
        "mapping_method": result["mapping_method"],
    })


if __name__ == "__main__":
    main()
