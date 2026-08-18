#!/usr/bin/env python3
"""Build exact candidate-buffer to AOT semantic endpoint comparison tasks.

Generated implementation buffers that have no exact AOT ``origin_node`` are
kept as internal members of a closed semantic region.  They are never paired
to an eager tensor by name, shape, ordinal, or numerical similarity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--candidate-fb-bridge", type=Path, required=True)
    parser.add_argument("--proof-capture", type=Path, required=True)
    parser.add_argument("--parameter-gradient-closures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = load(args.inventory)
    bridge = load(args.candidate_fb_bridge)
    proof = load(args.proof_capture)
    parameter_closures = (
        load(args.parameter_gradient_closures)
        if args.parameter_gradient_closures else None
    )
    if parameter_closures is not None:
        if parameter_closures.get("status") != "COMPLETE_TERMINAL_OUTPUTS_BOUND_TO_PARAMETER_GRADIENTS":
            raise RuntimeError("parameter-gradient closure artifact is incomplete")
        if parameter_closures["bindings"]["inventory_result_sha256"] != inventory["result_sha256"]:
            raise RuntimeError("parameter-gradient closure binds another inventory")
        if parameter_closures["bindings"].get("proof_capture_result_sha256") != proof["result_sha256"]:
            raise RuntimeError("parameter-gradient closure binds another proof capture")
    origins = proof.get("inductor_buffer_origins")
    if not origins:
        raise RuntimeError("proof capture lacks compiler-live IR-buffer origins")
    proof_id_aliases = {
        str(key): str(value)
        for key, value in bridge.get("bindings", {}).get(
            "proof_id_aliases", {}
        ).items()
    }
    proof_ids_in_capture = {
        str(row["proof_id"])
        for graph in proof["proof_graphs"] for row in graph["rows"]
    }
    normalized_to_proof: dict[str, str] = {}
    for proof_id in sorted(proof_ids_in_capture):
        normalized = proof_id_aliases.get(proof_id, proof_id)
        previous = normalized_to_proof.get(normalized)
        if previous is not None and previous != proof_id:
            raise RuntimeError(
                "normalized semantic endpoint has multiple proof-graph nodes: "
                f"{normalized}: {previous}, {proof_id}"
            )
        normalized_to_proof[normalized] = proof_id
    tagged_to_proof = {
        str(row["tagged_fx_name"]): proof_id_aliases.get(
            str(row["proof_id"]), str(row["proof_id"])
        )
        for graph in proof["proof_graphs"] for row in graph["rows"]
    }
    region_bridge = {
        str(row["candidate_region_id"]): row for row in bridge["rows"]
    }
    dataflow = {
        str(row["region_id"]): row
        for row in inventory["compute_dataflow"]["rows"]
    }
    if set(region_bridge) != set(dataflow):
        raise RuntimeError("candidate bridge and executed dataflow denominators differ")

    successors: dict[str, set[str]] = defaultdict(set)
    for region_id, row in dataflow.items():
        successors[region_id].update(
            str(edge["consumer_region_id"])
            for edge in row.get("direct_consumer_edges", [])
        )
        for edge in row.get("previous_storage_writers", []):
            successors[str(edge["previous_writer_region_id"])].add(region_id)

    origins_by_phase_buffer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, compiler_rows in origins.get("kernel_buffer_origins", {}).items():
        phase, _kernel, buffer = key.split("\0", 2)
        origins_by_phase_buffer[(phase, buffer)].extend(compiler_rows)

    rows = []
    exact_endpoint_regions: dict[str, list[str]] = defaultdict(list)
    for region_id, flow in dataflow.items():
        binding = flow.get("boundary_witness", {}).get(
            "formal_to_actual_pointer_binding", {}
        )
        bridge_row = region_bridge[region_id]
        allowed = set(map(str, bridge_row["aot_node_ids"]))
        kind = str(flow["kind"])
        compiler_theorem_closed = (
            kind == "DIRECT_TENSOR_METHOD"
            and str(flow["symbol"]) == "copy_"
            and len(bridge_row.get("proof_owners", [])) == 1
            and bridge_row["proof_owners"][0].get("owner_kind")
            == "EXPLICIT_COMPILER_ADDED_THEOREM"
            and bridge_row["proof_owners"][0].get("theorem", {}).get("proof_kind")
            == "EXACT_SEGMENT_BOUNDARY_COPY"
        )
        port_specs: list[tuple[str, str, str, list[dict[str, Any]]]] = []
        for formal, pointer in sorted(binding.items()):
            if pointer.get("stored"):
                variable = str(pointer["tensor_variable"])
                key = f"{str(flow['phase']).upper()}\0{flow['symbol']}\0{variable}"
                port_specs.append((formal, variable, key, list(
                    origins.get("kernel_buffer_origins", {}).get(key, [])
                )))
        if kind == "EXTERN":
            for index, variable_value in enumerate(flow.get("output_tensor_variables", [])):
                variable = str(variable_value)
                key = (
                    f"{str(flow['phase']).upper()}\0"
                    f"extern_kernels.{flow['symbol']}\0{variable}"
                )
                port_specs.append((f"output_{index}", variable, key, list(
                    origins.get("kernel_buffer_origins", {}).get(key, [])
                )))
        elif kind.startswith("DIRECT_"):
            for index, variable_value in enumerate(flow.get("output_tensor_variables", [])):
                variable = str(variable_value)
                key = (
                    f"{str(flow['phase']).upper()}\0{kind}:{flow['symbol']}\0{variable}"
                )
                port_specs.append((
                    (
                        f"output_{index}"
                        if kind == "DIRECT_TORCH_OP"
                        else f"mutated_output_{index}"
                    ), variable, key,
                    list(origins_by_phase_buffer.get(
                        (str(flow["phase"]).upper(), variable), []
                    )),
                ))
        for formal, variable, key, compiler_rows in port_specs:
            exact_tagged = sorted({
                str(row["exact_origin_node"]) for row in compiler_rows
                if row.get("origin_node_exact") and row.get("exact_origin_node")
            })
            proof_ids = sorted({tagged_to_proof[tag] for tag in exact_tagged if tag in tagged_to_proof})
            exact = len(proof_ids) == 1 and (
                proof_ids[0] in allowed or kind in {"EXTERN", "DIRECT_ATEN", "DIRECT_TORCH_OP", "DIRECT_TENSOR_METHOD"}
            )
            row = {
                "task_id": f"{region_id}:{formal}",
                "candidate_region_id": region_id,
                "phase": flow["phase"],
                "implementation_kind": kind,
                "symbol": flow["symbol"],
                "formal_pointer": formal,
                "tensor_variable": variable,
                "compiler_origin_key": key,
                "compiler_origin_rows": compiler_rows,
                "exact_aot_endpoint_id": proof_ids[0] if exact else None,
                "exact_semantic_endpoint_id": proof_ids[0] if exact else None,
                "status": (
                    "COMPILER_ADDED_BOUNDARY_CLOSED_BY_EXACT_THEOREM"
                    if compiler_theorem_closed else
                    "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT"
                    if exact else "INTERNAL_IMPLEMENTATION_BUFFER_PENDING_REGION_CLOSURE"
                ),
                "checks": {
                    "compiler_live_origin_observed": bool(compiler_rows),
                    "unique_exact_origin_node": len(exact_tagged) == 1,
                    "origin_maps_to_proof_id": len(proof_ids) == 1,
                    "origin_is_in_candidate_fb_bridge": (
                        len(proof_ids) == 1 and proof_ids[0] in allowed
                    ),
                    "exact_runtime_output_origin_used": exact,
                    "name_shape_ordinal_or_candidate_value_pairing_used": False,
                },
                "proof_owner_ids": sorted(
                    str(owner["owner_id"])
                    for owner in bridge_row.get("proof_owners", [])
                ),
                "compiler_added_boundary_theorem": (
                    bridge_row["proof_owners"][0].get("theorem")
                    if compiler_theorem_closed else None
                ),
            }
            rows.append(row)
            if exact:
                exact_endpoint_regions[region_id].append(row["task_id"])

    if parameter_closures is not None:
        rows_by_task = {str(row["task_id"]): row for row in rows}
        for closure in parameter_closures["rows"]:
            task_id = str(closure["task_id"])
            if task_id not in rows_by_task:
                raise RuntimeError(f"parameter-gradient closure task is absent: {task_id}")
            row = rows_by_task[task_id]
            if row["exact_semantic_endpoint_id"] is not None:
                raise RuntimeError(f"parameter-gradient closure duplicates an AOT endpoint: {task_id}")
            row["exact_semantic_endpoint_id"] = str(closure["semantic_endpoint_id"])
            row["parameter_gradient_aliases"] = list(closure["parameter_aliases"])
            row["parameter_gradient_identity_mode"] = closure["identity_mode"]
            row["status"] = "EXACT_CANDIDATE_OUTPUT_TO_PARAMETER_GRADIENT_ENDPOINT"
            row["checks"]["exact_accumulate_grad_runtime_identity"] = True
            exact_endpoint_regions[row["candidate_region_id"]].append(task_id)

    # Close implementation-only buffers at the first downstream exact semantic
    # endpoint. Cross-phase traversal is admitted only through a shared proved
    # F+B owner, never through names or tensor metadata.
    for row in rows:
        if row["status"] == "COMPILER_ADDED_BOUNDARY_CLOSED_BY_EXACT_THEOREM":
            row["closed_by_semantic_endpoint_tasks"] = []
            row["closure_uses_candidate_values"] = False
            continue
        if row["exact_semantic_endpoint_id"] is not None:
            row["closed_by_semantic_endpoint_tasks"] = [row["task_id"]]
            continue
        starts = set(successors.get(row["candidate_region_id"], ()))
        starts.discard(row["candidate_region_id"])
        queue = deque(sorted(starts))
        visited = set()
        closures = []
        while queue and not closures:
            level = [queue.popleft() for _ in range(len(queue))]
            for region_id in level:
                if region_id in visited:
                    continue
                visited.add(region_id)
                closures.extend(exact_endpoint_regions.get(region_id, ()))
            if closures:
                break
            for region_id in level:
                queue.extend(sorted(successors.get(region_id, ())))
        row["closed_by_semantic_endpoint_tasks"] = sorted(set(closures))
        if closures:
            row["status"] = "INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT"
        else:
            row["status"] = "UNRESOLVED_NO_EXACT_SEMANTIC_ENDPOINT_CLOSURE"
        row["closure_uses_candidate_values"] = False

    graphs = {
        (str(graph["phase"]), int(graph["graph_index"])): graph
        for graph in proof["standard_aot_capture"]["graphs"]
    }
    nodes_by_graph = {
        key: {str(node["name"]): node for node in graph["nodes"]}
        for key, graph in graphs.items()
    }
    reference_cut_tasks = []
    for endpoint_id in sorted({
        str(row["exact_aot_endpoint_id"])
        for row in rows if row["exact_aot_endpoint_id"] is not None
    }):
        proof_endpoint_id = normalized_to_proof.get(endpoint_id, endpoint_id)
        phase_token, graph_token, node_name = proof_endpoint_id.split(":", 2)
        phase = phase_token.upper()
        graph_index = int(graph_token.removeprefix("graph"))
        graph = graphs[(phase, graph_index)]
        nodes = nodes_by_graph[(phase, graph_index)]
        node = nodes[node_name]
        input_rows = [{
            "source_node": str(edge["source_node"]),
            "consumer_node_id": proof_endpoint_id,
            "consumer_argument_path": edge["argument_path"],
        } for edge in node["input_edges"]]
        output_rows = []
        for user_name in node["users"]:
            user = nodes[str(user_name)]
            for edge in user["input_edges"]:
                if str(edge["source_node"]) == node_name:
                    output_rows.append({
                        "source_node_id": proof_endpoint_id,
                        "consumer_node_id": (
                            f"{phase_token}:graph{graph_index}:{user_name}"
                        ),
                        "consumer_argument_path": edge["argument_path"],
                    })
        reference_cut_tasks.append({
            "task_id": f"same-dtype:{endpoint_id}",
            "cut_id": f"same-dtype:{endpoint_id}",
            "phase": phase,
            "graph_index": graph_index,
            "expected_graph_code_sha256": graph["code_sha256"],
            "aot_node_ids": [proof_endpoint_id],
            "aot_node_names": [node_name],
            "semantic_endpoint_id": endpoint_id,
            "expected_boundary_inputs": input_rows,
            "expected_boundary_outputs": output_rows,
            "required_extractor_schema": "forkcert.fx-reference-cut.v2",
        })

    counts = {
        "candidate_compute_regions": len(dataflow),
        "candidate_regions_with_observed_output_port": len({
            row["candidate_region_id"] for row in rows
        }),
        "stored_candidate_ports": len(rows),
        "exact_semantic_endpoints": sum(
            row["exact_semantic_endpoint_id"] is not None for row in rows
        ),
        "reference_cut_tasks": len(reference_cut_tasks),
        "internal_ports_closed_by_semantic_endpoint": sum(
            row["status"] == "INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT"
            for row in rows
        ),
        "compiler_added_ports_closed_by_exact_theorem": sum(
            row["status"] == "COMPILER_ADDED_BOUNDARY_CLOSED_BY_EXACT_THEOREM"
            for row in rows
        ),
        "unresolved": sum(row["status"].startswith("UNRESOLVED") for row in rows),
        "ports_by_implementation_kind": dict(sorted(Counter(
            row["implementation_kind"] for row in rows
        ).items())),
    }
    counts["candidate_regions_without_observed_output_port"] = (
        counts["candidate_compute_regions"]
        - counts["candidate_regions_with_observed_output_port"]
    )
    payload = {
        "schema": "kernel-analyzer-same-dtype-semantic-task-plan-v1",
        "status": (
            "COMPLETE_ALL_CANDIDATE_PORTS_ASSIGNED_TO_EXACT_SEMANTIC_ENDPOINTS"
            if counts["unresolved"] == 0
            and counts["candidate_regions_without_observed_output_port"] == 0
            else "PARTIAL_FAIL_CLOSED"
        ),
        "bindings": {
            "inventory_result_sha256": inventory["result_sha256"],
            "candidate_fb_bridge_result_sha256": bridge["result_sha256"],
            "proof_capture_result_sha256": proof["result_sha256"],
            "buffer_origin_result_sha256": origins["result_sha256"],
            "proof_id_aliases_sha256": bridge.get("bindings", {}).get(
                "proof_id_aliases_sha256"
            ),
            "parameter_gradient_closure_result_sha256": (
                parameter_closures["result_sha256"]
                if parameter_closures is not None else None
            ),
        },
        "denominator": counts,
        "rows": rows,
        "reference_cut_tasks": reference_cut_tasks,
        "claim_boundary": (
            "Every materialized candidate port is either an exact compiler-carried AOT "
            "semantic endpoint, an internal member covered by a downstream exact endpoint, "
            "or an explicit compiler-added boundary closed by its exact forward/VJP theorem. "
            "This is a comparison plan; numerical same-dtype execution remains separate."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
