#!/usr/bin/env python3
"""Map every changed closed F+B unit to all reachable parameter gradients."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_from_meta(meta: Any) -> list[int]:
    if not isinstance(meta, list) or not meta or not isinstance(meta[0], list):
        raise ValueError(f"unexpected tensor_meta: {meta!r}")
    return [int(value) for value in meta[0]]


def module_path(raw: str) -> str | None:
    root = "L['self'].subject"
    if raw == root:
        return ""
    prefix = root + "."
    return raw[len(prefix):] if raw.startswith(prefix) else None


def parameter_bindings(
    forward: dict[str, Any], model_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    canonical = {id(parameter): name for name, parameter in model.named_parameters()}
    aliases: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)

    nodes = {row["name"]: row for row in forward["nodes"]}
    rows = []
    by_primal = {}
    for primal in [row for row in forward["nodes"] if row["op"] == "placeholder"]:
        meta = primal.get("tensor_meta")
        requires_grad = bool(isinstance(meta, list) and len(meta) > 2 and meta[2])
        if not requires_grad:
            continue
        candidates: dict[int, dict[str, Any]] = {}
        binding_users = []
        binding_modules = set()
        for user_name in primal["users"]:
            user = nodes[user_name]
            for value in (user.get("nn_module_stack") or {}).values():
                raw = value[0] if isinstance(value, list) and value else ""
                path = module_path(raw)
                if path is None:
                    continue
                module = model.get_submodule(path) if path else model
                for _, parameter in module.named_parameters(recurse=False):
                    parameter_id = id(parameter)
                    if parameter_id not in canonical:
                        continue
                    candidates[parameter_id] = {
                        "name": canonical[parameter_id],
                        "aliases": sorted(aliases[parameter_id]),
                        "shape": list(parameter.shape),
                    }
                    binding_users.append(user_name)
                    binding_modules.add(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"{primal['name']} has {len(candidates)} direct module parameter bindings"
            )
        parameter_id, value = next(iter(candidates.items()))
        if shape_from_meta(meta) != value["shape"]:
            raise RuntimeError(f"shape mismatch for {primal['name']} and {value['name']}")
        row = {
            "parameter_id": "",
            "name": value["name"],
            "aliases": value["aliases"],
            "shape": value["shape"],
            "primal": primal["name"],
            "binding_user_nodes": sorted(set(binding_users)),
            "binding_module_paths": sorted(binding_modules),
            "binding_basis": "EXACT_FORWARD_CONSUMER_MODULE_STACK_AND_DIRECT_PARAMETER_IDENTITY",
            "_object_id": parameter_id,
        }
        rows.append(row)
        by_primal[primal["name"]] = row

    if len(rows) != len(canonical) or len({row["_object_id"] for row in rows}) != len(canonical):
        raise RuntimeError("not every unique trainable parameter was bound exactly once")
    rows.sort(key=lambda row: row["name"])
    for index, row in enumerate(rows):
        row["parameter_id"] = f"p{index:03d}"
        row.pop("_object_id")
    return rows, by_primal


def output_endpoints(
    forward: dict[str, Any],
    backward: dict[str, Any],
    parameters: list[dict[str, Any]],
    by_primal: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    placeholders = [row for row in forward["nodes"] if row["op"] == "placeholder"]
    output = [row for row in backward["nodes"] if row["op"] == "output"]
    if len(output) != 1:
        raise RuntimeError("backward graph must have exactly one output node")
    endpoints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for edge in output[0]["input_edges"]:
        output_index = int(edge["argument_path"][-1])
        if output_index >= len(placeholders):
            raise RuntimeError(f"backward output index {output_index} exceeds forward inputs")
        primal = placeholders[output_index]["name"]
        parameter = by_primal.get(primal)
        if parameter is None:
            raise RuntimeError(f"gradient output {output_index} maps to nonparameter {primal}")
        source = edge["source_node"]
        endpoints[source].append(parameter)
        parameter["backward_output_index"] = output_index
        parameter["gradient_source_node"] = source
        seen.add(parameter["parameter_id"])
    expected = {row["parameter_id"] for row in parameters}
    if seen != expected or len(output[0]["input_edges"]) != len(parameters):
        raise RuntimeError("backward output endpoints do not partition all parameters once")
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=FINAL / "aot_carrier_graph.json.gz"
    )
    parser.add_argument(
        "--atlas", type=Path, default=FINAL / "invocation_atlas.json"
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--output", type=Path, default=FINAL / "carrier_census.json"
    )
    args = parser.parse_args()

    capture = read_json(args.capture)
    atlas = read_json(args.atlas)
    if capture["status"] != "STABLE_AOT_FORWARD_BACKWARD_CAPTURE":
        raise RuntimeError("AOT capture is not stable")
    required_capture_gates = (
        "all_parameter_gradients_exact",
        "all_nodes_have_input_edge_records",
        "all_graph_outputs_have_port_records",
        "loss_exact",
    )
    if not all(capture["gates"].get(gate) for gate in required_capture_gates):
        raise RuntimeError("AOT capture gates are incomplete")
    graphs = capture["capture"]["graphs"]
    forward = next(row for row in graphs if row["phase"] == "FORWARD")
    backward = next(row for row in graphs if row["phase"] == "BACKWARD")
    parameters, by_primal = parameter_bindings(forward, args.model)
    endpoints = output_endpoints(forward, backward, parameters, by_primal)

    forward_nodes = {row["name"]: row for row in forward["nodes"]}
    backward_nodes = {row["name"]: row for row in backward["nodes"]}
    nodes = {
        **{f"F:{name}": row for name, row in forward_nodes.items()},
        **{f"B:{name}": row for name, row in backward_nodes.items()},
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    for prefix, graph in (("F", forward), ("B", backward)):
        graph_names = {row["name"] for row in graph["nodes"]}
        for node in graph["nodes"]:
            for edge in node["input_edges"]:
                if edge["source_node"] in graph_names:
                    adjacency[f"{prefix}:{edge['source_node']}"].add(
                        f"{prefix}:{node['name']}"
                    )

    bridge = capture["capture"]["cross_phase_runtime_bridge"]
    if bridge["run_count"] < 1 or not all(bridge["gates"].values()):
        raise RuntimeError("cross-phase runtime bridge is incomplete")
    bridge_run = bridge["runs"][0]
    forward_output_by_token = {
        row["runtime_token"]: row["source_node"]
        for row in bridge_run["forward_outputs"]
    }
    cross_phase_edges = set()
    for backward_input in bridge_run["backward_inputs"]:
        for match in backward_input["forward_output_matches"]:
            source = forward_output_by_token[match["runtime_token"]]
            target = backward_input["placeholder"]
            edge = (f"F:{source}", f"B:{target}")
            if edge[0] not in nodes or edge[1] not in nodes:
                raise RuntimeError(f"invalid cross-phase edge: {edge}")
            adjacency[edge[0]].add(edge[1])
            cross_phase_edges.add(edge)

    combined_endpoints = {
        f"B:{source}": endpoint_rows for source, endpoint_rows in endpoints.items()
    }

    rows = []
    all_reached = set()
    pair_count = 0
    reach_counts = Counter()
    for unit in atlas["changed_units"]:
        candidate_phases = {
            value.split(":", 1)[0].upper()
            for value in unit["candidate_region_ids"]
        }
        start_ids = []
        starts = []
        if "FORWARD" in candidate_phases:
            start_ids.extend(unit["forward_node_ids"])
            starts.extend(
                f"F:{value.rsplit(':', 1)[-1]}"
                for value in unit["forward_node_ids"]
            )
        if "BACKWARD" in candidate_phases:
            start_ids.extend(unit["actual_backward_node_ids"])
            starts.extend(
                f"B:{value.rsplit(':', 1)[-1]}"
                for value in unit["actual_backward_node_ids"]
            )
        missing = [value for value in starts if value not in nodes]
        if missing:
            rows.append({
                "unit_id": unit["unit_id"],
                "status": "UNRESOLVED_AOT_NODE",
                "start_node_ids": start_ids,
                "missing_node_names": missing,
                "parameter_endpoints": [],
            })
            continue

        distance = {name: 0 for name in starts}
        parent: dict[str, str] = {}
        origin = {name: name for name in starts}
        queue = deque(sorted(starts))
        while queue:
            source = queue.popleft()
            for target in sorted(adjacency[source]):
                if target in distance:
                    continue
                distance[target] = distance[source] + 1
                parent[target] = source
                origin[target] = origin[source]
                queue.append(target)

        reached = []
        full_witness = []
        for source, endpoint_rows in combined_endpoints.items():
            if source not in distance:
                continue
            path = [source]
            while path[-1] not in starts:
                path.append(parent[path[-1]])
            path.reverse()
            for parameter in endpoint_rows:
                reached.append([parameter["parameter_id"], distance[source]])
                full_witness.append({
                    "parameter_id": parameter["parameter_id"],
                    "start": origin[source],
                    "path": path,
                })
                all_reached.add(parameter["parameter_id"])
        reached.sort()
        full_witness.sort(key=lambda row: row["parameter_id"])
        status = (
            "EXACT_AOT_PARAMETER_REACHABILITY"
            if reached else "UNRESOLVED_NO_PARAMETER_ENDPOINT"
        )
        pair_count += len(reached)
        reach_counts[len(reached)] += 1
        rows.append({
            "unit_id": unit["unit_id"],
            "status": status,
            "start_node_ids": start_ids,
            "parameter_endpoints": reached,
            "reachable_parameter_count": len(reached),
            "path_witness_sha256": hashlib.sha256(
                json.dumps(full_witness, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })

    exact_units = sum(row["status"] == "EXACT_AOT_PARAMETER_REACHABILITY" for row in rows)
    unresolved = [row for row in rows if row["status"] != "EXACT_AOT_PARAMETER_REACHABILITY"]
    output = {
        "schema": "kernel-analyzer-all-parameter-carrier-census-v1",
        "subject": "Qwen3-1.7B changed closed F+B units to parameter-gradient endpoints",
        "sources": {
            "aot_capture": str(args.capture),
            "aot_capture_file_sha256": sha256(args.capture),
            "aot_capture_payload_sha256": capture["capture"]["capture_sha256"],
            "invocation_atlas": str(args.atlas),
            "invocation_atlas_sha256": sha256(args.atlas),
            "model_config": str(args.model / "config.json"),
            "model_config_sha256": sha256(args.model / "config.json"),
        },
        "denominator": {
            "changed_closed_fbv_units": len(atlas["changed_units"]),
            "exact_parameter_reachability_units": exact_units,
            "unresolved_units": len(unresolved),
            "trainable_unique_parameters": len(parameters),
            "parameters_reached_by_any_changed_unit": len(all_reached),
            "unit_parameter_reachability_pairs": pair_count,
            "exact_forward_saved_tensor_backward_edges": len(cross_phase_edges),
        },
        "reachability_count_histogram": {
            str(key): value for key, value in sorted(reach_counts.items())
        },
        "gates": {
            "candidate_values_used_to_select_or_classify": False,
            "all_parameter_gradients_exact_in_capture": True,
            "cross_phase_runtime_identity_bridge_exact": all(bridge["gates"].values()),
            "every_parameter_bound_by_forward_consumer_module_stack": len(parameters) == 310,
            "name_shape_or_ordinal_pairing_used_for_parameter_identity": False,
            "every_changed_unit_start_node_present": not any(
                row["status"] == "UNRESOLVED_AOT_NODE" for row in rows
            ),
            "every_changed_unit_reaches_a_parameter_gradient": not any(
                row["status"] == "UNRESOLVED_NO_PARAMETER_ENDPOINT" for row in rows
            ),
            "all_changed_units_exactly_mapped": not unresolved,
        },
        "parameters": parameters,
        "rows": rows,
        "unresolved_units": unresolved,
        "boundary": (
            "This is exact static AOT dataflow reachability from each real-changed phase of a "
            "closed F+B unit, including runtime-identity-bound forward saved tensors entering "
            "the actual backward, to parameter-gradient outputs. Reachability is a possible "
            "carrier, not evidence that a candidate produces a nonzero, biased, or persistent "
            "numerical delta. Dynamic candidate values were not used."
        ),
        "natural_bias_case_added": False,
        "property_claim": False,
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
