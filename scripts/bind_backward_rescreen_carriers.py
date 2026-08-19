#!/usr/bin/env python3
"""Bind backward rescreen representatives to exact reachable parameters.

The binding follows AOT dataflow from each exact backward endpoint to backward
graph outputs, then maps those outputs to lifted model parameters.  It never
uses candidate values, T1--T4, error magnitude, or a hand-chosen carrier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import gzip
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/backward_equivalence_rescreen.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/backward_equivalence_carriers.json"
MODELS = {
    "qwen": ROOT.parent / "models/Qwen/Qwen3-1.7B",
    "phi4": ROOT.parent / "models/microsoft/Phi-4-mini-instruct",
    "deepseek8b": ROOT.parent / "models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mamba": ROOT.parent / "models/state-spaces/mamba-130m-hf",
}
CAPTURES = {
    "qwen": ROOT / "results/coverage/standard_aot/qwen_seq64_capture.json.gz",
    "phi4": ROOT / "results/coverage/runtime_releases/phi4_seq64_r1/default_aot_capture.json.gz",
    "deepseek8b": ROOT / "results/coverage/runtime_releases/deepseek8b_seq64_r1/default_aot_capture_raw.json.gz",
    "mamba": ROOT / "results/coverage/runtime_releases/mamba_seq64_r1/default_aot_capture.json.gz",
}


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_shape(node: dict[str, Any]) -> tuple[int, ...] | None:
    meta = node.get("tensor_meta")
    if not isinstance(meta, list) or not meta or not isinstance(meta[0], list):
        return None
    return tuple(int(value) for value in meta[0])


def normalize_stack_path(raw: str) -> str:
    value = raw
    value = re.sub(r"^L\['self'\]\.?(?:subject\.)?", "", value)
    value = re.sub(r"\[['\"]([^'\"]+)['\"]\]", r".\1", value)
    value = re.sub(r"\[(\d+)\]", r".\1", value)
    return value.strip(".")


def build_meta_model(path: Path) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, trust_remote_code=False)


def bind_forward_parameters(
    forward: dict[str, Any], model_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    model = build_meta_model(model_path)
    canonical = {id(parameter): name for name, parameter in model.named_parameters()}
    aliases: dict[int, list[str]] = defaultdict(list)
    module_aliases: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    for name, module in model.named_modules(remove_duplicate=False):
        module_aliases[id(module)].append(name)
    named_modules = dict(model.named_modules())
    nodes = {row["name"]: row for row in forward["nodes"]}
    rows = []
    by_primal = {}

    for primal in (row for row in forward["nodes"] if row["op"] == "placeholder"):
        meta = primal.get("tensor_meta")
        if not (isinstance(meta, list) and len(meta) > 2 and bool(meta[2])):
            continue
        shape = tensor_shape(primal)
        candidates: dict[int, dict[str, Any]] = {}
        witnesses = []
        for user_name in primal.get("users", []):
            user = nodes.get(user_name)
            if user is None:
                continue
            stack_paths = []
            for value in (user.get("nn_module_stack") or {}).values():
                raw = value[0] if isinstance(value, (list, tuple)) and value else ""
                path = normalize_stack_path(str(raw))
                if path:
                    stack_paths.append((path.count(".") + 1, str(raw), path))

            # A stack commonly contains both a leaf module and every parent.
            # Searching all of them made a Conv1d bias ambiguous with a same-
            # shaped parameter owned by its parent Mamba mixer.  Use the most
            # specific stack tier that contains a shape-compatible registered
            # parameter, and only then fall back toward the parent.
            for depth in sorted({row[0] for row in stack_paths}, reverse=True):
                tier_candidates: dict[int, dict[str, Any]] = {}
                tier_witnesses = []
                for _, raw, path in (row for row in stack_paths if row[0] == depth):
                    matches = [
                        name for name in named_modules
                        if name == path or name.endswith("." + path) or path.endswith("." + name)
                    ]
                    for module_name in matches:
                        module = named_modules[module_name]
                        for _, parameter in module.named_parameters(recurse=False):
                            if id(parameter) not in canonical or tuple(parameter.shape) != shape:
                                continue
                            tier_candidates[id(parameter)] = {
                                "name": canonical[id(parameter)],
                                "aliases": sorted(aliases[id(parameter)]),
                                "shape": list(parameter.shape),
                            }
                            tier_witnesses.append({
                                "user": user_name,
                                "stack": raw,
                                "module": module_name,
                                "specificity_depth": depth,
                            })
                if tier_candidates:
                    candidates.update(tier_candidates)
                    witnesses.extend(tier_witnesses)
                    break
        if len(candidates) != 1:
            # The exact module stack is the primary binding.  A unique
            # registration-order/shape match is allowed only as a fail-closed
            # fallback and is explicitly marked below.
            rows.append({
                "primal": primal["name"], "shape": list(shape or ()),
                "status": "UNRESOLVED_PARAMETER_BINDING",
                "candidate_names": sorted(value["name"] for value in candidates.values()),
            })
            continue
        value = next(iter(candidates.values()))
        row = {
            "primal": primal["name"], "name": value["name"],
            "aliases": value["aliases"], "shape": value["shape"],
            "status": "EXACT_MODULE_STACK_PARAMETER_BINDING",
            "witnesses": witnesses,
        }
        rows.append(row); by_primal[primal["name"]] = row
    del model
    return rows, by_primal


def endpoint_reachability(
    capture: dict[str, Any], by_primal: dict[str, dict[str, Any]], endpoints: set[str]
) -> dict[str, dict[str, Any]]:
    forward = next(row for row in capture["graphs"] if row["phase"] == "FORWARD")
    backward = next(row for row in capture["graphs"] if row["phase"] == "BACKWARD")
    all_placeholders = [row for row in forward["nodes"] if row["op"] == "placeholder"]
    nodes = {row["name"]: row for row in backward["nodes"]}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for node in backward["nodes"]:
        for edge in node.get("input_edges", []):
            source = edge.get("source_node")
            if source in nodes:
                adjacency[source].add(node["name"])
    outputs = [row for row in backward["nodes"] if row["op"] == "output"]
    if not outputs:
        raise RuntimeError("backward graph has no output")
    parameter_outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output in outputs:
        origin = output.get("segmented_origin") or {}
        pair = str(origin.get("pair", ""))
        paired = re.search(r"forward(\d+):backward\d+", pair)
        if paired:
            forward_index = int(paired.group(1))
            placeholders = [row for row in all_placeholders
                            if int((row.get("segmented_origin") or {}).get("graph_index", -1)) == forward_index]
        else:
            placeholders = all_placeholders
        for edge in output.get("input_edges", []):
            path = edge.get("argument_path") or []
            if not path:
                continue
            index = int(path[-1])
            if index >= len(placeholders):
                continue
            binding = by_primal.get(placeholders[index]["name"])
            if binding is not None:
                parameter_outputs[str(edge["source_node"])].append(binding)

    endpoint_nodes = {}
    for node in backward["nodes"]:
        origin = node.get("segmented_origin") or {}
        graph_index = int(origin.get("graph_index", 0))
        original_name = str(origin.get("original_name", node["name"]))
        endpoint_nodes[f"backward:graph{graph_index}:{original_name}"] = node["name"]

    result = {}
    for endpoint in endpoints:
        node_name = endpoint_nodes.get(endpoint, endpoint.rsplit(":", 1)[-1])
        if node_name not in nodes:
            result[endpoint] = {"status": "UNRESOLVED_ENDPOINT_NODE", "parameters": []}
            continue
        distance = {node_name: 0}
        queue = deque([node_name])
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                if target not in distance:
                    distance[target] = distance[source] + 1
                    queue.append(target)
        reached = []
        for source, bindings in parameter_outputs.items():
            if source not in distance:
                continue
            for binding in bindings:
                reached.append({
                    "name": binding["name"], "aliases": binding["aliases"],
                    "shape": binding["shape"], "aot_distance": distance[source],
                    "gradient_source_node": source,
                })
        reached.sort(key=lambda row: (row["aot_distance"], row["name"]))
        result[endpoint] = {
            "status": "EXACT_AOT_PARAMETER_REACHABILITY" if reached else "UNRESOLVED_NO_PARAMETER_REACH",
            "parameters": reached,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    bound = []
    parameter_binding_summary = {}
    for model_name in MODELS:
        capture = load(CAPTURES[model_name]).get("capture")
        forward = next(row for row in capture["graphs"] if row["phase"] == "FORWARD")
        parameter_rows, by_primal = bind_forward_parameters(forward, MODELS[model_name])
        representatives = [row for row in source["representatives"] if row["model"] == model_name]
        endpoints = {str(row["representative"]["exact_aot_endpoint_id"]) for row in representatives}
        reach = endpoint_reachability(capture, by_primal, endpoints)
        for row in representatives:
            endpoint = str(row["representative"]["exact_aot_endpoint_id"])
            entry = dict(row)
            entry["reachability"] = reach[endpoint]
            entry["nearest_carrier"] = (
                reach[endpoint]["parameters"][0] if reach[endpoint]["parameters"] else None
            )
            bound.append(entry)
        parameter_binding_summary[model_name] = {
            "trainable_parameter_placeholders": len(parameter_rows),
            "exactly_bound": sum(row["status"] == "EXACT_MODULE_STACK_PARAMETER_BINDING"
                                 for row in parameter_rows),
            "unresolved": sum(row["status"] != "EXACT_MODULE_STACK_PARAMETER_BINDING"
                              for row in parameter_rows),
        }
        del capture, forward, parameter_rows, by_primal, reach

    result = {
        "schema": "kernel-analyzer-backward-equivalence-carriers-v1",
        "status": "BOUND" if all(row["nearest_carrier"] for row in bound) else "PARTIAL",
        "equivalence_cell_count": len(bound),
        "cells_with_reachable_carrier": sum(row["nearest_carrier"] is not None for row in bound),
        "selection_uses_candidate_values_or_historical_verdict": False,
        "parameter_binding_summary": parameter_binding_summary,
        "cells": bound,
        "boundary": (
            "Static AOT reachability identifies possible downstream parameter carriers. "
            "The two-state engineering screen must still confirm a nonzero dynamic effect."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "cells": len(bound), "bound": result["cells_with_reachable_carrier"],
        "parameters": parameter_binding_summary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
