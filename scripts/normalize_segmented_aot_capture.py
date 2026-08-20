#!/usr/bin/env python3
"""Normalize paired segmented AOT graphs into one proof-only namespaced atlas."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def unique_runtime_pairs(
    runs: list[dict[str, Any]], observation_stability: dict[str, Any],
) -> list[tuple[dict[str, Any], list[int], bool]]:
    """Group one compiled pair while exposing repeat-identity disagreement."""

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    order: list[tuple[int, int]] = []
    for run in runs:
        key = (
            int(run["forward_phase"]["graph_index"]),
            int(run["backward_phase"]["graph_index"]),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(run)

    result = []
    for key in order:
        observations = grouped[key]
        if len(observations) > 1:
            if not observation_stability or not all(observation_stability.values()):
                raise RuntimeError(
                    "repeated segmented AOT pair lacks exact observation stability"
                )
            canonical = []
            for run in observations:
                value = copy.deepcopy(run)
                value.pop("run_index", None)
                # The weak global forward registry intentionally survives
                # repeated calls to the same compiled program.  A later run
                # can therefore contain byte-identical duplicate identity
                # rows.  Collapse only exact duplicates; any new unique
                # identity evidence must still make the repeat comparison
                # fail closed.
                for backward_input in value.get("backward_inputs", []):
                    matches = backward_input.get("global_forward_matches", [])
                    unique = {
                        json.dumps(row, sort_keys=True, separators=(",", ":")): row
                        for row in matches
                    }
                    backward_input["global_forward_matches"] = [
                        unique[key] for key in sorted(unique)
                    ]
                canonical.append(digest(value))
            identity_evidence_exact = len(set(canonical)) == 1
        else:
            identity_evidence_exact = True
        # Dynamic programs (notably MoE routing) can execute the same compiled
        # graph pair more than once while exposing different runtime objects to
        # the global identity registry.  The graph-local program is still a
        # valid proof object, but the differing cross-segment identity evidence
        # must remain unresolved.  Retain the first observation as the local
        # graph witness and record the disagreement instead of either merging
        # identities or dropping the graph from the denominator.
        result.append((
            observations[0],
            [int(row["run_index"]) for row in observations],
            identity_evidence_exact,
        ))
    return result


def mapped_name(value: str, prefix: str, aliases: dict[str, str]) -> str:
    return aliases.get(value, prefix + value)


def rewrite_nodes(value: Any, prefix: str, aliases: dict[str, str]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"node"}:
            return {"node": mapped_name(str(value["node"]), prefix, aliases)}
        return {key: rewrite_nodes(item, prefix, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_nodes(item, prefix, aliases) for item in value]
    return value


def without_segment_marker(stack: Any) -> Any:
    if (
        isinstance(stack, list) and stack
        and stack[0][0] == "__kernel_analyzer_segment_pair__"
    ):
        return stack[1:]
    return stack


def namespaced_graph(
    graph: dict[str, Any], *, pair: str, ordinal_start: int,
    aliases: dict[str, str] | None = None,
    equivalences: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    aliases = aliases or {}
    equivalences = equivalences or {}
    prefix = f"{graph['phase'].lower()}_g{graph['graph_index']}__"
    result = []
    ordinal = ordinal_start
    for raw in graph["nodes"]:
        node = copy.deepcopy(raw)
        node["name"] = prefix + str(raw["name"])
        node["arguments"] = rewrite_nodes(raw.get("arguments"), prefix, aliases)
        node["input_nodes"] = [
            mapped_name(str(value), prefix, aliases) for value in raw.get("input_nodes", [])
        ]
        node["input_edges"] = [
            {**edge, "source_node": mapped_name(str(edge["source_node"]), prefix, aliases)}
            for edge in raw.get("input_edges", [])
        ]
        node["users"] = [
            "output" if value == "output" else prefix + str(value)
            for value in raw.get("users", [])
        ]
        marker = [["__kernel_analyzer_segment_pair__", pair]]
        if raw.get("source_fn_stack") is not None:
            node["source_fn_stack"] = marker + list(raw["source_fn_stack"])
        if raw.get("fwd_source_fn_stack") is not None:
            node["fwd_source_fn_stack"] = marker + list(raw["fwd_source_fn_stack"])
        if raw.get("nn_module_stack") is not None:
            node["nn_module_stack"] = {f"segment::{pair}": pair, **raw["nn_module_stack"]}
        node["ordinal"] = ordinal
        node["segmented_origin"] = {
            "phase": graph["phase"], "graph_index": graph["graph_index"],
            "original_name": raw["name"], "pair": pair,
        }
        equivalent_forward_nodes = equivalences.get(str(raw["name"]))
        if equivalent_forward_nodes is not None:
            node["runtime_identity_mode"] = "EXACT_STORAGE_VIEW_EQUIVALENCE_CLASS"
            node["runtime_identity_forward_equivalence_nodes"] = equivalent_forward_nodes
            node["runtime_identity_equivalence_is_value_exact"] = True
        result.append(node)
        ordinal += 1
    return result, ordinal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outer = load(args.capture)
    capture = outer["capture"]
    graphs = {(graph["phase"], int(graph["graph_index"])): graph for graph in capture["graphs"]}
    bridge = capture["cross_phase_runtime_bridge"]
    forward_pair: dict[int, str] = {}
    backward_pair: dict[int, str] = {}
    backward_aliases: dict[int, dict[str, str]] = {}
    backward_equivalences: dict[int, dict[str, list[str]]] = {}
    pair_rows = []
    unique_pairs = unique_runtime_pairs(
        bridge["runs"], outer.get("observation_stability", {})
    )
    for run, observed_run_indices, repeat_identity_evidence_exact in unique_pairs:
        forward_index = int(run["forward_phase"]["graph_index"])
        backward_index = int(run["backward_phase"]["graph_index"])
        pair = f"run{run['run_index']}:forward{forward_index}:backward{backward_index}"
        if forward_index in forward_pair or backward_index in backward_pair:
            raise RuntimeError("segmented AOT graph appears in more than one runtime pair")
        forward_pair[forward_index] = pair
        backward_pair[backward_index] = pair
        aliases: dict[str, str] = {}
        equivalences: dict[str, list[str]] = {}
        forward_prefix = f"forward_g{forward_index}__"
        for row in run["backward_inputs"]:
            global_matches_by_identity = {
                json.dumps(match, sort_keys=True, separators=(",", ":")): match
                for match in row.get("global_forward_matches", [])
            }
            global_matches = [
                global_matches_by_identity[key]
                for key in sorted(global_matches_by_identity)
            ]
            pair_local_matches = [
                match for match in global_matches
                if int(match["phase_graph_index"]) == forward_index
            ]
            pair_local_exact = [
                match for match in pair_local_matches
                if match["identity_mode"] == "EXACT_PYTHON_OBJECT"
            ]
            selected_pair_local = pair_local_exact or pair_local_matches
            if len(selected_pair_local) == 1:
                match = selected_pair_local[0]
                aliases[str(row["placeholder"])] = (
                    f"forward_g{forward_index}__{match['source_node']}"
                )
                continue
            if (
                len(selected_pair_local) > 1
                and all(
                    match["identity_mode"]
                    in {"EXACT_PYTHON_OBJECT", "EXACT_STORAGE_VIEW"}
                    for match in selected_pair_local
                )
            ):
                equivalences[str(row["placeholder"])] = sorted({
                    f"forward_g{forward_index}__{match['source_node']}"
                    for match in selected_pair_local
                })
                continue
            global_exact = [
                match for match in global_matches
                if match["identity_mode"] == "EXACT_PYTHON_OBJECT"
            ]
            selected_global = global_exact or global_matches
            if len(selected_global) == 1:
                match = selected_global[0]
                graph_index = int(match["phase_graph_index"])
                source_node = str(match["source_node"])
                aliases[str(row["placeholder"])] = f"forward_g{graph_index}__{source_node}"
                continue
            if (
                len(selected_global) > 1
                and all(
                    match["identity_mode"] in {"EXACT_PYTHON_OBJECT", "EXACT_STORAGE_VIEW"}
                    for match in selected_global
                )
            ):
                equivalences[str(row["placeholder"])] = sorted({
                    f"forward_g{int(match['phase_graph_index'])}__{match['source_node']}"
                    for match in selected_global
                })
                continue
            exact_outputs = [
                match for match in row["forward_output_matches"]
                if match["identity_mode"] == "EXACT_PYTHON_OBJECT"
            ]
            output_matches = exact_outputs or row["forward_output_matches"]
            input_matches = row["forward_input_matches"]
            selected = None
            if len(output_matches) == 1:
                token = str(output_matches[0]["runtime_token"])
                selected = token.rsplit(":", 1)[1]
            elif len(input_matches) == 1:
                token = str(input_matches[0]["runtime_token"])
                selected = token.rsplit(":", 1)[1]
            if selected is not None:
                aliases[str(row["placeholder"])] = forward_prefix + selected
            elif (
                len(output_matches) > 1
                and all(match["identity_mode"] == "EXACT_STORAGE_VIEW" for match in output_matches)
            ):
                equivalences[str(row["placeholder"])] = sorted(
                    forward_prefix + str(match["runtime_token"]).rsplit(":", 1)[1]
                    for match in output_matches
                )
        backward_aliases[backward_index] = aliases
        backward_equivalences[backward_index] = equivalences
        pair_rows.append({
            "pair": pair, "forward_graph_index": forward_index,
            "backward_graph_index": backward_index,
            "observed_run_indices": observed_run_indices,
            "repeat_observations_present": len(observed_run_indices) > 1,
            "repeat_runtime_identity_evidence_exact": repeat_identity_evidence_exact,
            "runtime_identity_gates": run["gates"],
            "backward_placeholder_runtime_identity_aliases": dict(sorted(aliases.items())),
            "backward_placeholder_runtime_identity_equivalence_classes": dict(
                sorted(equivalences.items())
            ),
        })

    forward_nodes = []
    backward_nodes = []
    ordinal = 0
    for phase, destination, pairs in (
        ("FORWARD", forward_nodes, forward_pair),
        ("BACKWARD", backward_nodes, backward_pair),
    ):
        phase_graphs = sorted(
            (graph for (candidate_phase, _), graph in graphs.items() if candidate_phase == phase),
            key=lambda graph: int(graph["graph_index"]),
        )
        for graph in phase_graphs:
            index = int(graph["graph_index"])
            pair = pairs.get(index, f"{phase.lower()}-only-{index}")
            nodes, ordinal = namespaced_graph(
                graph, pair=pair, ordinal_start=ordinal,
                aliases=(backward_aliases.get(index, {}) if phase == "BACKWARD" else {}),
                equivalences=(
                    backward_equivalences.get(index, {}) if phase == "BACKWARD" else {}
                ),
            )
            destination.extend(nodes)

    # A graph break can place a forward op and its actual AOT backward in
    # different runtime pairs.  Recover only bindings that are unique under
    # autograd's exact seq_nr plus the complete, marker-free source stack.
    # Tensor names, ordinals, shapes and candidate values are deliberately not
    # part of this bridge.
    forward_origins = {
        json.dumps(node.get("source_fn_stack"), sort_keys=True)
        for node in forward_nodes if node["op"] == "call_function"
    }
    cross_segment_origin_bindings = []
    for backward in backward_nodes:
        if backward["op"] != "call_function":
            continue
        current = json.dumps(backward.get("fwd_source_fn_stack"), sort_keys=True)
        if current in forward_origins:
            continue
        candidates = [
            forward for forward in forward_nodes
            if forward["op"] == "call_function"
            and forward.get("seq_nr") is not None
            and forward.get("seq_nr") == backward.get("seq_nr")
            and without_segment_marker(forward.get("source_fn_stack"))
            == without_segment_marker(backward.get("fwd_source_fn_stack"))
        ]
        if len(candidates) != 1:
            continue
        forward = candidates[0]
        backward["cross_segment_origin_binding"] = {
            "mode": "UNIQUE_EXACT_SEQ_NR_AND_FULL_SOURCE_STACK",
            "forward_node": forward["name"],
            "seq_nr": forward["seq_nr"],
            "name_shape_ordinal_or_candidate_value_used": False,
        }
        backward["fwd_source_fn_stack"] = copy.deepcopy(forward["source_fn_stack"])
        cross_segment_origin_bindings.append({
            "forward_node": forward["name"],
            "backward_node": backward["name"],
            **backward["cross_segment_origin_binding"],
        })

    def graph_payload(phase: str, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "phase": phase, "graph_index": 0,
            "code_sha256": digest([
                (node["segmented_origin"], node["target"], node["arguments"])
                for node in nodes
            ]),
            "input_count": sum(node["op"] == "placeholder" for node in nodes),
            "node_count": len(nodes),
            "call_function_count": sum(node["op"] == "call_function" for node in nodes),
            "nodes": nodes,
        }

    normalized_capture = {
        **capture,
        "schema_version": "kernel-analyzer-segment-namespaced-proof-atlas-v1",
        "graphs": [graph_payload("FORWARD", forward_nodes), graph_payload("BACKWARD", backward_nodes)],
        "phase_graph_counts": {"FORWARD": 1, "BACKWARD": 1},
        "segmented_source_capture_sha256": capture["capture_sha256"],
        "segment_pairing": {
            "pairs": pair_rows,
            "raw_runtime_pair_observations": len(bridge["runs"]),
            "unique_compiled_runtime_pairs": len(unique_pairs),
            "unpaired_forward_graph_indices": sorted(
                index for phase, index in graphs if phase == "FORWARD" and index not in forward_pair
            ),
            "unpaired_backward_graph_indices": sorted(
                index for phase, index in graphs if phase == "BACKWARD" and index not in backward_pair
            ),
            "pairing_uses_runtime_identity_only": True,
            "cross_segment_unresolved_edges_remain_explicit": (
                not all(bridge["gates"].values())
                or not all(
                    row["repeat_runtime_identity_evidence_exact"] for row in pair_rows
                )
            ),
            "cross_segment_origin_bindings": cross_segment_origin_bindings,
            "cross_segment_origin_binding_rule": "unique exact seq_nr plus complete marker-free source stack",
        },
    }
    normalized_capture["capture_sha256"] = digest(normalized_capture)
    payload = {
        **outer,
        "schema": "kernel-analyzer-segment-namespaced-aot-proof-capture-v1",
        "status": "COMPLETE_SEGMENT_LOCAL_CAPTURE_WITH_EXPLICIT_CROSS_SEGMENT_BOUNDARIES",
        "source_result_sha256": outer["result_sha256"],
        "capture": normalized_capture,
        "claim_boundary": (
            "Graph names are namespaced and runtime-paired only for local mathematical proof. "
            "Unresolved eager cross-segment control/data edges are not converted into AOT edges."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "pairing": {
            "raw_runtime_pair_observations": len(bridge["runs"]),
            "unique_compiled_runtime_pairs": len(unique_pairs),
            "unpaired_forward_graphs": len(
                normalized_capture["segment_pairing"]["unpaired_forward_graph_indices"]
            ),
            "unpaired_backward_graphs": len(
                normalized_capture["segment_pairing"]["unpaired_backward_graph_indices"]
            ),
            "cross_segment_origin_bindings": len(cross_segment_origin_bindings),
            "unresolved_edges_remain_explicit": normalized_capture[
                "segment_pairing"
            ]["cross_segment_unresolved_edges_remain_explicit"],
        },
        "forward_call_function_nodes": normalized_capture["graphs"][0]["call_function_count"],
        "backward_call_function_nodes": normalized_capture["graphs"][1]["call_function_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
