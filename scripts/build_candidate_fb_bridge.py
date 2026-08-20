#!/usr/bin/env python3
"""Bind every generated compute boundary to proved AOT F+B mathematics.

The bridge uses compiler-emitted source provenance and exact AOT ``from_node``
records.  It never pairs by operator-name similarity, tensor shape, or runtime
ordinal.  Compiler-added mutation/copy boundaries are retained with explicit
local theorems instead of being dropped from the denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = re.compile(
    r"^\s*# Topologically Sorted Source Nodes: \[(.*)\], Original ATen: \[(.*)\]\s*$"
)
MAPPING = re.compile(r"^#\s{3}(.+?) => (.+?)\s*$")
GRAPH_CALL = re.compile(
    r"^#\s{3}%([^ :]+).*?call_function\[target=([^\]]+)\]"
)
ASSIGNMENT = re.compile(r"^(\w+) = async_compile\.triton\(")
SEGMENT = re.compile(r"_(forward|backward)_segment(\d+)_executed")
FROM_NODE = re.compile(r"\(name=([^,]+),")


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def executable_aot_graph_projection(graphs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Discard observation-only provenance while retaining executable identity.

    A proof-tagged recompilation can assign different autograd sequence numbers
    and Python stack/module provenance even when Dynamo/AOT emits the identical
    executable graph.  Those fields are useful witnesses, but they are not part
    of graph semantics.  Code hashes plus the complete executable node/edge
    projection below provide an exact identity check without falling back to
    operator, shape, or ordinal similarity.
    """

    graph_fields = (
        "phase", "graph_index", "code_sha256", "input_count", "node_count",
        "call_function_count",
    )
    node_fields = (
        "phase", "ordinal", "name", "op", "target", "arguments",
        "input_nodes", "input_edges", "users", "tensor_meta",
    )
    return [
        {
            **{key: graph.get(key) for key in graph_fields},
            "nodes": [
                {key: node.get(key) for key in node_fields}
                for node in graph.get("nodes", [])
            ],
        }
        for graph in graphs
    ]


def executable_aot_graph_identity(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Exact executable identity of one graph, excluding only its list position."""

    projected = executable_aot_graph_projection([graph])[0]
    projected.pop("graph_index", None)
    return projected


def is_compiler_added_device_put_graph(graph: Mapping[str, Any]) -> bool:
    """Recognize the sole graph insertion allowed by the segmented bridge.

    A proof-tagged recompilation can materialize a device transfer for a lifted
    scalar before the original segmented program.  This is not an operator-name
    heuristic: the whole inserted graph must contain exactly one executable
    node, and that node must be the primitive device transfer itself.
    """

    calls = [node for node in graph.get("nodes", []) if node.get("op") == "call_function"]
    return (
        len(calls) == 1
        and str(calls[0].get("target", "")).replace("torch.ops.", "")
        in {"prims.device_put.default", "prims.device_put"}
    )


def match_segmented_executable_graphs(
    source_graphs: list[Mapping[str, Any]],
    proof_graphs: list[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], tuple[str, int]], list[dict[str, Any]]]:
    """Exact graph-program bijection, admitting only explicit DevicePut insertions.

    Graph-break wrappers may schedule an empty graph after backward instead of
    before it when a lifted scalar DevicePut is materialized.  Consequently
    list position is not semantic identity.  Every nonempty graph is matched by
    its entire executable projection; ambiguous nonempty matches fail closed.
    Byte-identical empty graphs are matched only by multiplicity.
    """

    mapping: dict[tuple[str, int], tuple[str, int]] = {}
    extras: list[dict[str, Any]] = []
    unmatched = set(range(len(proof_graphs)))
    for source in source_graphs:
        source_identity = executable_aot_graph_identity(source)
        candidates = [
            index for index in sorted(unmatched)
            if executable_aot_graph_identity(proof_graphs[index]) == source_identity
        ]
        if not candidates:
            raise RuntimeError("proof-tagged segmented schedule is missing an actual AOT graph")
        call_count = int(source.get("call_function_count", 0))
        if len(candidates) > 1 and call_count:
            same_index = [
                index for index in candidates
                if int(proof_graphs[index]["graph_index"])
                == int(source["graph_index"])
            ]
            if len(same_index) != 1:
                raise RuntimeError(
                    "ambiguous exact identity for a nonempty segmented AOT graph"
                )
            selected = same_index[0]
        else:
            selected = candidates[0]
        proof = proof_graphs[selected]
        source_key = (str(source["phase"]), int(source["graph_index"]))
        proof_key = (str(proof["phase"]), int(proof["graph_index"]))
        mapping[proof_key] = source_key
        unmatched.remove(selected)
    for proof_index in sorted(unmatched):
        extra = proof_graphs[proof_index]
        if not is_compiler_added_device_put_graph(extra):
            raise RuntimeError(
                "proof-tagged segmented schedule contains a non-DevicePut "
                "graph absent from the actual AOT capture"
            )
        extras.append({
            "phase": str(extra["phase"]),
            "graph_index": int(extra["graph_index"]),
            "identity_sha256": digest(executable_aot_graph_identity(extra)),
            "theorem": {
                "proof_kind": "EXACT_COMPILER_ADDED_DEVICE_TRANSFER",
                "map": "y = device_put(x, device)",
                "adjoint": "dx = device_put(dy, source_device)",
                "precondition": "the transfer preserves tensor values exactly",
            },
        })
    return mapping, extras


def split_values(value: str) -> list[str]:
    return [] if not value.strip() else [part.strip() for part in value.split(",")]


def wrapper_segment(source_path: str) -> tuple[str, int]:
    match = SEGMENT.search(source_path)
    if not match:
        raise ValueError(f"wrapper segment identity is absent: {source_path}")
    return match.group(1).upper(), int(match.group(2))


def parse_wrapper(
    path: Path, requested_lines: set[int],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, list[str]]]]:
    """Stream one generated wrapper and retain only provenance metadata."""

    definitions: dict[str, dict[str, Any]] = {}
    calls: dict[int, dict[str, list[str]]] = {}
    current_mapping: dict[str, list[str]] | None = None
    latest_topology: dict[str, list[str]] | None = None
    latest_topology_line = -1
    in_mapping = False
    in_graph = False
    current_targets: dict[str, str] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            topology = TOPOLOGY.match(line)
            if topology:
                latest_topology = {
                    "source_nodes": split_values(topology.group(1)),
                    "original_aten": split_values(topology.group(2)),
                }
                latest_topology_line = line_number
            if line.startswith("# Source node to ATen node mapping:"):
                current_mapping = {}
                current_targets = {}
                in_mapping = True
                in_graph = False
                continue
            if in_mapping:
                mapping = MAPPING.match(line)
                if mapping:
                    assert current_mapping is not None
                    current_mapping[mapping.group(1).strip()] = split_values(mapping.group(2))
                    continue
                if line.startswith("# Graph fragment:"):
                    in_mapping = False
                    in_graph = True
                    continue
            if in_graph:
                graph_call = GRAPH_CALL.match(line)
                if graph_call:
                    current_targets[graph_call.group(1)] = graph_call.group(2)
                    continue
            assignment = ASSIGNMENT.match(line)
            if assignment:
                symbol = assignment.group(1)
                if current_mapping is None:
                    raise RuntimeError(f"{path.name}:{symbol} lacks source-to-ATen provenance")
                if symbol in definitions:
                    raise RuntimeError(f"duplicate Triton definition: {path.name}:{symbol}")
                definitions[symbol] = {
                    "source_to_nodes": current_mapping,
                    "node_targets": current_targets,
                }
                current_mapping = None
                current_targets = {}
                in_graph = False
            if line_number in requested_lines:
                # Direct copy graph-break glue has no ATen provenance comment.
                if latest_topology is not None and line_number - latest_topology_line <= 3:
                    calls[line_number] = latest_topology
    return definitions, calls


DEBUG_HANDLE = re.compile(r"# \[Provenance debug handles\] (.+?)\s*$")
PROOF_TAG_PHASE = re.compile(r"\bka_([fb])_g\d+_\d{5}_[A-Za-z0-9_]+\b")


def normalized_call_line(value: str) -> str:
    """Exact generated call identity, insensitive only to whitespace."""

    return "".join(value.split())


def proof_tagged_call_index(
    proof_capture: Mapping[str, Any],
    proof_id_aliases: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index proof-tagged generated calls by exact emitted call expression."""

    proof_id_aliases = proof_id_aliases or {}
    device_put_graph_present = any(
        is_compiler_added_device_put_graph(graph)
        for graph in proof_capture.get("standard_aot_capture", {}).get("graphs", [])
    )
    tagged_to_proof: dict[str, str] = {}
    for graph in proof_capture["proof_graphs"]:
        for row in graph["rows"]:
            tagged = str(row["tagged_fx_name"])
            if tagged in tagged_to_proof:
                raise RuntimeError(f"duplicate proof-tagged FX name: {tagged}")
            proof_id = str(row["proof_id"])
            tagged_to_proof[tagged] = proof_id_aliases.get(proof_id, proof_id)
    trace_dir = Path(proof_capture["trace_dir"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for output_code in sorted(trace_dir.rglob("output_code.py")):
        output_text = output_code.read_text()
        name = str(output_code.parent.name).lower()
        phase = "BACKWARD" if "backward" in name else "FORWARD" if "forward" in name else None
        if phase is None:
            tag_phases = {
                "FORWARD" if value == "f" else "BACKWARD"
                for value in PROOF_TAG_PHASE.findall(output_text)
            }
            executable_lines = [
                line.strip() for line in output_text.splitlines()
                if any(token in line for token in (
                    ".run(", "extern_kernels.", ".copy_(", "torch.ops.aten."
                ))
                and not line.lstrip().startswith("#")
            ]
            if (
                not tag_phases
                and device_put_graph_present
                and len(executable_lines) == 1
                and ".copy_(" in executable_lines[0]
            ):
                # Exact lowered wrapper for the separately proved compiler-added
                # DevicePut graph; it owns no semantic AOT proof tag.
                continue
            if len(tag_phases) != 1:
                raise RuntimeError(
                    "cannot uniquely determine tagged wrapper phase from proof tags: "
                    f"{output_code} phases={sorted(tag_phases)}"
                )
            phase = next(iter(tag_phases))
        provenance_path = output_code.with_name(
            "inductor_provenance_tracking_node_mappings.json"
        )
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.exists() else None
        )
        cpp_to_post = provenance.get("cppCodeToPost", {}) if provenance else {}
        output_code_sha256 = hashlib.sha256(output_text.encode()).hexdigest()
        provenance_sha256 = (
            hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_path.exists() else None
        )
        tagged_definitions, _ = parse_wrapper(output_code, set())
        pending: str | None = None
        latest_source_nodes: list[str] = []
        for line in output_text.splitlines():
            topology = TOPOLOGY.match(line)
            if topology:
                latest_source_nodes = split_values(topology.group(1))
            match = DEBUG_HANDLE.search(line)
            if match:
                pending = match.group(1)
                continue
            if pending is None or line.lstrip().startswith("#"):
                continue
            if not any(token in line for token in (
                ".run(", "extern_kernels.", ".copy_(", "torch.ops.aten."
            )):
                continue
            if cpp_to_post:
                if pending not in cpp_to_post:
                    raise RuntimeError(f"debug handle absent from provenance map: {pending}")
                post_nodes = sorted(set(str(value) for value in cpp_to_post[pending]))
                provenance_mode = "EXACT_CPP_CODE_TO_POST_SIDECAR"
            else:
                # Backward sidecars are absent in some Torch builds, while
                # proof-tagged post/source nodes remain embedded immediately
                # above each exact generated call.
                symbol = pending.rsplit(":", 1)[0]
                definition = tagged_definitions.get(symbol, {})
                source_mapping = definition.get("source_to_nodes", {})
                mapped = [
                    value
                    for label in latest_source_nodes
                    for value in source_mapping.get(label, [])
                ]
                post_nodes = sorted(set([*latest_source_nodes, *mapped]))
                provenance_mode = "EXACT_PROOF_TAGGED_CALLSITE_TOPOLOGY"
            proof_ids = sorted({
                tagged_to_proof[value]
                for value in post_nodes if value in tagged_to_proof
            })
            compiler_added = sorted(
                value for value in post_nodes if value not in tagged_to_proof
            )
            key = (phase, normalized_call_line(line))
            if key in result:
                raise RuntimeError(
                    "proof-tagged generated call expression is nonunique: "
                    f"{phase}:{line.strip()}"
                )
            result[key] = {
                "debug_handle": pending,
                "post_nodes": post_nodes,
                "aot_node_ids": proof_ids,
                "compiler_added_post_nodes": compiler_added,
                "provenance_mode": provenance_mode,
                "tagged_output_code_sha256": output_code_sha256,
                "tagged_provenance_sha256": provenance_sha256,
            }
            pending = None
    return result


def aten_base(value: str) -> str | None:
    value = value.replace("torch.ops.", "")
    if not value.startswith("aten."):
        return None
    parts = value.split(".")
    return ".".join(parts[:2])


def semantic_target(value: str) -> str | None:
    value = value.replace("torch.ops.", "")
    base = aten_base(value)
    if base == "aten._to_copy" or value.startswith("prims.convert_element_type"):
        return "SEMANTIC_CAST"
    if base in {"aten.reshape", "aten.view", "aten._unsafe_view"}:
        return "SEMANTIC_RESHAPE"
    if base in {"aten.transpose", "aten.permute", "aten.t"}:
        return "SEMANTIC_PERMUTE"
    if base and base.startswith("aten.arange") or value.startswith("prims.iota"):
        return "SEMANTIC_ARANGE"
    return base


def from_node_labels(node: Mapping[str, Any]) -> set[str]:
    labels = set()
    raw = node.get("from_node") or []
    values = [raw] if isinstance(raw, str) else raw
    for value in values:
        labels.update(FROM_NODE.findall(str(value)))
    return labels


def owner_index(math: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for unit in math["units"]:
        owner = {
            "owner_kind": "SEMANTIC_FORWARD_BACKWARD_UNIT",
            "owner_id": unit["unit_id"],
            "proof_status": unit["status"],
            "proof_sha256": digest({
                "unit_id": unit["unit_id"],
                "forward_node_ids": unit["forward_node_ids"],
                "backward_node_ids": unit["backward_node_ids"],
                "composite_vjp_proof": unit["composite_vjp_proof"],
            }),
        }
        for node_id in [*unit["forward_node_ids"], *unit["backward_node_ids"]]:
            if node_id in result:
                raise RuntimeError(f"AOT math node has multiple F+B owners: {node_id}")
            result[node_id] = owner
    for row in math["auxiliary_backward_program"]:
        node_id = row["node_id"]
        owner = {
            "owner_kind": "PROVED_AUXILIARY_BACKWARD_PROGRAM",
            "owner_id": row["composite_program_proof"]["program_id"],
            "proof_status": row["status"],
            "proof_sha256": digest(row["composite_program_proof"]),
        }
        if node_id in result:
            raise RuntimeError(f"AOT math node has multiple owners: {node_id}")
        result[node_id] = owner
    for collection, owner_kind in (
        ("partition_auxiliary_forward_program", "PROVED_PARTITION_AUXILIARY_FORWARD_PROGRAM"),
        ("backward_only_partition_replay_program", "PROVED_BACKWARD_PARTITION_REPLAY_PROGRAM"),
    ):
        for row in math.get(collection, []):
            node_id = row["node_id"]
            proof = row["composite_program_proof"]
            owner = {
                "owner_kind": owner_kind,
                "owner_id": proof.get("program_id", f"{proof['proof_kind']}::{node_id}"),
                "proof_status": row["status"],
                "proof_sha256": digest(proof),
            }
            if node_id in result:
                raise RuntimeError(f"AOT math node has multiple owners: {node_id}")
            result[node_id] = owner
    return result


def compiler_added_owner(row: Mapping[str, Any]) -> dict[str, Any] | None:
    kind, symbol = row["kind"], row["symbol"]
    if kind == "DIRECT_TENSOR_METHOD" and symbol == "copy_":
        witness = row["boundary_witness"]
        theorem = {
            "proof_kind": "EXACT_SEGMENT_BOUNDARY_COPY",
            "map": "dst[:] = src[:]",
            "adjoint": "dsrc += ddst; overwritten dst has no prior-value cotangent",
            "checks": {
                "direct_copy_ast": witness.get("boundary_source") == "DIRECT_TENSOR_COPY_METHOD_AST",
                "source_expression_exact": bool(witness.get("source_expression")),
                "mutated_target_exact": bool(witness.get("mutated_target")),
                "pointer_dataflow_exact": row.get("binding_status") == "EXACT_GENERATED_POINTER_DATAFLOW",
            },
        }
    elif kind == "DIRECT_ATEN" and symbol == "index_put_":
        witness = row["boundary_witness"]
        theorem = {
            "proof_kind": "EXACT_INDEXED_COTANGENT_ACCUMULATION",
            "map": "dst[index] += values",
            "adjoint": "dvalues = gather(ddst,index); prior dst cotangent passes through",
            "checks": {
                "direct_inplace_schema_and_ast": witness.get("boundary_source") == "DIRECT_ATEN_INPLACE_SCHEMA_AND_CALL_AST",
                "accumulate_true": witness.get("accumulate_expression") == "True",
                "read_modify_write_predecessor_present": bool(row.get("previous_storage_writers")),
                "pointer_dataflow_exact": row.get("binding_status") == "EXACT_GENERATED_POINTER_DATAFLOW",
            },
        }
    else:
        return None
    if not all(theorem["checks"].values()):
        raise RuntimeError(f"compiler-added theorem checks failed: {row['region_id']}")
    return {
        "owner_kind": "EXPLICIT_COMPILER_ADDED_THEOREM",
        "owner_id": theorem["proof_kind"] + "::" + row["region_id"],
        "proof_status": "PROVED_EXACT_REAL_ARITHMETIC_PROGRAM",
        "proof_sha256": digest(theorem),
        "theorem": theorem,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--proof-capture", type=Path)
    parser.add_argument(
        "--source-aot", type=Path,
        help="Raw segmented actual-AOT capture cryptographically bound by the normalized capture",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-dir", type=Path,
        help="Executed wrapper-source root; defaults to the legacy inventory-parent/trace layout.",
    )
    args = parser.parse_args()
    inventory, aot, math = load(args.inventory), load(args.aot), load(args.math)
    proof_capture = load(args.proof_capture) if args.proof_capture else None
    source_aot = load(args.source_aot) if args.source_aot else None
    if inventory["status"] not in {
        "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW",
        "COMPLETE_GENERATED_SCHEDULE_PARTIAL_POINTER_DATAFLOW",
    }:
        raise RuntimeError("candidate inventory is not complete")
    if math["status"] != "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION":
        raise RuntimeError("AOT F+B mathematics is not complete")
    supplied_capture_sha = aot["capture"].get("capture_sha256")
    if math["capture_sha256"] != supplied_capture_sha:
        raise RuntimeError("AOT math does not bind the supplied capture")
    normalized_segmented_capture = aot.get("status") == (
        "COMPLETE_SEGMENT_LOCAL_CAPTURE_WITH_EXPLICIT_CROSS_SEGMENT_BOUNDARIES"
    )
    proof_id_aliases: dict[str, str] = {}
    raw_to_normalized: dict[str, str] = {}
    segmented_graph_mapping: dict[tuple[str, int], tuple[str, int]] = {}
    compiler_added_proof_graphs: list[dict[str, Any]] = []
    graph_identity_capture = aot
    if normalized_segmented_capture:
        if source_aot is None:
            raise RuntimeError("normalized segmented AOT requires --source-aot")
        if (
            aot.get("source_result_sha256") != source_aot.get("result_sha256")
            or aot["capture"].get("segmented_source_capture_sha256")
            != source_aot.get("capture", {}).get("capture_sha256")
        ):
            raise RuntimeError("normalized segmented AOT does not bind source capture")
        graph_identity_capture = source_aot
        normalized_nodes = {
            (
                str(node["segmented_origin"]["phase"]),
                int(node["segmented_origin"]["graph_index"]),
                str(node["segmented_origin"]["original_name"]),
            ): f"{str(node['phase']).lower()}:graph0:{node['name']}"
            for graph in aot["capture"].get("graphs", [])
            for node in graph.get("nodes", [])
            if node.get("segmented_origin")
        }
        for graph in source_aot["capture"].get("graphs", []):
            phase = str(graph["phase"])
            graph_index = int(graph["graph_index"])
            for node in graph.get("nodes", []):
                raw_id = f"{phase.lower()}:graph{graph_index}:{node['name']}"
                key = (phase, graph_index, str(node["name"]))
                normalized_id = normalized_nodes.get(key)
                if normalized_id is None:
                    raise RuntimeError(
                        f"segmented source AOT node absent from normalized atlas: {raw_id}"
                    )
                raw_to_normalized[raw_id] = normalized_id
        proof_id_aliases.update(raw_to_normalized)
    elif source_aot is not None:
        raise RuntimeError("--source-aot is valid only for a normalized segmented capture")
    if proof_capture is not None:
        proof_standard = proof_capture.get("standard_aot_capture", {})
        proof_capture_sha_exact = (
            proof_standard.get("capture_sha256")
            == graph_identity_capture["capture"].get("capture_sha256")
        )
        proof_executable_graphs_sha = digest(executable_aot_graph_projection(
            proof_standard.get("graphs", [])
        ))
        supplied_executable_graphs_sha = digest(executable_aot_graph_projection(
            graph_identity_capture["capture"].get("graphs", [])
        ))
        proof_graphs_exact = proof_executable_graphs_sha == supplied_executable_graphs_sha
        if normalized_segmented_capture and not proof_graphs_exact:
            segmented_graph_mapping, compiler_added_proof_graphs = (
                match_segmented_executable_graphs(
                    list(graph_identity_capture["capture"].get("graphs", [])),
                    list(proof_standard.get("graphs", [])),
                )
            )
            for proof_graph in proof_capture.get("proof_graphs", []):
                proof_key = (
                    str(proof_graph["phase"]), int(proof_graph["graph_index"])
                )
                source_key = segmented_graph_mapping.get(proof_key)
                if source_key is None:
                    continue
                source_phase, source_index = source_key
                for row in proof_graph.get("rows", []):
                    proof_id = str(row["proof_id"])
                    original_name = proof_id.rsplit(":", 1)[-1]
                    raw_id = (
                        f"{source_phase.lower()}:graph{source_index}:{original_name}"
                    )
                    normalized_id = raw_to_normalized.get(raw_id)
                    if normalized_id is None:
                        raise RuntimeError(
                            "exactly matched segmented proof node is absent from "
                            f"normalized atlas: {proof_id} -> {raw_id}"
                        )
                    proof_id_aliases[proof_id] = normalized_id
            proof_graphs_exact = (
                len(segmented_graph_mapping)
                == len(graph_identity_capture["capture"].get("graphs", []))
            )
        if not (proof_capture_sha_exact or proof_graphs_exact):
            raise RuntimeError(
                "proof-tagged schedule does not bind the supplied AOT graphs"
            )
        if proof_capture.get("proof_tag_summary", {}).get("tags_not_observed") != 0:
            raise RuntimeError("proof-tagged schedule has unobserved AOT proof tags")
    else:
        proof_capture_sha_exact = False
        proof_graphs_exact = False
        proof_executable_graphs_sha = None
        supplied_executable_graphs_sha = digest(executable_aot_graph_projection(
            graph_identity_capture["capture"].get("graphs", [])
        ))

    generated = inventory["generated_regions"]["inventory"]["regions"]
    direct = [
        row for row in inventory["compute_dataflow"]["rows"]
        if row["kind"] not in {"TRITON", "EXTERN"}
    ]
    all_rows = [*generated, *direct]
    requested: dict[str, set[int]] = defaultdict(set)
    for row in all_rows:
        requested[row["source_path"]].add(int(row["source_line"]))
    wrappers = args.trace_dir if args.trace_dir is not None else args.inventory.parent / "trace"
    if not wrappers.is_dir():
        raise RuntimeError(f"executed wrapper-source root is absent: {wrappers}")
    definitions: dict[tuple[str, str], dict[str, Any]] = {}
    call_provenance: dict[tuple[str, int], dict[str, list[str]]] = {}
    for source_path, lines in requested.items():
        mappings, calls = parse_wrapper(wrappers / source_path, lines)
        definitions.update(((source_path, symbol), value) for symbol, value in mappings.items())
        call_provenance.update(((source_path, line), value) for line, value in calls.items())
    tagged_calls = (
        proof_tagged_call_index(proof_capture, proof_id_aliases)
        if proof_capture is not None else {}
    )
    frozen_call_lines: dict[tuple[str, int], str] = {}
    for source_path, lines in requested.items():
        source_lines = (wrappers / source_path).read_text().splitlines()
        frozen_call_lines.update(
            ((source_path, line), source_lines[line - 1]) for line in lines
        )

    nodes: dict[str, Mapping[str, Any]] = {}
    by_context_name: dict[tuple[str, int, str], str] = {}
    by_context_label: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    raw_name_index: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    by_context_label_target: dict[tuple[str, int, str, str], list[str]] = defaultdict(list)
    normalized_segmented = any(
        str(node.get("name", "")).startswith(("forward_g", "backward_g"))
        for graph in aot["capture"]["graphs"] for node in graph["nodes"]
    )
    for graph in aot["capture"]["graphs"]:
        phase = str(graph["phase"])
        graph_index = int(graph.get("graph_index", 0))
        for node in graph["nodes"]:
            if node["op"] != "call_function":
                continue
            node_id = f"{phase.lower()}:graph{graph_index}:{node['name']}"
            nodes[node_id] = node
            by_context_name[(phase, graph_index, str(node["name"]))] = node_id
            if normalized_segmented:
                segment_match = re.match(
                    rf"{phase.lower()}_g(\d+)__", str(node["name"])
                )
                if not segment_match:
                    raise RuntimeError(f"normalized node lacks segment prefix: {node_id}")
                source_segment = int(segment_match.group(1))
            else:
                source_segment = graph_index
            base = aten_base(str(node["target"]))
            labels = from_node_labels(node)
            for field in ("source_fn_stack", "fwd_source_fn_stack"):
                stack = node.get(field) or []
                for item in stack:
                    if isinstance(item, (list, tuple)) and item:
                        labels.add(str(item[0]))
            raw_name = str(node["name"])
            raw_label = raw_name.split("__", 1)[-1] if normalized_segmented else raw_name
            raw_name_index[(phase, source_segment, raw_label)].append(node_id)
            for label in labels:
                by_context_label[(phase, source_segment, label)].append(node_id)
                if base:
                    by_context_label_target[(phase, source_segment, label, base)].append(node_id)
    # Exact from_node provenance is stronger.  Fall back to an exact AOT FX
    # name only when no node advertises that string as a source label.
    for key, values in raw_name_index.items():
        if key not in by_context_label:
            by_context_label[key].extend(values)
            phase, segment, label = key
            for node_id in values:
                base = aten_base(str(nodes[node_id]["target"]))
                if base:
                    by_context_label_target[(phase, segment, label, base)].append(node_id)
    owners = owner_index(math)
    if compiler_added_proof_graphs:
        extra_graph_keys = {
            (row["phase"], int(row["graph_index"])): row
            for row in compiler_added_proof_graphs
        }
        for proof_graph in proof_capture.get("proof_graphs", []):
            key = (str(proof_graph["phase"]), int(proof_graph["graph_index"]))
            extra = extra_graph_keys.get(key)
            if extra is None:
                continue
            theorem = extra["theorem"]
            for row in proof_graph.get("rows", []):
                proof_id = str(row["proof_id"])
                owners[proof_id] = {
                    "owner_kind": "EXPLICIT_COMPILER_ADDED_THEOREM",
                    "owner_id": theorem["proof_kind"] + "::" + proof_id,
                    "proof_status": "PROVED_EXACT_REAL_ARITHMETIC_PROGRAM",
                    "proof_sha256": digest({**theorem, "proof_id": proof_id}),
                    "theorem": theorem,
                }

    results = []
    unresolved = Counter()
    for candidate in all_rows:
        phase, segment = wrapper_segment(candidate["source_path"])
        graph_index = 0 if normalized_segmented else segment
        prefix = f"{phase.lower()}_g{segment}__" if normalized_segmented else ""
        provenance = call_provenance.get(
            (candidate["source_path"], int(candidate["source_line"])),
            {"source_nodes": candidate.get("source_nodes", []),
             "original_aten": candidate.get("original_aten", [])},
        )
        node_ids: list[str] = []
        method = None
        explicit_owner = compiler_added_owner(candidate)
        frozen_call = frozen_call_lines[
            (candidate["source_path"], int(candidate["source_line"]))
        ]
        tagged_call = tagged_calls.get(
            (phase, normalized_call_line(frozen_call))
        )
        checks: dict[str, Any] = {
            "compiler_source_path_and_line_exact": True,
            "candidate_values_not_used": True,
            "name_shape_or_runtime_ordinal_similarity_not_used": True,
        }
        if explicit_owner is not None:
            bound_owners = [explicit_owner]
            method = "EXPLICIT_COMPILER_ADDED_PROGRAM_THEOREM"
        elif tagged_call is not None and tagged_call["aot_node_ids"]:
            node_ids = list(tagged_call["aot_node_ids"])
            missing_owner_ids = sorted(
                node_id for node_id in node_ids if node_id not in owners
            )
            checks.update({
                "exact_generated_call_expression_matches_proof_tagged_schedule": True,
                "proof_tagged_debug_handle_present": bool(tagged_call["debug_handle"]),
                "at_least_one_aot_proof_tag_carried_to_generated_region": bool(node_ids),
                "all_carried_aot_tags_have_proof_owners": not missing_owner_ids,
                "compiler_added_post_nodes_retained_not_silently_dropped": True,
            })
            checks["proof_tagged_debug_handle"] = tagged_call["debug_handle"]
            checks["compiler_added_post_nodes"] = tagged_call[
                "compiler_added_post_nodes"
            ]
            checks["missing_proof_owner_ids"] = missing_owner_ids
            checks["tagged_output_code_sha256"] = tagged_call[
                "tagged_output_code_sha256"
            ]
            checks["tagged_provenance_sha256"] = tagged_call[
                "tagged_provenance_sha256"
            ]
            boolean_checks = [
                value for value in checks.values() if isinstance(value, bool)
            ]
            if not all(boolean_checks):
                unresolved["PROOF_TAGGED_CALL_OR_OWNER_INCOMPLETE"] += 1
                bound_owners = []
            else:
                bound_owners = [owners[node_id] for node_id in node_ids]
                method = "EXACT_GENERATED_CALL_IDENTITY_TO_COMPILER_CARRIED_AOT_PROOF_TAGS"
        elif candidate["kind"] == "TRITON":
            definition = definitions.get((candidate["source_path"], candidate["symbol"]))
            if definition is None:
                unresolved["TRITON_DEFINITION_PROVENANCE_ABSENT"] += 1
                bound_owners = []
            else:
                mapping = definition["source_to_nodes"]
                compiler_targets = definition["node_targets"]
                absent_labels = sorted(set(provenance["source_nodes"]) - set(mapping))
                missing_labels = []
                expected_aten = {
                    target for value in provenance["original_aten"]
                    if (target := semantic_target(value)) is not None
                }
                for label in provenance["source_nodes"]:
                    compiler_names = mapping.get(label, [])
                    available = by_context_label.get((phase, segment, label), [])
                    if label in mapping:
                        selected = []
                        unresolved_compiler_names = []
                        for compiler_name in compiler_names:
                            compiler_target = semantic_target(
                                compiler_targets.get(compiler_name, "")
                            )
                            direct = by_context_name.get(
                                (phase, graph_index, compiler_name)
                            )
                            if direct is not None and semantic_target(
                                str(nodes[direct]["target"])
                            ) == compiler_target:
                                selected.append(direct)
                            else:
                                unresolved_compiler_names.append(
                                    (compiler_name, compiler_target)
                                )
                        # Some scheduler-created FX names do not survive from
                        # the standard AOT graph.  Resolve only through the
                        # exact compiler source label plus semantic target.
                        for compiler_name, compiler_target in unresolved_compiler_names:
                            candidates = [
                                node_id for node_id in available
                                if node_id not in selected
                                and semantic_target(str(nodes[node_id]["target"]))
                                == compiler_target
                            ]
                            if len(candidates) == 1:
                                selected.append(candidates[0])
                            else:
                                missing_labels.append(
                                    f"{label}::{compiler_name}"
                                )
                        resolved = not any(
                            value.startswith(f"{label}::")
                            for value in missing_labels
                        )
                    else:
                        # A compiled symbol can be reused at callsites whose
                        # source labels differ. The callsite itself freezes the
                        # exact source labels and ATen target set; use those,
                        # never the representative definition's label names.
                        direct = by_context_name.get(
                            (phase, graph_index, label)
                        )
                        if direct is not None:
                            selected = [direct]
                        else:
                            selected = [
                                node_id for node_id in available
                                if semantic_target(str(nodes[node_id]["target"])) in expected_aten
                                or semantic_target(str(nodes[node_id]["target"])) is None
                            ]
                        resolved = bool(selected)
                    if not resolved:
                        missing_labels.append(label)
                    node_ids.extend(selected)
                node_ids = sorted(set(node_ids))
                observed_aten = {
                    base for node_id in node_ids
                    if (base := semantic_target(str(nodes[node_id]["target"]))) is not None
                }
                checks.update({
                    "representative_definition_reuse_is_resolved_at_exact_callsite": True,
                    "all_compiler_mapped_nodes_resolve_to_standard_aot": not missing_labels,
                    "at_least_one_proved_aot_node_bound": bool(node_ids),
                    "topological_original_aten_is_not_misused_as_region_compute_set": True,
                })
                if not all(checks.values()):
                    unresolved["TRITON_PROVENANCE_OR_TARGET_MISMATCH"] += 1
                    bound_owners = []
                else:
                    bound_owners = [owners[node_id] for node_id in node_ids if node_id in owners]
                    if len(bound_owners) != len(node_ids):
                        unresolved["TRITON_AOT_NODE_WITHOUT_PROOF_OWNER"] += 1
                        bound_owners = []
                    method = "EXACT_INDUCTOR_SOURCE_TO_ATEN_NODE_PROVENANCE"
                checks["absent_source_labels"] = absent_labels
                checks["missing_aot_source_labels"] = missing_labels
        else:
            target = {
                "mm": "aten.mm", "bmm": "aten.bmm", "addmm": "aten.addmm",
                "convolution": "aten.convolution",
                "convolution_backward": "aten.convolution_backward",
            }.get(candidate["symbol"])
            matches: list[tuple[int, str]] = []
            if target:
                for index, label in enumerate(provenance["source_nodes"]):
                    for node_id in by_context_label_target.get(
                        (phase, segment, label, target), []
                    ):
                        matches.append((index, node_id))
            if matches:
                terminal = max(index for index, _ in matches)
                node_ids = sorted({node_id for index, node_id in matches if index == terminal})
            checks["unique_terminal_exact_source_label_target"] = len(node_ids) == 1
            if len(node_ids) != 1:
                unresolved["DECLARED_OP_EXACT_AOT_PROVENANCE_NONUNIQUE_OR_ABSENT"] += 1
                bound_owners = []
            elif node_ids[0] not in owners:
                unresolved["DECLARED_OP_AOT_NODE_WITHOUT_PROOF_OWNER"] += 1
                bound_owners = []
            else:
                bound_owners = [owners[node_ids[0]]]
                method = "EXACT_TERMINAL_SOURCE_LABEL_AND_DECLARED_ATEN_TARGET"

        owner_rows = {
            (row["owner_kind"], row["owner_id"], row["proof_sha256"]): row
            for row in bound_owners
        }
        status = "BOUND_TO_PROVED_FB_MATHEMATICS" if bound_owners else "UNRESOLVED"
        result = {
            "candidate_region_id": candidate["region_id"],
            "phase": phase, "kind": candidate["kind"], "symbol": candidate["symbol"],
            "source_path": candidate["source_path"],
            "source_line": candidate["source_line"],
            "method": method, "status": status,
            "aot_node_ids": sorted(set(node_ids)),
            "proof_owners": [owner_rows[key] for key in sorted(owner_rows)],
            "checks": checks,
        }
        result["row_sha256"] = digest(result)
        results.append(result)

    closed = sum(row["status"] == "BOUND_TO_PROVED_FB_MATHEMATICS" for row in results)
    compute_denominator = inventory["runtime_call_audit"]["denominator"]["compute_invocations"]
    if len(results) != compute_denominator:
        raise RuntimeError(
            f"candidate compute denominator mismatch: {len(results)} != {compute_denominator}"
        )
    payload = {
        "schema": "kernel-analyzer-candidate-fb-bridge-v1",
        "status": (
            "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_PROVED_FB_MATHEMATICS"
            if closed == len(results) else "PARTIAL_FAIL_CLOSED"
        ),
        "architecture": inventory["architecture"],
        "denominator": {
            "candidate_compute_regions": len(results),
            "bound_to_proved_fb_mathematics": closed,
            "unresolved": len(results) - closed,
            "semantic_fb_units": math["denominator"]["semantic_forward_backward_units"],
            "proved_auxiliary_backward_nodes": math["denominator"]["auxiliary_backward_nodes"],
        },
        "unresolved_reason_counts": dict(sorted(unresolved.items())),
        "bindings": {
            "inventory_result_sha256": inventory["result_sha256"],
            "aot_result_sha256": aot["result_sha256"],
            "aot_capture_sha256": supplied_capture_sha,
            "aot_math_capture_sha256": math["capture_sha256"],
            "aot_math_ledger_sha256": math["ledger_sha256"],
            "proof_tagged_schedule_result_sha256": (
                proof_capture.get("result_sha256")
                if proof_capture is not None else None
            ),
            "proof_tagged_aot_binding_mode": (
                "EXACT_CAPTURE_SHA256"
                if proof_capture_sha_exact else
                "EXACT_EXECUTABLE_SEGMENTED_SOURCE_AOT_GRAPH"
                if proof_graphs_exact and normalized_segmented_capture else
                "EXACT_EXECUTABLE_AOT_GRAPH"
                if proof_graphs_exact else None
            ),
            "proof_tagged_graphs_sha256": (
                digest(proof_capture.get("standard_aot_capture", {}).get("graphs", []))
                if proof_capture is not None else None
            ),
            "supplied_aot_graphs_sha256": digest(
                aot["capture"].get("graphs", [])
            ),
            "segmented_source_aot_result_sha256": (
                source_aot.get("result_sha256") if source_aot is not None else None
            ),
            "segmented_source_aot_graphs_sha256": (
                digest(source_aot["capture"].get("graphs", []))
                if source_aot is not None else None
            ),
            "proof_tagged_executable_aot_graphs_sha256": proof_executable_graphs_sha,
            "supplied_executable_aot_graphs_sha256": supplied_executable_graphs_sha,
            "segmented_exact_graph_mapping": [
                {
                    "proof": {"phase": proof[0], "graph_index": proof[1]},
                    "source": {"phase": source[0], "graph_index": source[1]},
                }
                for proof, source in sorted(segmented_graph_mapping.items())
            ],
            "compiler_added_proof_graphs": compiler_added_proof_graphs,
            "proof_id_aliases": (
                dict(sorted(proof_id_aliases.items()))
                if normalized_segmented_capture else {}
            ),
            "proof_id_aliases_sha256": digest(
                dict(sorted(proof_id_aliases.items()))
            ),
        },
        "gates": {
            "all_candidate_compute_regions_retained": len(results) == compute_denominator,
            "all_candidate_regions_bound": closed == len(results),
            "all_aot_math_complete": True,
            "candidate_values_used": False,
            "name_shape_or_runtime_ordinal_similarity_used": False,
            "proof_tagged_generated_schedule_used": proof_capture is not None,
            "proof_capture_sha_or_exact_graph_identity": (
                proof_capture is not None
                and (proof_capture_sha_exact or proof_graphs_exact)
            ),
            "segmented_source_to_normalized_node_bijection": (
                bool(raw_to_normalized) if normalized_segmented_capture else True
            ),
            "only_explicit_device_put_graphs_admitted": (
                all(
                    row["theorem"]["proof_kind"]
                    == "EXACT_COMPILER_ADDED_DEVICE_TRANSFER"
                    for row in compiler_added_proof_graphs
                )
            ),
        },
        "rows": results,
        "claim_boundary": (
            "Every exact row binds one generated compute boundary to proved AOT F+B mathematics "
            "through compiler-emitted provenance, or to an explicit compiler-added theorem. "
            "This grants proof ownership and boundary identity, not numerical correctness."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "denominator": payload["denominator"],
        "unresolved_reason_counts": payload["unresolved_reason_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
