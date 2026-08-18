"""Extract a pure executable FX reference region with explicit boundaries."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import torch
from torch.fx import Graph, GraphModule, Node

from scripts.aot_capture import _input_edges


SCHEMA_VERSION = "forkcert.fx-reference-cut.v2"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [str(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    return str(value)


def _node_descriptor(node: Node) -> dict[str, Any]:
    return {
        "name": node.name,
        "op": node.op,
        "target": str(node.target),
        "tensor_meta": _jsonable(node.meta.get("tensor_meta")),
    }


def _schema_mutable(node: Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    return bool(schema is not None and getattr(schema, "is_mutable", False))


def extract_pure_fx_reference_cut(
    *,
    graph_module: GraphModule,
    cut_node_names: Iterable[str],
    cut_id: str,
    node_index: Mapping[str, Node] | None = None,
    node_ordinal: Mapping[str, int] | None = None,
) -> tuple[GraphModule, dict[str, Any]]:
    """Return a standalone pure subgraph and its explicit interface.

    Every dependency outside the cut becomes a boundary placeholder.  Every
    cut value consumed outside the cut becomes a tuple output.  Call order,
    node ordinals, and guessed operator equivalences never define the cut.
    """

    requested = {str(name) for name in cut_node_names}
    if not requested:
        raise ValueError("reference cut must contain at least one node")
    by_name = (
        dict(node_index)
        if node_index is not None
        else {node.name: node for node in graph_module.graph.nodes}
    )
    missing = sorted(requested - set(by_name))
    if missing:
        raise ValueError(f"reference cut nodes are absent: {missing}")
    ordinal = (
        dict(node_ordinal)
        if node_ordinal is not None
        else {
            node.name: index
            for index, node in enumerate(graph_module.graph.nodes)
        }
    )
    cut_nodes = sorted(
        (by_name[name] for name in requested), key=lambda node: ordinal[node.name]
    )
    unsupported = [
        node.name
        for node in cut_nodes
        if node.op != "call_function"
    ]
    if unsupported:
        raise ValueError(
            "reference cut only supports functional call_function nodes: "
            f"{unsupported}"
        )
    mutable = [
        node.name
        for node in cut_nodes
        if _schema_mutable(node) or node.is_impure()
    ]
    if mutable:
        raise ValueError(
            f"reference cut contains mutable/impure nodes: {mutable}"
        )

    cut_set = set(cut_nodes)
    boundary_inputs = sorted(
        {
            node
            for cut_node in cut_nodes
            for node in cut_node.all_input_nodes
            if node not in cut_set
        },
        key=lambda node: ordinal[node.name],
    )
    boundary_outputs = [
        node
        for node in cut_nodes
        if any(user not in cut_set for user in node.users)
    ]
    if not boundary_outputs:
        raise ValueError("reference cut has no externally observed output")

    graph = Graph()
    environment: dict[Node, Node] = {}
    for source in boundary_inputs:
        placeholder = graph.placeholder(f"boundary_{source.name}")
        placeholder.meta = dict(source.meta)
        environment[source] = placeholder
    for node in cut_nodes:
        unavailable = [
            source.name
            for source in node.all_input_nodes
            if source not in environment
        ]
        if unavailable:
            raise ValueError(
                f"reference cut is not topologically closed at {node.name}: "
                f"{unavailable}"
            )
        copied = graph.node_copy(node, lambda source: environment[source])
        copied.meta = dict(node.meta)
        environment[node] = copied
    graph.output(tuple(environment[node] for node in boundary_outputs))
    extracted = GraphModule(graph_module, graph)
    extracted.recompile()

    input_routes = []
    for consumer in cut_nodes:
        for edge in _input_edges(consumer):
            if edge["source_node"] not in {
                node.name for node in boundary_inputs
            }:
                continue
            input_routes.append(
                {
                    "source_node": edge["source_node"],
                    "consumer_node": consumer.name,
                    "consumer_argument_path": edge["argument_path"],
                }
            )
    output_routes = []
    for source in boundary_outputs:
        for user in source.users:
            if user in cut_set:
                continue
            for edge in _input_edges(user):
                if edge["source_node"] != source.name:
                    continue
                output_routes.append(
                    {
                        "source_node": source.name,
                        "consumer_node": user.name,
                        "consumer_op": user.op,
                        "consumer_argument_path": edge["argument_path"],
                    }
                )
    certificate = {
        "schema_version": SCHEMA_VERSION,
        "status": "PURE_FX_REFERENCE_CUT_EXTRACTED",
        "cut_id": str(cut_id),
        "cut_nodes": [_node_descriptor(node) for node in cut_nodes],
        "boundary_inputs": [
            _node_descriptor(node) for node in boundary_inputs
        ],
        "boundary_input_routes": input_routes,
        "boundary_outputs": [
            _node_descriptor(node) for node in boundary_outputs
        ],
        "boundary_output_routes": output_routes,
        "gates": {
            "all_requested_nodes_present": len(cut_nodes) == len(requested),
            "functional_call_nodes_only": not unsupported,
            "mutable_or_impure_nodes_absent": not mutable,
            "boundary_inputs_explicit": all(
                node in environment for node in boundary_inputs
            ),
            "boundary_outputs_nonempty": bool(boundary_outputs),
            "tuple_output_contract": True,
            "same_input_replay_executed": False,
            "semantic_closure_granted": False,
        },
        "graph_code_sha256": hashlib.sha256(
            extracted.code.encode()
        ).hexdigest(),
        "claim_boundary": {
            "supported": (
                "pure executable FX reference region with explicit input "
                "placeholders and externally consumed tuple outputs"
            ),
            "not_supported": [
                "candidate boundary tensors were captured",
                "same-input replay was executed",
                "numerical equivalence",
                "semantic closure",
                "endpoint repair",
            ],
        },
    }
    certificate["certificate_sha256"] = hashlib.sha256(
        json.dumps(
            certificate, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return extracted, certificate
