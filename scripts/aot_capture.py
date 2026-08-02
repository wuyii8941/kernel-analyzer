"""Capture AOTAutograd forward/backward graphs as candidate semantic regions.

A dispatcher mode around a warmed ``torch.compile`` callable can cause a
guard miss and expose the eager path.  Candidate coverage therefore cannot be
proven by applying the eager dispatcher observer to the compiled callable.
This module captures the functionalized AOT forward and backward graphs at
compile time, before lower-level fusion.

The AOT graphs are a structural bridge, not yet the generated-kernel
inventory.  A later mapping must bind their nodes to Inductor/Triton regions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import torch


SCHEMA_VERSION = "forkcert.aot-forward-backward-graph-capture.v4"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [str(item) for item in value.shape],
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "unknown")),
            "stride": (
                [str(item) for item in value.stride()]
                if callable(getattr(value, "stride", None))
                else None
            ),
        }
    return str(value)


def _tensor_meta(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_tensor_meta(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return _jsonable(value)
    stride = getattr(value, "stride", None)
    return {
        "shape": [str(item) for item in shape],
        "dtype": str(dtype),
        "stride": (
            [str(item) for item in stride]
            if stride is not None and not callable(stride)
            else None
        ),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
    }


def _input_edges(node: torch.fx.Node) -> tuple[dict[str, Any], ...]:
    edges = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, torch.fx.Node):
            edges.append(
                {
                    "source_node": value.name,
                    "source_op": value.op,
                    "argument_path": list(path),
                }
            )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, (*path, str(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(node.args, ("args",))
    walk(node.kwargs, ("kwargs",))
    return tuple(edges)


@dataclass(frozen=True)
class AOTNodeDescriptor:
    phase: str
    ordinal: int
    name: str
    op: str
    target: str
    arguments: Any
    input_nodes: tuple[str, ...]
    input_edges: tuple[dict[str, Any], ...]
    users: tuple[str, ...]
    tensor_meta: Any
    nn_module_stack: Any
    source_fn_stack: Any
    seq_nr: int | None
    fwd_source_fn_stack: Any
    fwd_nn_module_stack: Any
    original_aten: Any
    from_node: Any
    is_gradient_acc: bool
    partitioner_tag: Any
    stack_trace: str | None

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class AOTGraphDescriptor:
    phase: str
    graph_index: int
    code_sha256: str
    nodes: tuple[AOTNodeDescriptor, ...]
    input_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "graph_index": self.graph_index,
            "code_sha256": self.code_sha256,
            "input_count": self.input_count,
            "node_count": len(self.nodes),
            "call_function_count": sum(
                node.op == "call_function" for node in self.nodes
            ),
            "nodes": [node.as_dict() for node in self.nodes],
        }


class AOTForwardBackwardCapture:
    def __init__(
        self,
        *,
        reference_cut_tasks: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.graphs: list[AOTGraphDescriptor] = []
        self.reference_cut_tasks = tuple(
            dict(task) for task in reference_cut_tasks
        )
        self.reference_cut_extractions: list[dict[str, Any]] = []
        self.reference_cut_replay_runs: list[dict[str, Any]] = []
        self.cross_phase_runtime_bridges: list[dict[str, Any]] = []
        self._active_bridge: dict[str, Any] | None = None

    @staticmethod
    def _tensor_leaves(value: Any) -> list[torch.Tensor]:
        leaves = []
        if isinstance(value, torch.Tensor):
            leaves.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                leaves.extend(AOTForwardBackwardCapture._tensor_leaves(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                leaves.extend(AOTForwardBackwardCapture._tensor_leaves(item))
        return leaves

    @staticmethod
    def _runtime_identity_mode(left: Any, right: Any) -> str | None:
        if left is right:
            return "EXACT_PYTHON_OBJECT"
        if not (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
        ):
            return None
        if (
            left.device != right.device
            or left.dtype != right.dtype
            or left.shape != right.shape
            or left.stride() != right.stride()
            or left.storage_offset() != right.storage_offset()
        ):
            return None
        try:
            same_storage = (
                left.untyped_storage().data_ptr()
                == right.untyped_storage().data_ptr()
                and left.untyped_storage().nbytes()
                == right.untyped_storage().nbytes()
            )
        except RuntimeError:
            same_storage = False
        return "EXACT_STORAGE_VIEW" if same_storage else None

    def bind_user_outputs(self, value: Any) -> None:
        """Bind returned user tensors to live AOT forward graph outputs."""

        if self._active_bridge is None:
            raise RuntimeError("no active AOT forward bridge")
        user_values = self._tensor_leaves(value)
        for row in self._active_bridge["forward_outputs"]:
            row["is_user_output"] = any(
                self._runtime_identity_mode(row["_value"], item)
                is not None
                for item in user_values
            )

    def bind_user_cotangent(self, value: Any) -> Any:
        """Record the exact cotangent object supplied to the AOT backward."""

        if self._active_bridge is None:
            raise RuntimeError("no active AOT forward bridge")
        self._active_bridge["_user_cotangents"] = self._tensor_leaves(value)
        return value

    @staticmethod
    def _public_bridge(bridge: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "run_index": bridge["run_index"],
            "forward_phase": bridge["forward_phase"],
            "backward_phase": bridge.get("backward_phase"),
            "forward_outputs": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "_value"
                }
                for row in bridge["forward_outputs"]
            ],
            "forward_inputs": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "_value"
                }
                for row in bridge["forward_inputs"]
            ],
            "backward_inputs": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "_value"
                }
                for row in bridge.get("backward_inputs", [])
            ],
            "gates": bridge.get("gates", {}),
        }

    def _record(
        self,
        phase: str,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
    ) -> None:
        descriptors = []
        for ordinal, node in enumerate(graph_module.graph.nodes):
            input_nodes = tuple(
                sorted(
                    {
                        input_node.name
                        for input_node in node.all_input_nodes
                    }
                )
            )
            descriptors.append(
                AOTNodeDescriptor(
                    phase=phase,
                    ordinal=ordinal,
                    name=node.name,
                    op=node.op,
                    target=str(node.target),
                    arguments=_jsonable(
                        {
                            "args": node.args,
                            "kwargs": node.kwargs,
                        }
                    ),
                    input_nodes=input_nodes,
                    input_edges=_input_edges(node),
                    users=tuple(sorted(user.name for user in node.users)),
                    tensor_meta=_tensor_meta(node.meta.get("tensor_meta")),
                    nn_module_stack=_jsonable(
                        node.meta.get("nn_module_stack")
                    ),
                    source_fn_stack=_jsonable(
                        node.meta.get("source_fn_stack")
                    ),
                    seq_nr=(
                        int(node.meta["seq_nr"])
                        if node.meta.get("seq_nr") is not None
                        else None
                    ),
                    fwd_source_fn_stack=_jsonable(
                        node.meta.get("fwd_source_fn_stack")
                    ),
                    fwd_nn_module_stack=_jsonable(
                        node.meta.get("fwd_nn_module_stack")
                    ),
                    original_aten=_jsonable(
                        node.meta.get("original_aten")
                    ),
                    from_node=_jsonable(node.meta.get("from_node")),
                    is_gradient_acc=bool(
                        node.meta.get("is_gradient_acc", False)
                    ),
                    partitioner_tag=_jsonable(
                        node.meta.get("partitioner_tag")
                    ),
                    stack_trace=node.meta.get("stack_trace"),
                )
            )
        code = graph_module.code
        self.graphs.append(
            AOTGraphDescriptor(
                phase=phase,
                graph_index=sum(
                    graph.phase == phase for graph in self.graphs
                ),
                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                nodes=tuple(descriptors),
                input_count=len(example_inputs),
            )
        )

    def compiler(
        self,
        phase: str,
    ) -> Callable[[torch.fx.GraphModule, Sequence[Any]], Callable[..., Any]]:
        from torch._dynamo.backends.debugging import boxed_nop

        def compile_graph(
            graph_module: torch.fx.GraphModule,
            example_inputs: Sequence[Any],
        ) -> Callable[..., Any]:
            graph_index = sum(
                graph.phase == phase for graph in self.graphs
            )
            self._record(phase, graph_module, example_inputs)
            tasks = [
                task
                for task in self.reference_cut_tasks
                if task["phase"] == phase
                and all(
                    str(node_id).startswith(
                        f"{phase.lower()}:graph{graph_index}:"
                    )
                    for node_id in task["aot_node_ids"]
                )
            ]
            graph_output_edges = [
                edge
                for node in graph_module.graph.nodes
                if node.op == "output"
                for edge in _input_edges(node)
                if edge["source_op"] == "call_function"
            ]
            requires_runtime_interpreter = bool(tasks) or phase in {
                "FORWARD",
                "BACKWARD",
            }
            if requires_runtime_interpreter:
                from scripts.fx_replay import (
                    ReferenceCutReplayInterpreter,
                    prepare_reference_cut_tasks,
                )

                if tasks:
                    extracted, certificates = prepare_reference_cut_tasks(
                        graph_module=graph_module,
                        phase=phase,
                        graph_index=graph_index,
                        tasks=tasks,
                    )
                else:
                    extracted, certificates = [], []
                self.reference_cut_extractions.extend(certificates)
                placeholders = [
                    node
                    for node in graph_module.graph.nodes
                    if node.op == "placeholder"
                ]

                def run(args: Any) -> Any:
                    if phase == "BACKWARD":
                        if self._active_bridge is None:
                            raise RuntimeError(
                                "AOT backward has no live forward bridge"
                            )
                        forward_outputs = self._active_bridge[
                            "forward_outputs"
                        ]
                        forward_inputs = self._active_bridge[
                            "forward_inputs"
                        ]
                        user_cotangents = self._active_bridge.get(
                            "_user_cotangents", []
                        )
                        backward_inputs = []
                        for placeholder, value in zip(
                            placeholders, args, strict=True
                        ):
                            output_matches = [
                                {
                                    "runtime_token": row[
                                        "runtime_token"
                                    ],
                                    "identity_mode": mode,
                                }
                                for row in forward_outputs
                                if (
                                    mode
                                    := self._runtime_identity_mode(
                                        row["_value"], value
                                    )
                                )
                            ]
                            input_matches = [
                                {
                                    "runtime_token": row[
                                        "runtime_token"
                                    ],
                                    "identity_mode": mode,
                                }
                                for row in forward_inputs
                                if (
                                    mode
                                    := self._runtime_identity_mode(
                                        row["_value"], value
                                    )
                                )
                            ]
                            cotangent_modes = [
                                mode
                                for item in user_cotangents
                                if (
                                    mode
                                    := self._runtime_identity_mode(
                                        item, value
                                    )
                                )
                            ]
                            backward_inputs.append(
                                {
                                    "placeholder": placeholder.name,
                                    "runtime_token": (
                                        f"backward-input:"
                                        f"{placeholder.name}"
                                    ),
                                    "forward_output_matches": output_matches,
                                    "forward_input_matches": input_matches,
                                    "user_cotangent_identity_modes": (
                                        cotangent_modes
                                    ),
                                    "_value": value,
                                }
                            )
                        self._active_bridge["backward_phase"] = {
                            "graph_index": graph_index
                        }
                        self._active_bridge[
                            "backward_inputs"
                        ] = backward_inputs
                    interpreter = ReferenceCutReplayInterpreter(
                        graph_module,
                        extracted,
                        extra_capture_names=(
                            [
                                edge["source_node"]
                                for edge in graph_output_edges
                            ]
                            if phase == "FORWARD"
                            else []
                        ),
                    )
                    result = interpreter.run(*args)
                    if phase == "FORWARD":
                        forward_inputs = [
                            {
                                "placeholder": placeholder.name,
                                "runtime_token": (
                                    "forward-input:"
                                    f"{placeholder.name}"
                                ),
                                "_value": value,
                            }
                            for placeholder, value in zip(
                                placeholders, args, strict=True
                            )
                        ]
                        forward_outputs = []
                        for output_index, edge in enumerate(
                            graph_output_edges
                        ):
                            value = interpreter.values[
                                edge["source_node"]
                            ]
                            forward_outputs.append(
                                {
                                    "source_node": edge["source_node"],
                                    "output_path": edge["argument_path"],
                                    "runtime_token": (
                                        "forward-output:"
                                        f"{output_index}:"
                                        f"{edge['source_node']}"
                                    ),
                                    "is_user_output": False,
                                    "forward_input_matches": [
                                        {
                                            "runtime_token": input_row[
                                                "runtime_token"
                                            ],
                                            "identity_mode": mode,
                                        }
                                        for input_row in forward_inputs
                                        if (
                                            mode
                                            := self._runtime_identity_mode(
                                                input_row["_value"], value
                                            )
                                        )
                                    ],
                                    "_value": value,
                                }
                            )
                        self._active_bridge = {
                            "run_index": len(
                                self.cross_phase_runtime_bridges
                            ),
                            "forward_phase": {
                                "graph_index": graph_index
                            },
                            "forward_inputs": forward_inputs,
                            "forward_outputs": forward_outputs,
                            "_user_cotangents": [],
                        }
                    else:
                        bridge = self._active_bridge
                        assert bridge is not None
                        forward_resolved = all(
                            row["is_user_output"]
                            or bool(row["forward_input_matches"])
                            or any(
                                any(
                                    match["runtime_token"]
                                    == row["runtime_token"]
                                    for match in backward[
                                        "forward_output_matches"
                                    ]
                                )
                                for backward in bridge[
                                    "backward_inputs"
                                ]
                            )
                            for row in bridge["forward_outputs"]
                        )
                        backward_resolved = all(
                            bool(row["forward_output_matches"])
                            or bool(row["forward_input_matches"])
                            or bool(
                                row["user_cotangent_identity_modes"]
                            )
                            for row in bridge["backward_inputs"]
                        )
                        bridge["gates"] = {
                            "every_forward_output_is_user_or_backward_input": (
                                forward_resolved
                            ),
                            "every_backward_input_is_forward_or_cotangent": (
                                backward_resolved
                            ),
                            "runtime_identity_only": True,
                            "storage_view_identity_allowed": True,
                            "name_shape_or_ordinal_pairing_used": False,
                        }
                        self.cross_phase_runtime_bridges.append(
                            self._public_bridge(bridge)
                        )
                        self._active_bridge = None
                    self.reference_cut_replay_runs.append(
                        {
                            "phase": phase,
                            "graph_index": graph_index,
                            "run_index": sum(
                                row["phase"] == phase
                                and row["graph_index"] == graph_index
                                for row in self.reference_cut_replay_runs
                            ),
                            "observations": interpreter.observations,
                        }
                    )
                    return result

                run._boxed_call = True  # type: ignore[attr-defined]
                return run
            return boxed_nop(graph_module, list(example_inputs))

        return compile_graph

    def backend(self) -> Callable[..., Any]:
        from torch._dynamo.backends.common import aot_autograd

        return aot_autograd(
            fw_compiler=self.compiler("FORWARD"),
            bw_compiler=self.compiler("BACKWARD"),
            keep_inference_input_mutations=True,
        )

    def as_dict(self) -> dict[str, Any]:
        phase_counts = {
            phase: sum(graph.phase == phase for graph in self.graphs)
            for phase in ("FORWARD", "BACKWARD")
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "phase_graph_counts": phase_counts,
            "graphs": [graph.as_dict() for graph in self.graphs],
            "reference_cut_runtime": {
                "task_count": len(self.reference_cut_tasks),
                "extraction_count": len(
                    self.reference_cut_extractions
                ),
                "replay_run_count": len(
                    self.reference_cut_replay_runs
                ),
                "extractions": self.reference_cut_extractions,
                "replay_runs": self.reference_cut_replay_runs,
                "gates": {
                    "all_tasks_extracted": (
                        len(self.reference_cut_extractions)
                        == len(self.reference_cut_tasks)
                    ),
                    "all_extracted_ports_exact": all(
                        row["graph_binding"]["port_routes_exact"]
                        for row in self.reference_cut_extractions
                    ),
                    "all_replays_bitwise_equal": all(
                        observation["bitwise_equal"]
                        for run in self.reference_cut_replay_runs
                        for observation in run["observations"]
                    ),
                    "all_replays_finite": all(
                        observation["all_finite"]
                        for run in self.reference_cut_replay_runs
                        for observation in run["observations"]
                    ),
                },
            },
            "cross_phase_runtime_bridge": {
                "run_count": len(self.cross_phase_runtime_bridges),
                "runs": self.cross_phase_runtime_bridges,
                "gates": {
                    "no_unfinished_forward_bridge": (
                        self._active_bridge is None
                    ),
                    "all_forward_outputs_resolved": all(
                        row["gates"][
                            "every_forward_output_is_user_or_backward_input"
                        ]
                        for row in self.cross_phase_runtime_bridges
                    ),
                    "all_backward_inputs_resolved": all(
                        row["gates"][
                            "every_backward_input_is_forward_or_cotangent"
                        ]
                        for row in self.cross_phase_runtime_bridges
                    ),
                    "identity_pairing_only": all(
                        row["gates"]["runtime_identity_only"]
                        and not row["gates"][
                            "name_shape_or_ordinal_pairing_used"
                        ]
                        for row in self.cross_phase_runtime_bridges
                    ),
                },
            },
            "claim_boundary": {
                "supported": (
                    "functionalized AOTAutograd forward/backward graph "
                    "structure captured at compile time, including exact "
                    "autograd seq_nr and forward-origin metadata"
                ),
                "not_yet_supported": [
                    "generated-kernel coverage",
                    "runtime candidate identity",
                    "eager-op to fused-region equivalence",
                ],
            },
        }
        payload["capture_sha256"] = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        return payload
