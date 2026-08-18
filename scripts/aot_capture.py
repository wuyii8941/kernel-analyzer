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
import weakref
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
        reference_value_sink: Callable[[Any, tuple[Any, ...]], None] | None = None,
    ) -> None:
        self.graphs: list[AOTGraphDescriptor] = []
        self.reference_cut_tasks = tuple(
            dict(task) for task in reference_cut_tasks
        )
        self.reference_value_sink = reference_value_sink
        self.reference_cut_extractions: list[dict[str, Any]] = []
        self.reference_cut_replay_runs: list[dict[str, Any]] = []
        self._verified_reference_cut_task_ids: set[str] = set()
        self.cross_phase_runtime_bridges: list[dict[str, Any]] = []
        self.segmented_forward_only_bridges: list[dict[str, Any]] = []
        self._active_bridges: list[dict[str, Any]] = []
        # Weak references let segmented backward graphs recover saved tensors
        # produced by any earlier forward graph without retaining additional
        # CUDA storage.  Dead tensors simply cannot form a proof edge.
        self._global_forward_runtime_values: list[dict[str, Any]] = []

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

        if not self._active_bridges:
            raise RuntimeError("no active AOT forward bridge")
        user_values = self._tensor_leaves(value)
        for row in self._active_bridges[-1]["forward_outputs"]:
            row["is_user_output"] = any(
                self._runtime_identity_mode(row["_value"], item)
                is not None
                for item in user_values
            )

    def bind_user_cotangent(self, value: Any) -> Any:
        """Record the exact cotangent object supplied to the AOT backward."""

        if not self._active_bridges:
            raise RuntimeError("no active AOT forward bridge")
        self._active_bridges[-1]["_user_cotangents"] = self._tensor_leaves(value)
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

    @staticmethod
    def _forward_output_resolved(row: Mapping[str, Any]) -> bool:
        return bool(
            row.get("is_user_output")
            or row.get("forward_input_matches")
            or row.get("downstream_forward_input_matches")
            or row.get("backward_input_matches")
            or row.get("non_differentiable_terminal_boundary")
        )

    @classmethod
    def _bridge_forward_outputs_resolved(
        cls, bridge: Mapping[str, Any]
    ) -> bool:
        return all(
            cls._forward_output_resolved(row)
            for row in bridge["forward_outputs"]
        )

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
                    if phase == "FORWARD" and graph_index == 0 and self._active_bridges:
                        # A new segmented forward starts only after the prior
                        # full forward/backward invocation returned.  Graph
                        # breaks can leave forward-only segments with no AOT
                        # backward graph of their own; their outputs must have
                        # been consumed by a later forward segment or backward
                        # input via runtime identity before they are retired.
                        if not all(
                            self._bridge_forward_outputs_resolved(bridge)
                            for bridge in self._active_bridges
                        ):
                            raise RuntimeError(
                                "segmented forward-only bridge has unresolved output"
                            )
                        self.segmented_forward_only_bridges.extend(
                            self._public_bridge(bridge)
                            for bridge in self._active_bridges
                        )
                        self._active_bridges.clear()
                        self._global_forward_runtime_values = [
                            row for row in self._global_forward_runtime_values
                            if row["_weak_value"]() is not None
                        ]
                    if phase == "BACKWARD":
                        if not self._active_bridges:
                            raise RuntimeError(
                                "AOT backward has no live forward bridge"
                            )
                        active_bridge = self._active_bridges[-1]
                        forward_outputs = active_bridge[
                            "forward_outputs"
                        ]
                        forward_inputs = active_bridge[
                            "forward_inputs"
                        ]
                        user_cotangents = active_bridge.get(
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
                            global_forward_matches = []
                            for global_row in self._global_forward_runtime_values:
                                prior = global_row["_weak_value"]()
                                if prior is None:
                                    continue
                                mode = self._runtime_identity_mode(prior, value)
                                if mode is not None:
                                    global_forward_matches.append({
                                        "phase_graph_index": global_row["phase_graph_index"],
                                        "value_kind": global_row["value_kind"],
                                        "source_node": global_row["source_node"],
                                        "runtime_token": global_row["runtime_token"],
                                        "identity_mode": mode,
                                    })
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
                                    "global_forward_matches": global_forward_matches,
                                    "aot_partitioner_tag": _jsonable(
                                        placeholder.meta.get("partitioner_tag")
                                    ),
                                    "compiler_declared_cotangent": (
                                        placeholder.meta.get("partitioner_tag")
                                        == "is_backward"
                                    ),
                                    "_value": value,
                                }
                            )
                        for backward_row in backward_inputs:
                            for match in backward_row["global_forward_matches"]:
                                if match["value_kind"] != "FORWARD_OUTPUT":
                                    continue
                                for prior_bridge in self._active_bridges:
                                    if (
                                        prior_bridge["forward_phase"]["graph_index"]
                                        != match["phase_graph_index"]
                                    ):
                                        continue
                                    for output_row in prior_bridge["forward_outputs"]:
                                        if output_row["source_node"] == match["source_node"]:
                                            output_row.setdefault(
                                                "backward_input_matches", []
                                            ).append({
                                                "backward_graph_index": graph_index,
                                                "backward_placeholder": backward_row["placeholder"],
                                                "identity_mode": match["identity_mode"],
                                            })
                        active_bridge["backward_phase"] = {
                            "graph_index": graph_index
                        }
                        active_bridge[
                            "backward_inputs"
                        ] = backward_inputs
                    verify_replay = any(
                        task.task_id not in self._verified_reference_cut_task_ids
                        for task in extracted
                    )
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
                        value_sink=self.reference_value_sink,
                        verify_replay=verify_replay,
                    )
                    result = interpreter.run(*args)
                    self._verified_reference_cut_task_ids.update(
                        observation["task_id"]
                        for observation in interpreter.observations
                    )
                    if phase == "FORWARD":
                        forward_inputs = []
                        for placeholder, value in zip(placeholders, args, strict=True):
                            global_matches = []
                            for global_row in self._global_forward_runtime_values:
                                prior = global_row["_weak_value"]()
                                if prior is None:
                                    continue
                                mode = self._runtime_identity_mode(prior, value)
                                if mode is not None:
                                    global_matches.append({
                                        "phase_graph_index": global_row["phase_graph_index"],
                                        "value_kind": global_row["value_kind"],
                                        "source_node": global_row["source_node"],
                                        "runtime_token": global_row["runtime_token"],
                                        "identity_mode": mode,
                                    })
                            input_row = {
                                "placeholder": placeholder.name,
                                "runtime_token": f"forward-input:{placeholder.name}",
                                "global_forward_matches": global_matches,
                                "_value": value,
                            }
                            forward_inputs.append(input_row)
                            for match in global_matches:
                                if match["value_kind"] != "FORWARD_OUTPUT":
                                    continue
                                for prior_bridge in self._active_bridges:
                                    if (
                                        prior_bridge["forward_phase"]["graph_index"]
                                        != match["phase_graph_index"]
                                    ):
                                        continue
                                    for output_row in prior_bridge["forward_outputs"]:
                                        if output_row["source_node"] == match["source_node"]:
                                            output_row.setdefault(
                                                "downstream_forward_input_matches", []
                                            ).append({
                                                "forward_graph_index": graph_index,
                                                "forward_placeholder": placeholder.name,
                                                "identity_mode": match["identity_mode"],
                                            })
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
                                    "downstream_forward_input_matches": [],
                                    "backward_input_matches": [],
                                    "runtime_value_kind": (
                                        "TENSOR"
                                        if isinstance(value, torch.Tensor)
                                        else type(value).__name__
                                    ),
                                    "runtime_dtype": (
                                        str(value.dtype)
                                        if isinstance(value, torch.Tensor)
                                        else None
                                    ),
                                    "non_differentiable_terminal_boundary": (
                                        not isinstance(value, torch.Tensor)
                                        or not (
                                            value.is_floating_point()
                                            or value.is_complex()
                                        )
                                    ),
                                    "terminal_boundary_classification_uses_name_or_shape": False,
                                    "_value": value,
                                }
                            )
                        self._active_bridges.append({
                            "run_index": len(
                                self.cross_phase_runtime_bridges
                            ),
                            "forward_phase": {
                                "graph_index": graph_index
                            },
                            "forward_inputs": forward_inputs,
                            "forward_outputs": forward_outputs,
                            "_user_cotangents": [],
                        })
                        for row in forward_inputs:
                            self._global_forward_runtime_values.append({
                                "phase_graph_index": graph_index,
                                "value_kind": "FORWARD_INPUT",
                                "source_node": row["placeholder"],
                                "runtime_token": (
                                    f"forward-g{graph_index}-input:{row['placeholder']}"
                                ),
                                "_weak_value": weakref.ref(row["_value"]),
                            })
                        for row in forward_outputs:
                            self._global_forward_runtime_values.append({
                                "phase_graph_index": graph_index,
                                "value_kind": "FORWARD_OUTPUT",
                                "source_node": row["source_node"],
                                "runtime_token": (
                                    f"forward-g{graph_index}-output:{row['source_node']}"
                                ),
                                "_weak_value": weakref.ref(row["_value"]),
                            })
                    else:
                        bridge = self._active_bridges[-1]
                        forward_resolved = self._bridge_forward_outputs_resolved(
                            bridge
                        )
                        backward_resolved = all(
                            bool(row["forward_output_matches"])
                            or bool(row["forward_input_matches"])
                            or bool(row["global_forward_matches"])
                            or bool(
                                row["user_cotangent_identity_modes"]
                            )
                            or bool(row["compiler_declared_cotangent"])
                            for row in bridge["backward_inputs"]
                        )
                        bridge["gates"] = {
                            "every_forward_output_is_user_or_backward_input": (
                                forward_resolved
                            ),
                            "every_backward_input_is_forward_or_cotangent": (
                                backward_resolved
                            ),
                            "global_forward_runtime_registry_uses_weak_references": True,
                            "compiler_cotangent_classification_uses_partitioner_tag": True,
                            "runtime_identity_only": True,
                            "storage_view_identity_allowed": True,
                            "name_shape_or_ordinal_pairing_used": False,
                        }
                        self.cross_phase_runtime_bridges.append(
                            self._public_bridge(bridge)
                        )
                        self._active_bridges.pop()
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

    def backend(
        self,
        *,
        decompositions: Mapping[Any, Callable[..., Any]] | None = None,
    ) -> Callable[..., Any]:
        from torch._dynamo.backends.common import aot_autograd

        options = {
            "fw_compiler": self.compiler("FORWARD"),
            "bw_compiler": self.compiler("BACKWARD"),
            "keep_inference_input_mutations": True,
        }
        if decompositions is not None:
            options["decompositions"] = decompositions
        return aot_autograd(
            **options,
        )

    def inductor_partition_backend(self) -> Callable[..., Any]:
        """Capture/replay the exact post-AOT graphs used by ``compile_fx``.

        ``backend()`` installs a fresh AOTAutograd partitioner.  That is useful
        for canonical eager/AOT proofs, but it is not an identity-preserving
        reference for an Inductor candidate: ``compile_fx`` may select a
        different decomposition/partition and consequently reuse the same FX
        node names for unrelated values.  This backend follows Inductor's
        normal partition path and replaces only its inner compiler with the
        boxed reference interpreter.
        """

        from torch._inductor.compile_fx import compile_fx

        def inner_compile(
            graph_module: torch.fx.GraphModule,
            example_inputs: Sequence[Any],
            compile_region_name: str | None = None,
            **kwargs: Any,
        ) -> Callable[..., Any]:
            del compile_region_name
            phase = "BACKWARD" if kwargs.get("is_backward", False) else "FORWARD"
            return self.compiler(phase)(graph_module, example_inputs)

        def backend(
            graph_module: torch.fx.GraphModule,
            example_inputs: Sequence[Any],
        ) -> Callable[..., Any]:
            return compile_fx(
                graph_module,
                list(example_inputs),
                inner_compile=inner_compile,
            )

        return backend

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
                    "all_tasks_replayed_at_least_once": (
                        self._verified_reference_cut_task_ids
                        == {
                            str(task["task_id"])
                            for task in self.reference_cut_tasks
                        }
                    ),
                },
            },
            "cross_phase_runtime_bridge": {
                "run_count": len(self.cross_phase_runtime_bridges),
                "runs": self.cross_phase_runtime_bridges,
                "segmented_forward_only_run_count": len(
                    self.segmented_forward_only_bridges
                ),
                "segmented_forward_only_runs": self.segmented_forward_only_bridges,
                "active_forward_only_run_count": len(self._active_bridges),
                "active_forward_only_runs": [
                    self._public_bridge(bridge)
                    for bridge in self._active_bridges
                ],
                "gates": {
                    "no_unfinished_forward_bridge": (
                        all(
                            self._bridge_forward_outputs_resolved(bridge)
                            for bridge in self._active_bridges
                        )
                    ),
                    "all_forward_outputs_resolved": all(
                        row["gates"][
                            "every_forward_output_is_user_or_backward_input"
                        ]
                        for row in self.cross_phase_runtime_bridges
                    ) and all(
                        self._bridge_forward_outputs_resolved(bridge)
                        for bridge in self.segmented_forward_only_bridges
                    ) and all(
                        self._bridge_forward_outputs_resolved(bridge)
                        for bridge in self._active_bridges
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
