"""Execute extracted FX reference cuts on their actual AOT boundary values."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch.fx import GraphModule, Interpreter, Node

from scripts.fx_cut import (
    SCHEMA_VERSION as EXTRACTOR_SCHEMA_VERSION,
    extract_pure_fx_reference_cut,
)


SCHEMA_VERSION = "forkcert.fx-reference-cut-runtime-replay.v1"


def _route_key(
    *,
    source: str,
    consumer: str,
    argument_path: Iterable[Any],
) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(source),
        str(consumer),
        tuple(str(item) for item in argument_path),
    )


def _expected_input_routes(task: Mapping[str, Any]) -> Counter:
    return Counter(
        _route_key(
            source=edge["source_node"],
            consumer=str(edge["consumer_node_id"]).split(":", 2)[2],
            argument_path=edge["consumer_argument_path"],
        )
        for edge in task["expected_boundary_inputs"]
    )


def _actual_input_routes(certificate: Mapping[str, Any]) -> Counter:
    return Counter(
        _route_key(
            source=edge["source_node"],
            consumer=edge["consumer_node"],
            argument_path=edge["consumer_argument_path"],
        )
        for edge in certificate["boundary_input_routes"]
    )


def _expected_output_routes(task: Mapping[str, Any]) -> Counter:
    return Counter(
        _route_key(
            source=str(edge["source_node_id"]).split(":", 2)[2],
            consumer=str(edge["consumer_node_id"]).split(":", 2)[2],
            argument_path=edge["consumer_argument_path"],
        )
        for edge in task["expected_boundary_outputs"]
    )


def _actual_output_routes(certificate: Mapping[str, Any]) -> Counter:
    return Counter(
        _route_key(
            source=edge["source_node"],
            consumer=edge["consumer_node"],
            argument_path=edge["consumer_argument_path"],
        )
        for edge in certificate["boundary_output_routes"]
    )


def _tensor_comparison(actual: torch.Tensor, replay: torch.Tensor) -> dict[str, Any]:
    metadata_equal = (
        actual.shape == replay.shape
        and actual.dtype == replay.dtype
        and actual.device == replay.device
    )
    if not metadata_equal:
        return {
            "kind": "tensor",
            "metadata_equal": False,
            "bitwise_equal": False,
            "all_finite": False,
            "max_abs_error": None,
        }
    if actual.is_floating_point() or actual.is_complex():
        # Some terminal gradients are several GiB.  A whole-tensor subtraction
        # creates another tensor of that size and can make an otherwise valid
        # full-step replay OOM.  Flattening is a view and chunking preserves the
        # exact all-finite/max/bitwise predicates with bounded temporary memory.
        chunk_elements = 1_048_576
        actual_flat = actual.reshape(-1)
        replay_flat = replay.reshape(-1)
        finite = True
        bitwise_equal = True
        maximum = 0.0
        for start in range(0, actual.numel(), chunk_elements):
            left = actual_flat[start : start + chunk_elements]
            right = replay_flat[start : start + chunk_elements]
            finite = finite and bool(
                torch.isfinite(left).all().item()
                and torch.isfinite(right).all().item()
            )
            bitwise_equal = bitwise_equal and bool(torch.equal(left, right))
            chunk_maximum = float((right - left).abs().max().item())
            # Preserve the original whole-tensor reduction's NaN propagation
            # (for example inf - inf), rather than letting Python ``max`` mask it.
            if chunk_maximum != chunk_maximum:
                maximum = chunk_maximum
            elif maximum == maximum:
                maximum = max(maximum, chunk_maximum)
    else:
        finite = True
        bitwise_equal = bool(torch.equal(actual, replay))
        maximum = 0.0 if torch.equal(actual, replay) else None
    return {
        "kind": "tensor",
        "metadata_equal": True,
        "bitwise_equal": bitwise_equal,
        "all_finite": finite,
        "max_abs_error": maximum,
    }


def _compare_values(actual: Any, replay: Any) -> dict[str, Any]:
    if isinstance(actual, torch.Tensor) and isinstance(replay, torch.Tensor):
        return _tensor_comparison(actual, replay)
    if isinstance(actual, (tuple, list)) and isinstance(
        replay, (tuple, list)
    ):
        children = [
            _compare_values(left, right)
            for left, right in zip(actual, replay)
        ]
        same_length = len(actual) == len(replay)
        return {
            "kind": "sequence",
            "same_length": same_length,
            "children": children,
            "bitwise_equal": (
                same_length
                and all(child["bitwise_equal"] for child in children)
            ),
            "all_finite": (
                same_length
                and all(child["all_finite"] for child in children)
            ),
        }
    equal = type(actual) is type(replay) and actual == replay
    return {
        "kind": "scalar",
        "bitwise_equal": bool(equal),
        "all_finite": True,
    }


@dataclass(frozen=True)
class ExtractedReplayTask:
    task_id: str
    cut_id: str
    module: GraphModule
    certificate: dict[str, Any]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    trigger_ordinal: int


def prepare_reference_cut_tasks(
    *,
    graph_module: GraphModule,
    phase: str,
    graph_index: int,
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[list[ExtractedReplayTask], list[dict[str, Any]]]:
    """Extract and port-check every task assigned to one exact AOT graph."""

    by_name = {node.name: node for node in graph_module.graph.nodes}
    ordinal = {
        node.name: index
        for index, node in enumerate(graph_module.graph.nodes)
    }
    extracted_tasks = []
    certificates = []
    for task in tasks:
        if task["phase"] != phase:
            raise ValueError("reference cut task phase does not match graph")
        node_ids = [str(item) for item in task["aot_node_ids"]]
        expected_prefix = f"{phase.lower()}:graph{graph_index}:"
        if not all(item.startswith(expected_prefix) for item in node_ids):
            raise ValueError("reference cut task graph identity mismatch")
        module, certificate = extract_pure_fx_reference_cut(
            graph_module=graph_module,
            cut_node_names=task["aot_node_names"],
            cut_id=task["cut_id"],
            node_index=by_name,
            node_ordinal=ordinal,
        )
        if task["required_extractor_schema"] != EXTRACTOR_SCHEMA_VERSION:
            raise ValueError("reference cut extractor schema mismatch")
        input_routes_equal = (
            _expected_input_routes(task)
            == _actual_input_routes(certificate)
        )
        output_routes_equal = (
            _expected_output_routes(task)
            == _actual_output_routes(certificate)
        )
        if not input_routes_equal or not output_routes_equal:
            raise ValueError(
                f"reference cut port identity mismatch: {task['cut_id']} "
                f"expected_inputs={dict(_expected_input_routes(task))} "
                f"actual_inputs={dict(_actual_input_routes(certificate))} "
                f"expected_outputs={dict(_expected_output_routes(task))} "
                f"actual_outputs={dict(_actual_output_routes(certificate))}"
            )
        actual_graph_hash = hashlib.sha256(
            graph_module.code.encode()
        ).hexdigest()
        expected_graph_hash = str(task["expected_graph_code_sha256"])
        if actual_graph_hash != expected_graph_hash:
            raise ValueError(
                "reference cut AOT graph hash mismatch after exact port match: "
                f"{task['cut_id']} expected={expected_graph_hash} "
                f"actual={actual_graph_hash}"
            )
        input_names = tuple(
            row["name"] for row in certificate["boundary_inputs"]
        )
        output_names = tuple(
            row["name"] for row in certificate["boundary_outputs"]
        )
        if not all(name in by_name for name in (*input_names, *output_names)):
            raise ValueError("reference cut boundary node is absent")
        certificate = dict(certificate)
        certificate["graph_binding"] = {
            "phase": phase,
            "graph_index": graph_index,
            "graph_code_sha256": actual_graph_hash,
            "port_routes_exact": True,
        }
        certificates.append(certificate)
        extracted_tasks.append(
            ExtractedReplayTask(
                task_id=str(task["task_id"]),
                cut_id=str(task["cut_id"]),
                module=module,
                certificate=certificate,
                input_names=input_names,
                output_names=output_names,
                trigger_ordinal=max(ordinal[name] for name in output_names),
            )
        )
    return extracted_tasks, certificates


class ReferenceCutReplayInterpreter(Interpreter):
    """Capture only required values and replay each cut as soon as it closes."""

    def __init__(
        self,
        module: GraphModule,
        tasks: Sequence[ExtractedReplayTask],
        *,
        extra_capture_names: Iterable[str] = (),
        value_sink: Callable[[ExtractedReplayTask, tuple[Any, ...]], None] | None = None,
        verify_replay: bool = True,
    ) -> None:
        super().__init__(module)
        self.tasks = tuple(tasks)
        self.verify_replay = verify_replay
        self.required_names = {
            name
            for task in self.tasks
            for name in (
                (*task.input_names, *task.output_names)
                if verify_replay
                else task.output_names
            )
        } | {str(name) for name in extra_capture_names}
        self.values: dict[str, Any] = {}
        self.pending = set(range(len(self.tasks)))
        self.ready_by_ordinal: dict[int, tuple[int, ...]] = {}
        for index, task in enumerate(self.tasks):
            self.ready_by_ordinal.setdefault(task.trigger_ordinal, ())
            self.ready_by_ordinal[task.trigger_ordinal] += (index,)
        self.observations: list[dict[str, Any]] = []
        self.ordinal = -1
        self.value_sink = value_sink

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        self.ordinal += 1
        if node.name in self.required_names:
            self.values[node.name] = result
        # Tasks are assigned the ordinal of their latest boundary output.  An
        # ordinal-indexed schedule avoids scanning every pending cut at every
        # FX node (O(nodes * cuts)) while preserving the exact trigger point.
        ready = self.ready_by_ordinal.get(self.ordinal, ())
        for index in ready:
            if index not in self.pending:
                raise RuntimeError("reference cut scheduled more than once")
            task = self.tasks[index]
            required = (
                (*task.input_names, *task.output_names)
                if self.verify_replay
                else task.output_names
            )
            missing = [name for name in required if name not in self.values]
            if missing:
                raise RuntimeError(
                    f"reference cut runtime values are absent: {missing}"
                )
            actual = tuple(self.values[name] for name in task.output_names)
            if self.value_sink is not None:
                self.value_sink(task, actual)
            if self.verify_replay:
                replay = task.module(
                    *(self.values[name] for name in task.input_names)
                )
                if not isinstance(replay, tuple):
                    raise RuntimeError("reference cut violated tuple output contract")
                comparison = _compare_values(actual, replay)
                self.observations.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task_id": task.task_id,
                        "cut_id": task.cut_id,
                        "bitwise_equal": comparison["bitwise_equal"],
                        "all_finite": comparison["all_finite"],
                        "comparison": comparison,
                    }
                )
            self.pending.remove(index)
        return result

    def run(self, *args: Any, **kwargs: Any) -> Any:
        result = super().run(*args, **kwargs)
        if self.pending:
            missing = [
                self.tasks[index].cut_id
                for index in sorted(self.pending)
            ]
            raise RuntimeError(f"reference cuts were not replayed: {missing}")
        return result
