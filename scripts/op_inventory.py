"""Complete metadata-only ATen census for one model forward/backward step.

The inventory denominator is execution-derived: every dispatcher invocation
that occurs while evaluating the loss or its backward pass is recorded.  No
tensor payload is cloned, so the observer can be used on a real language-model
step without multiplying activation memory.

This module establishes coverage, phase, lineage and semantic context.  It
does not by itself align an eager invocation to a compiled/fused region and
does not issue a numerical-correctness verdict.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence
import weakref

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _disable_current_modes,
)


SCHEMA_VERSION = "forkcert.full-step-operator-inventory.v4"
_DISPATCH_PROFILER_MARKER_PREFIX = "forkcert_dispatch_event::"
_NO_SEQUENCE_NR = -1
_ACCUMULATE_GRAD_SEQUENCE_NR = (1 << 64) - 1


class StepPhase(str, Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"


class OperatorCategory(str, Enum):
    MATRIX = "MATRIX"
    REDUCTION = "REDUCTION"
    ELEMENTWISE = "ELEMENTWISE"
    VIEW_LAYOUT = "VIEW_LAYOUT"
    MEMORY_COPY = "MEMORY_COPY"
    MUTATION = "MUTATION"
    RANDOM = "RANDOM"
    COLLECTIVE = "COLLECTIVE"
    OTHER = "OTHER"


def _shape(value: torch.Tensor) -> tuple[str, ...]:
    return tuple(str(item) for item in value.shape)


def _stride(value: torch.Tensor) -> tuple[str, ...]:
    try:
        return tuple(str(item) for item in value.stride())
    except RuntimeError:
        return ()


def _storage_key(value: torch.Tensor) -> tuple[str, int] | None:
    try:
        return str(value.device), int(value.untyped_storage().data_ptr())
    except (RuntimeError, TypeError, AttributeError):
        return None


def _version(value: torch.Tensor) -> int | None:
    try:
        return int(value._version)
    except RuntimeError:
        return None


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    leaves, _ = _pytree.tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _current_autograd_node_info() -> tuple[str | None, int | None]:
    try:
        node = torch._C._current_autograd_node()
    except (AttributeError, RuntimeError):
        return None, None
    if node is None:
        return None, None
    try:
        name = str(node.name())
    except (AttributeError, RuntimeError):
        name = type(node).__name__
    try:
        sequence_nr = int(node._sequence_nr())
    except (AttributeError, RuntimeError, TypeError):
        sequence_nr = None
    if sequence_nr == _ACCUMULATE_GRAD_SEQUENCE_NR:
        sequence_nr = None
    return name, sequence_nr


def _profiler_operator_name(overload: str) -> str:
    namespace, qualified = overload.split(".", 1)
    operator = qualified.split(".", 1)[0]
    return f"{namespace}::{operator}"


def _dispatch_profiler_marker(ordinal: int) -> str:
    return f"{_DISPATCH_PROFILER_MARKER_PREFIX}{ordinal}"


def _bind_forward_autograd_sequence_numbers(
    events: Sequence["FullStepOperatorEvent"],
    profiler_events: Sequence[Any],
) -> tuple["FullStepOperatorEvent", ...]:
    """Audit each dispatch boundary against its enclosing profiler operator.

    The explicit ordinal marker is emitted inside ``__torch_dispatch__``.
    Its direct parent is the Python dispatch-mode callback record, whose
    direct parent is the dispatcher/autograd boundary that invoked that exact
    callback.  This parent relation identifies the boundary without using an
    overload name, shape, module, family or execution-order heuristic.
    Profiler display names are retained only as audit metadata because some
    exact ATen boundaries are displayed as ``detach`` rather than
    ``aten::detach``.  A positive profiler sequence counter does not itself
    prove that an invocation created a node; the exact origin is read from
    the concrete output object's grad_fn after forward returns.
    """

    markers: dict[int, list[Any]] = {}
    for profiler_event in profiler_events:
        name = str(profiler_event.name)
        if not name.startswith(_DISPATCH_PROFILER_MARKER_PREFIX):
            continue
        suffix = name[len(_DISPATCH_PROFILER_MARKER_PREFIX) :]
        try:
            ordinal = int(suffix)
        except ValueError as error:
            raise RuntimeError(
                f"invalid dispatch profiler marker: {name}"
            ) from error
        markers.setdefault(ordinal, []).append(profiler_event)

    expected_ordinals = set(range(len(events)))
    observed_ordinals = set(markers)
    if observed_ordinals != expected_ordinals:
        missing = sorted(expected_ordinals - observed_ordinals)
        extra = sorted(observed_ordinals - expected_ordinals)
        raise RuntimeError(
            "dispatch profiler marker coverage mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    outer_boundary_ordinal_by_event_id: dict[int, int] = {}
    for ordinal, event_markers in markers.items():
        if len(event_markers) != 1:
            continue
        dispatch_mode_record = event_markers[0].cpu_parent
        if (
            dispatch_mode_record is not None
            and str(dispatch_mode_record.name) == "PythonDispatchMode"
            and dispatch_mode_record.cpu_parent is not None
        ):
            outer = dispatch_mode_record.cpu_parent
            outer_id = id(outer)
            previous = outer_boundary_ordinal_by_event_id.get(outer_id)
            if previous is not None and previous != ordinal:
                raise RuntimeError(
                    "one profiler dispatcher boundary owns multiple direct "
                    f"dispatch markers: {previous}, {ordinal}"
                )
            outer_boundary_ordinal_by_event_id[outer_id] = ordinal

    enriched = []
    for event in events:
        event_markers = markers[event.ordinal]
        if len(event_markers) != 1:
            raise RuntimeError(
                "dispatch profiler marker is not unique for ordinal "
                f"{event.ordinal}: {len(event_markers)}"
            )
        marker = event_markers[0]
        enclosing_dispatch_invocation_id = None
        dispatch_mode_record = marker.cpu_parent
        if (
            dispatch_mode_record is not None
            and str(dispatch_mode_record.name) != "PythonDispatchMode"
        ):
            enclosing_ordinal = (
                outer_boundary_ordinal_by_event_id.get(
                    id(dispatch_mode_record)
                )
            )
            if enclosing_ordinal is not None:
                enclosing_phase = events[enclosing_ordinal].phase.value
                enclosing_dispatch_invocation_id = (
                    f"{enclosing_phase.lower()}:{enclosing_ordinal}"
                )
        if (
            dispatch_mode_record is None
            or str(dispatch_mode_record.name) != "PythonDispatchMode"
        ):
            parent_name = (
                str(dispatch_mode_record.name)
                if dispatch_mode_record is not None
                else None
            )
            audit_fields = {
                "profiler_dispatch_ancestor_depth": 0,
                "profiler_dispatch_boundary_name": parent_name,
                "profiler_boundary_display_name_matches_overload": False,
                "enclosing_dispatch_invocation_id": (
                    enclosing_dispatch_invocation_id
                ),
            }
            if event.phase is StepPhase.FORWARD:
                has_output_grad_fn = any(
                    value is not None
                    for value in event.forward_output_autograd_sequence_nrs
                )
                has_unobserved_output = any(
                    value
                    == "COLLECTED_BEFORE_FORWARD_ORIGIN_BINDING"
                    for value in event.forward_output_observation_statuses
                )
                enriched.append(
                    replace(
                        event,
                        sequence_binding_status=(
                            "EXACT_NESTED_FORWARD_OUTPUT_GRAD_FN"
                            if has_output_grad_fn
                            else
                            "UNRESOLVED_NESTED_FORWARD_OUTPUT_COLLECTED"
                            if has_unobserved_output
                            else
                            "EXACT_NESTED_FORWARD_NO_INDEPENDENT_"
                            "AUTOGRAD_BOUNDARY"
                        ),
                        **audit_fields,
                    )
                )
            elif event.backward_autograd_sequence_nr is not None:
                enriched.append(
                    replace(
                        event,
                        sequence_binding_status=(
                            "EXACT_CURRENT_BACKWARD_NODE"
                        ),
                        **audit_fields,
                    )
                )
            else:
                enriched.append(
                    replace(
                        event,
                        sequence_binding_status=(
                            "EXACT_BACKWARD_AUXILIARY_OR_ACCUMULATE_GRAD"
                        ),
                        **audit_fields,
                    )
                )
            continue
        outer = dispatch_mode_record.cpu_parent
        if outer is None:
            raise RuntimeError(
                "dispatch marker has no enclosing dispatcher boundary for "
                f"{event.invocation_id} {event.overload}"
            )
        expected_name = _profiler_operator_name(event.overload)
        profiler_name = str(outer.name)
        expected_unqualified_name = expected_name.split("::", 1)[-1]
        profiler_sequence_nr = int(outer.sequence_nr)
        audit_fields = {
            "profiler_dispatch_ancestor_depth": 1,
            "profiler_dispatch_boundary_name": profiler_name,
            "profiler_boundary_display_name_matches_overload": (
                profiler_name
                in {expected_name, expected_unqualified_name}
            ),
            "enclosing_dispatch_invocation_id": (
                enclosing_dispatch_invocation_id
            ),
        }
        if event.phase is StepPhase.FORWARD:
            output_sequences = {
                int(value)
                for value in event.forward_output_autograd_sequence_nrs
                if value is not None
            }
            has_unobserved_output = any(
                value == "COLLECTED_BEFORE_FORWARD_ORIGIN_BINDING"
                for value in event.forward_output_observation_statuses
            )
            if profiler_sequence_nr >= 0:
                if output_sequences and output_sequences != {
                    profiler_sequence_nr
                }:
                    # Composite/nested execution can expose a profiler
                    # boundary whose sequence counter belongs to the outer
                    # autograd node while the dispatched tensor carries the
                    # inner node's exact grad_fn sequence.  Neither identity
                    # may be substituted for the other.  Keep the invocation
                    # in the denominator and fail this binding closed instead
                    # of aborting the complete-step inventory.
                    enriched.append(
                        replace(
                            event,
                            sequence_binding_status=(
                                "UNRESOLVED_PROFILER_COUNTER_OUTPUT_"
                                "SEQUENCE_DISAGREEMENT"
                            ),
                            **audit_fields,
                        )
                    )
                    continue
                enriched.append(
                    replace(
                        event,
                        forward_autograd_sequence_nr=profiler_sequence_nr,
                        sequence_binding_status=(
                            "EXACT_FORWARD_OUTPUT_GRAD_FN"
                            if output_sequences
                            else
                            "UNRESOLVED_PROFILER_COUNTER_OUTPUT_COLLECTED"
                            if has_unobserved_output
                            else
                            "PROFILER_COUNTER_ONLY_NO_FORWARD_OUTPUT_"
                            "GRAD_FN"
                        ),
                        **audit_fields,
                    )
                )
            else:
                if output_sequences:
                    raise RuntimeError(
                        "forward output has grad_fn but profiler boundary "
                        f"has no sequence at {event.invocation_id}"
                    )
                enriched.append(
                    replace(
                        event,
                        sequence_binding_status=(
                            "EXACT_NO_FORWARD_AUTOGRAD_NODE_"
                            "PROFILER_NEGATIVE"
                        ),
                        **audit_fields,
                    )
                )
            continue
        if event.backward_autograd_sequence_nr is not None:
            enriched.append(
                replace(
                    event,
                    sequence_binding_status="EXACT_CURRENT_BACKWARD_NODE",
                    **audit_fields,
                )
            )
        else:
            enriched.append(
                replace(
                    event,
                    sequence_binding_status=(
                        "EXACT_BACKWARD_AUXILIARY_OR_ACCUMULATE_GRAD"
                    ),
                    **audit_fields,
                )
            )
    return tuple(enriched)


def _schema_writes(func: Any) -> tuple[int, ...]:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return ()
    result = []
    for index, argument in enumerate(schema.arguments):
        alias = argument.alias_info
        if alias is not None and alias.is_write:
            result.append(index)
    return tuple(result)


def _non_tensor_value(value: Any) -> tuple[str, Any]:
    """Return a JSON-safe scalar/control value without tensor payloads."""

    if value is None:
        return "NONE", None
    if isinstance(value, bool):
        return "BOOL", value
    if isinstance(value, int):
        return "INT", value
    if isinstance(value, float):
        return "FLOAT", value
    if isinstance(value, complex):
        return "COMPLEX", {
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, str):
        return "STRING", value
    if isinstance(value, (list, tuple)):
        encoded = [_non_tensor_value(item) for item in value]
        if any(kind == "UNSUPPORTED" for kind, _ in encoded):
            return "UNSUPPORTED", type(value).__name__
        return (
            "TUPLE" if isinstance(value, tuple) else "LIST",
            [
                {"value_type": kind, "value": item}
                for kind, item in encoded
            ],
        )
    if isinstance(
        value,
        (
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return type(value).__name__.upper(), str(value)
    # SymInt/SymFloat/SymBool values are control metadata, not tensor payloads.
    if type(value).__module__.startswith("torch") and type(value).__name__ in {
        "SymInt",
        "SymFloat",
        "SymBool",
    }:
        return type(value).__name__.upper(), str(value)
    return "UNSUPPORTED", f"{type(value).__module__}.{type(value).__name__}"


@dataclass(frozen=True)
class OperatorArgumentBinding:
    name: str
    schema_type: str
    source: str
    tensor_input_indices: tuple[int, ...]
    value_type: str | None
    value: Any


def _argument_bindings(
    func: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    input_tensors: Sequence[torch.Tensor],
) -> tuple[OperatorArgumentBinding, ...]:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return ()

    available_by_object: dict[int, list[int]] = {}
    for index, tensor in enumerate(input_tensors):
        available_by_object.setdefault(id(tensor), []).append(index)
    consumed_by_object: Counter[int] = Counter()

    rows = []
    for index, argument in enumerate(schema.arguments):
        if index < len(args):
            value = args[index]
            source = "EXPLICIT_POSITIONAL"
        elif argument.name in kwargs:
            value = kwargs[argument.name]
            source = "EXPLICIT_KEYWORD"
        elif argument.has_default_value():
            value = argument.default_value
            source = "SCHEMA_DEFAULT"
        else:
            rows.append(
                OperatorArgumentBinding(
                    name=str(argument.name),
                    schema_type=str(argument.type),
                    source="MISSING_REQUIRED_ARGUMENT",
                    tensor_input_indices=(),
                    value_type=None,
                    value=None,
                )
            )
            continue

        tensor_indices = []
        for tensor in _tensor_leaves(value):
            object_id = id(tensor)
            positions = available_by_object.get(object_id, ())
            occurrence = consumed_by_object[object_id]
            if occurrence >= len(positions):
                continue
            tensor_indices.append(positions[occurrence])
            consumed_by_object[object_id] += 1
        if tensor_indices:
            if isinstance(value, (list, tuple)):
                position_iter = iter(tensor_indices)
                structured = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        structured.append(
                            {
                                "value_type": "TENSOR_INPUT_INDEX",
                                "value": next(position_iter),
                            }
                        )
                    else:
                        kind, encoded = _non_tensor_value(item)
                        structured.append(
                            {"value_type": kind, "value": encoded}
                        )
                value_type = (
                    "TENSOR_TUPLE"
                    if isinstance(value, tuple)
                    else "TENSOR_LIST"
                )
                encoded_value = structured
            else:
                value_type = "TENSOR_INPUT"
                encoded_value = tensor_indices[0]
        else:
            value_type, encoded_value = _non_tensor_value(value)
        rows.append(
            OperatorArgumentBinding(
                name=str(argument.name),
                schema_type=str(argument.type),
                source=source,
                tensor_input_indices=tuple(tensor_indices),
                value_type=value_type,
                value=encoded_value,
            )
        )
    return tuple(rows)


def classify_operator(overload: str, *, writes: bool) -> OperatorCategory:
    lowered = overload.lower()
    if "c10d" in lowered or any(
        token in lowered
        for token in ("all_reduce", "all_gather", "reduce_scatter", "broadcast")
    ):
        return OperatorCategory.COLLECTIVE
    if any(
        token in lowered
        for token in ("rand", "bernoulli", "dropout", "normal_", "uniform_")
    ):
        return OperatorCategory.RANDOM
    if writes:
        return OperatorCategory.MUTATION
    if any(
        token in lowered
        for token in (
            ".view",
            "reshape",
            "transpose",
            "permute",
            "as_strided",
            "slice",
            "select",
            "squeeze",
            "unsqueeze",
            "expand",
            "detach",
            "alias",
        )
    ):
        return OperatorCategory.VIEW_LAYOUT
    if any(
        token in lowered
        for token in ("copy", "clone", "_to_copy", "contiguous", "pin_memory")
    ):
        return OperatorCategory.MEMORY_COPY
    if any(
        token in lowered
        for token in (
            ".mm",
            ".bmm",
            "addmm",
            "matmul",
            "convolution",
            "linear",
        )
    ):
        return OperatorCategory.MATRIX
    if any(
        token in lowered
        for token in (
            "sum",
            "mean",
            "amax",
            "amin",
            "norm",
            "softmax",
            "logsumexp",
            "var",
        )
    ):
        return OperatorCategory.REDUCTION
    if any(
        token in lowered
        for token in (
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "silu",
            "gelu",
            "relu",
            "where",
            "masked_fill",
            "cos",
            "sin",
        )
    ):
        return OperatorCategory.ELEMENTWISE
    return OperatorCategory.OTHER


@dataclass(frozen=True)
class TensorMetadata:
    tensor_id: int
    storage_id: int | None
    source_ordinal: int | None
    shape: tuple[str, ...]
    stride: tuple[str, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    is_leaf: bool
    storage_offset: str | None


@dataclass(frozen=True)
class FullStepOperatorEvent:
    ordinal: int
    phase: StepPhase
    invocation_id: str
    overload: str
    dispatcher_schema: str
    category: OperatorCategory
    module_context: tuple[str, ...]
    autograd_node: str | None
    grad_enabled: bool
    input_tensors: tuple[TensorMetadata, ...]
    output_tensors: tuple[TensorMetadata, ...]
    argument_bindings: tuple[OperatorArgumentBinding, ...]
    schema_write_argument_indices: tuple[int, ...]
    mutated_tensor_input_indices: tuple[int, ...]
    forward_output_autograd_nodes: tuple[str | None, ...] = ()
    forward_output_autograd_sequence_nrs: tuple[int | None, ...] = ()
    forward_output_observation_statuses: tuple[str, ...] = ()
    forward_autograd_sequence_nr: int | None = None
    backward_autograd_sequence_nr: int | None = None
    sequence_binding_status: str = "NOT_CAPTURED"
    profiler_dispatch_ancestor_depth: int | None = None
    profiler_dispatch_boundary_name: str | None = None
    profiler_boundary_display_name_matches_overload: bool | None = None
    enclosing_dispatch_invocation_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["category"] = self.category.value
        return payload


@dataclass(frozen=True)
class FullStepOperatorTrace:
    events: tuple[FullStepOperatorEvent, ...]
    endpoint_tensors: Mapping[str, TensorMetadata]
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_count": len(self.events),
            "events": [event.as_dict() for event in self.events],
            "endpoint_tensors": {
                name: asdict(value)
                for name, value in sorted(self.endpoint_tensors.items())
            },
        }


class _ModuleContext:
    """Read-only Python context for forward ops; hooks execute no tensor ops."""

    def __init__(self, module: Any | None) -> None:
        self.module = module
        self.stack: list[str] = []
        self.handles: list[Any] = []
        self.invocations: Counter[str] = Counter()

    def install(self) -> None:
        if self.module is None:
            return
        for name, child in self.module.named_modules():
            qualified = name or "<root>"

            def before(
                hooked_module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                module_name: str = qualified,
            ) -> None:
                del hooked_module, args, kwargs
                invocation = self.invocations[module_name]
                self.invocations[module_name] += 1
                self.stack.append(f"{module_name}#{invocation}")

            def after(
                hooked_module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                module_name: str = qualified,
            ) -> None:
                del hooked_module, args, kwargs, output
                if not self.stack:
                    raise RuntimeError(
                        f"module context underflow at {module_name}"
                    )
                observed = self.stack.pop().rsplit("#", 1)[0]
                if observed != module_name:
                    raise RuntimeError(
                        f"module context mismatch: {observed} != {module_name}"
                    )

            self.handles.append(
                child.register_forward_pre_hook(before, with_kwargs=True)
            )
            self.handles.append(
                child.register_forward_hook(
                    after,
                    with_kwargs=True,
                    always_call=True,
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _MetadataRecorder(TorchDispatchMode):
    def __init__(
        self,
        module_context: _ModuleContext,
        *,
        emit_profiler_markers: bool = False,
        retain_forward_outputs_for_origin_binding: bool = False,
    ) -> None:
        super().__init__()
        self.phase = StepPhase.FORWARD
        self.module_context = module_context
        self.emit_profiler_markers = emit_profiler_markers
        self.retain_forward_outputs_for_origin_binding = (
            retain_forward_outputs_for_origin_binding
        )
        self.events: list[FullStepOperatorEvent] = []
        self.producer_by_object: dict[int, int] = {}
        self.producer_by_storage: dict[tuple[str, int], int] = {}
        self.tensor_ids: dict[int, int] = {}
        self.storage_ids: dict[tuple[str, int], int] = {}
        # These references are deliberately transient.  Autograd attaches
        # grad_fn metadata outside __torch_dispatch__, so the exact output
        # objects must be inspected after the forward closure returns.  They
        # are released before backward and never serialized.
        self._forward_output_tensors: dict[
            int,
            tuple[
                torch.Tensor | weakref.ReferenceType[torch.Tensor], ...
            ],
        ] = {}

    def _metadata(
        self,
        value: torch.Tensor,
        *,
        source_override: int | None = None,
    ) -> TensorMetadata:
        object_key = id(value)
        if object_key not in self.tensor_ids:
            self.tensor_ids[object_key] = len(self.tensor_ids)
        storage_key = _storage_key(value)
        storage_id = None
        if storage_key is not None:
            if storage_key not in self.storage_ids:
                self.storage_ids[storage_key] = len(self.storage_ids)
            storage_id = self.storage_ids[storage_key]
        source = source_override
        if source is None:
            source = self.producer_by_object.get(object_key)
        if source is None and storage_key is not None:
            source = self.producer_by_storage.get(storage_key)
        try:
            storage_offset = str(value.storage_offset())
        except RuntimeError:
            storage_offset = None
        return TensorMetadata(
            tensor_id=self.tensor_ids[object_key],
            storage_id=storage_id,
            source_ordinal=source,
            shape=_shape(value),
            stride=_stride(value),
            dtype=str(value.dtype),
            device=str(value.device),
            layout=str(value.layout),
            requires_grad=bool(value.requires_grad),
            is_leaf=bool(value.is_leaf),
            storage_offset=storage_offset,
        )

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        call_kwargs = kwargs or {}
        input_tensors = _tensor_leaves((args, call_kwargs))
        before_versions = tuple(_version(value) for value in input_tensors)
        input_metadata = tuple(self._metadata(value) for value in input_tensors)
        argument_bindings = _argument_bindings(
            func,
            args,
            call_kwargs,
            input_tensors,
        )
        schema_writes = _schema_writes(func)
        ordinal = len(self.events)
        if self.emit_profiler_markers:
            with torch.autograd.profiler.record_function(
                _dispatch_profiler_marker(ordinal)
            ):
                output = func(*args, **call_kwargs)
        else:
            output = func(*args, **call_kwargs)
        after_versions = tuple(_version(value) for value in input_tensors)
        mutated = tuple(
            index
            for index, (before, after) in enumerate(
                zip(before_versions, after_versions, strict=True)
            )
            if before is not None and after is not None and before != after
        )
        output_tensors = _tensor_leaves(output)
        output_metadata = tuple(
            self._metadata(value, source_override=ordinal)
            for value in output_tensors
        )
        overload = str(func)
        autograd_node, backward_sequence_nr = _current_autograd_node_info()
        event = FullStepOperatorEvent(
            ordinal=ordinal,
            phase=self.phase,
            invocation_id=f"{self.phase.value.lower()}:{ordinal}",
            overload=overload,
            dispatcher_schema=str(getattr(func, "_schema", func)),
            category=classify_operator(
                overload,
                writes=bool(schema_writes or mutated),
            ),
            module_context=tuple(self.module_context.stack),
            autograd_node=autograd_node,
            grad_enabled=bool(torch.is_grad_enabled()),
            input_tensors=input_metadata,
            output_tensors=output_metadata,
            argument_bindings=argument_bindings,
            schema_write_argument_indices=schema_writes,
            mutated_tensor_input_indices=mutated,
            backward_autograd_sequence_nr=(
                backward_sequence_nr
                if self.phase is StepPhase.BACKWARD
                else None
            ),
        )
        self.events.append(event)
        if self.phase is StepPhase.FORWARD:
            self._forward_output_tensors[ordinal] = (
                tuple(output_tensors)
                if self.retain_forward_outputs_for_origin_binding
                else tuple(weakref.ref(value) for value in output_tensors)
            )
        for value in output_tensors:
            self.producer_by_object[id(value)] = ordinal
            storage_key = _storage_key(value)
            if storage_key is not None:
                self.producer_by_storage[storage_key] = ordinal
        for index in mutated:
            value = input_tensors[index]
            self.producer_by_object[id(value)] = ordinal
            storage_key = _storage_key(value)
            if storage_key is not None:
                self.producer_by_storage[storage_key] = ordinal
        return output

    def bind_forward_output_autograd_origins(self) -> None:
        """Read the grad_fn attached to each exact forward output object.

        A profiler range's ``sequence_nr`` is a counter value: operations
        that create no autograd node can carry the same positive value as a
        later operation that does.  It is therefore audit metadata, not
        sufficient forward-origin evidence.  The output object's grad_fn is
        the direct entry point into the graph and supplies the actual node
        identity and its stashed sequence number.
        """

        enriched = []
        # Some tensor wrappers consult ATen while exposing autograd metadata.
        # Disable dispatch modes so this audit cannot add observer-generated
        # invocations to the execution-derived denominator.
        with _disable_current_modes():
            for event in tuple(self.events):
                if event.phase is not StepPhase.FORWARD:
                    enriched.append(event)
                    continue
                outputs = self._forward_output_tensors.get(event.ordinal)
                if outputs is None:
                    raise RuntimeError(
                        "missing transient forward outputs for "
                        f"{event.invocation_id}"
                    )
                nodes: list[str | None] = []
                sequence_nrs: list[int | None] = []
                observation_statuses: list[str] = []
                for output_reference in outputs:
                    output = (
                        output_reference
                        if isinstance(output_reference, torch.Tensor)
                        else output_reference()
                    )
                    if output is None:
                        nodes.append(None)
                        sequence_nrs.append(None)
                        observation_statuses.append(
                            "COLLECTED_BEFORE_FORWARD_ORIGIN_BINDING"
                        )
                        continue
                    node = output.grad_fn
                    if node is None:
                        nodes.append(None)
                        sequence_nrs.append(None)
                        observation_statuses.append(
                            "LIVE_OUTPUT_WITHOUT_GRAD_FN"
                        )
                        continue
                    try:
                        name = str(node.name())
                    except (AttributeError, RuntimeError):
                        name = type(node).__name__
                    try:
                        sequence_nr = int(node._sequence_nr())
                    except (AttributeError, RuntimeError, TypeError):
                        sequence_nr = None
                    if sequence_nr == _ACCUMULATE_GRAD_SEQUENCE_NR:
                        sequence_nr = None
                    nodes.append(name)
                    sequence_nrs.append(sequence_nr)
                    observation_statuses.append(
                        "LIVE_OUTPUT_WITH_GRAD_FN"
                    )
                enriched.append(
                    replace(
                        event,
                        forward_output_autograd_nodes=tuple(nodes),
                        forward_output_autograd_sequence_nrs=tuple(
                            sequence_nrs
                        ),
                        forward_output_observation_statuses=tuple(
                            observation_statuses
                        ),
                    )
                )
        self.events = enriched
        self._forward_output_tensors.clear()

    def endpoint_metadata(
        self,
        endpoints: Mapping[str, torch.Tensor],
    ) -> dict[str, TensorMetadata]:
        return {
            name: self._metadata(value)
            for name, value in sorted(endpoints.items())
        }


def observe_full_forward_backward_step(
    *,
    loss_closure: Callable[[], torch.Tensor],
    endpoint_closure: Callable[[], Mapping[str, torch.Tensor]],
    model: Any | None = None,
    capture_autograd_sequence_numbers: bool = False,
    retain_forward_outputs_for_origin_binding: bool = False,
) -> FullStepOperatorTrace:
    """Observe every ATen dispatch in one loss forward and backward."""

    context = _ModuleContext(model)
    recorder = _MetadataRecorder(
        context,
        emit_profiler_markers=capture_autograd_sequence_numbers,
        retain_forward_outputs_for_origin_binding=(
            retain_forward_outputs_for_origin_binding
        ),
    )
    context.install()
    try:
        profiler = (
            torch.autograd.profiler.profile(
                use_cuda=False,
                record_shapes=False,
                profile_memory=False,
            )
            if capture_autograd_sequence_numbers
            else None
        )
        if profiler is None:
            with recorder:
                recorder.phase = StepPhase.FORWARD
                loss = loss_closure()
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError(
                        "loss_closure must return one scalar tensor"
                    )
                if capture_autograd_sequence_numbers:
                    recorder.bind_forward_output_autograd_origins()
                recorder.phase = StepPhase.BACKWARD
                loss.backward()
        else:
            with profiler, recorder:
                recorder.phase = StepPhase.FORWARD
                loss = loss_closure()
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError(
                        "loss_closure must return one scalar tensor"
                    )
                recorder.bind_forward_output_autograd_origins()
                recorder.phase = StepPhase.BACKWARD
                loss.backward()
            recorder.events = list(
                _bind_forward_autograd_sequence_numbers(
                    recorder.events,
                    profiler.function_events,
                )
            )
        endpoints = endpoint_closure()
        if not endpoints:
            raise ValueError("endpoint_closure must return named tensors")
        if any(not isinstance(value, torch.Tensor) for value in endpoints.values()):
            raise TypeError("all full-step endpoints must be tensors")
        endpoint_metadata = recorder.endpoint_metadata(endpoints)
    finally:
        context.remove()
    return FullStepOperatorTrace(
        events=tuple(recorder.events),
        endpoint_tensors=endpoint_metadata,
    )


def build_full_step_coverage_certificate(
    trace: FullStepOperatorTrace,
    *,
    observation_stable: bool,
    subject: str,
    implementation_id: str,
) -> dict[str, Any]:
    """Build a fail-closed census certificate, not a correctness certificate."""

    events = trace.events
    phase_counts = Counter(event.phase.value for event in events)
    category_counts = Counter(event.category.value for event in events)
    overload_counts = Counter(event.overload for event in events)
    gates = {
        "nonempty_execution": bool(events),
        "forward_observed": phase_counts[StepPhase.FORWARD.value] > 0,
        "backward_observed": phase_counts[StepPhase.BACKWARD.value] > 0,
        "ordinals_contiguous": all(
            event.ordinal == index for index, event in enumerate(events)
        ),
        "all_invocations_classified": all(
            isinstance(event.category, OperatorCategory) for event in events
        ),
        "all_dispatcher_schemas_present": all(
            bool(event.dispatcher_schema) for event in events
        ),
        "all_arguments_bound": all(
            event.argument_bindings
            and all(
                binding.source != "MISSING_REQUIRED_ARGUMENT"
                and binding.value_type != "UNSUPPORTED"
                for binding in event.argument_bindings
            )
            for event in events
        ),
        "endpoints_present": bool(trace.endpoint_tensors),
        "observation_stable": bool(observation_stable),
    }
    status = (
        "COMPLETE_ALL_OP_CENSUS"
        if all(gates.values())
        else "UNRESOLVED"
    )
    return {
        "schema_version": "forkcert.full-step-op-coverage-certificate.v1",
        "status": status,
        "subject": subject,
        "implementation_id": implementation_id,
        "atomic_observation": "ONE_COMPLETE_FORWARD_BACKWARD_STEP",
        "denominator": {
            "dispatch_invocations": len(events),
            "phase_counts": dict(sorted(phase_counts.items())),
            "unique_overloads": len(overload_counts),
            "category_counts": dict(sorted(category_counts.items())),
            "module_context_forward_invocations": sum(
                bool(event.module_context)
                for event in events
                if event.phase is StepPhase.FORWARD
            ),
            "autograd_context_backward_invocations": sum(
                event.autograd_node is not None
                for event in events
                if event.phase is StepPhase.BACKWARD
            ),
        },
        "gates": gates,
        "endpoint_names": sorted(trace.endpoint_tensors),
        "claim_boundary": {
            "supported": (
                "complete metadata census of actual ATen dispatcher "
                "invocations in one observed forward/backward step"
            ),
            "not_yet_supported": [
                "eager-to-compiled/fused region alignment",
                "per-op numerical correctness",
                "per-op mathematical derivation",
                "long-training stability",
                "source or kernel bug localization",
            ],
        },
    }
