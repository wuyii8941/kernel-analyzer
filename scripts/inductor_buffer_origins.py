"""Capture Inductor's exact IR-buffer to post-grad FX origin relation.

The generated wrapper comments identify every FX node that contributed to a
kernel, but they do not say which materialized buffer represents which node.
Inductor retains that stronger relation on each IR node as ``origin_node``.
This module records it while code generation is live, before the information
is discarded with the scheduler.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
import hashlib
import json
from typing import Any

import torch._inductor.debug as inductor_debug


SCHEMA_VERSION = "kernel-analyzer-inductor-buffer-origins-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _name(value: Any) -> str | None:
    getter = getattr(value, "get_name", None)
    if not callable(getter):
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _buffer_metadata(value: Any) -> dict[str, Any] | None:
    get_layout = getattr(value, "get_layout", None)
    if not callable(get_layout):
        return None
    try:
        layout = get_layout()
        return {
            "device": str(layout.device),
            "dtype": str(layout.dtype),
            "shape": [str(item) for item in layout.size],
            "stride": [str(item) for item in layout.stride],
        }
    except Exception:
        return None
def _node_record(kernel_name: str, scheduler_node: Any) -> dict[str, Any] | None:
    ir_node = getattr(scheduler_node, "node", None)
    if ir_node is None:
        ir_node = scheduler_node
    outputs = []
    get_outputs = getattr(scheduler_node, "get_outputs", None)
    if callable(get_outputs):
        try:
            outputs = [
                (name, _buffer_metadata(item))
                for item in get_outputs() if (name := _name(item))
            ]
        except Exception:
            outputs = []
    if not outputs and (name := _name(ir_node)):
        outputs = [(name, _buffer_metadata(ir_node))]
    if not outputs:
        return None

    origin = None
    get_origin = getattr(ir_node, "get_origin_node", None)
    if callable(get_origin):
        try:
            origin = get_origin()
        except Exception:
            origin = None
    origins = []
    get_origins = getattr(ir_node, "get_origins", None)
    if callable(get_origins):
        try:
            origins = sorted({str(item.name) for item in get_origins()})
        except Exception:
            origins = []
    origin_names = ([str(origin.name)] if origin is not None else []) + origins
    phase_markers = {
        "FORWARD" if name.startswith("ka_f_") else "BACKWARD"
        for name in origin_names
        if name.startswith(("ka_f_", "ka_b_"))
    }
    return {
        "kernel_name": str(kernel_name),
        "scheduler_node": _name(scheduler_node),
        "buffer_names": sorted({name for name, _ in outputs}),
        "buffer_metadata": {
            name: metadata for name, metadata in outputs
        },
        "exact_origin_node": str(origin.name) if origin is not None else None,
        "contributing_origin_nodes": origins,
        "origin_node_exact": origin is not None,
        "phase": next(iter(phase_markers)) if len(phase_markers) == 1 else "UNRESOLVED",
    }


class InductorBufferOriginRecorder(AbstractContextManager):
    """Temporarily wrap Inductor's provenance hook and retain exact origins."""

    def __init__(self) -> None:
        self._original: Any = None
        self._patched_modules: list[tuple[Any, Any]] = []
        self._records: list[dict[str, Any]] = []

    def __enter__(self) -> "InductorBufferOriginRecorder":
        if self._original is not None:
            raise RuntimeError("Inductor origin recorder cannot be nested")
        self._original = inductor_debug.set_kernel_post_grad_provenance_tracing
        original = self._original

        def wrapped(
            node_schedule: Any,
            kernel_name: str,
            is_extern: bool = False,
        ) -> Any:
            result = original(node_schedule, kernel_name, is_extern)
            nodes = [node_schedule] if is_extern else list(node_schedule)
            for node in nodes:
                record = _node_record(kernel_name, node)
                if record is not None:
                    record["is_external"] = bool(is_extern)
                    self._records.append(record)
            return result

        # Triton/CPP codegen import the hook into their module namespace. Patch
        # those already-imported bindings as well as the defining module.
        modules = [inductor_debug]
        try:
            import torch._inductor.codegen.triton as triton_codegen
            modules.append(triton_codegen)
        except ImportError:
            pass
        try:
            import torch._inductor.codegen.cpp as cpp_codegen
            modules.append(cpp_codegen)
        except ImportError:
            pass
        for module in modules:
            prior = getattr(module, "set_kernel_post_grad_provenance_tracing")
            self._patched_modules.append((module, prior))
            setattr(module, "set_kernel_post_grad_provenance_tracing", wrapped)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._original is not None:
            for module, prior in reversed(self._patched_modules):
                setattr(module, "set_kernel_post_grad_provenance_tracing", prior)
            self._patched_modules.clear()
            self._original = None

    def certificate(self) -> dict[str, Any]:
        by_buffer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_kernel_buffer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._records:
            for name in row["buffer_names"]:
                detail = {
                    key: value for key, value in row.items()
                    if key not in {"buffer_names", "buffer_metadata"}
                } | {
                    "tensor_metadata": row["buffer_metadata"].get(name)
                }
                by_buffer[name].append(detail)
                by_kernel_buffer[
                    f"{row['phase']}\0{row['kernel_name']}\0{name}"
                ].append(detail)
        ambiguous = {
            name: rows for name, rows in by_kernel_buffer.items()
            if len({row["exact_origin_node"] for row in rows}) > 1
        }
        exact_buffers = sum(
            any(row["origin_node_exact"] for row in rows)
            for rows in by_kernel_buffer.values()
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "status": (
                "COMPLETE_EXACT_IR_BUFFER_ORIGIN_CAPTURE"
                if by_kernel_buffer and exact_buffers == len(by_kernel_buffer) and not ambiguous
                else "PARTIAL_FAIL_CLOSED"
            ),
            "denominator": {
                "scheduler_records": len(self._records),
                "materialized_buffers": len(by_kernel_buffer),
                "distinct_unqualified_buffer_names": len(by_buffer),
                "buffers_with_exact_origin_node": exact_buffers,
                "ambiguous_buffer_origins": len(ambiguous),
            },
            "kernel_buffer_origins": dict(sorted(by_kernel_buffer.items())),
            "buffer_origins": dict(sorted(by_buffer.items())),
            "ambiguous_buffer_origins": ambiguous,
            "claim_boundary": (
                "Compiler-live IR buffer provenance only. This proves buffer-to-post-grad-FX "
                "origin identity; it does not by itself prove numerical equivalence."
            ),
        }
        payload["result_sha256"] = _digest(payload)
        return payload
