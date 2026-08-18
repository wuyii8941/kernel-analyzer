"""Capture exact generated endpoints selected by compiler-carried AOT origins."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import linecache
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from scripts.generated_fp32_observer import runtime_signature
from scripts.generated_nontriton_fp32_observer import _AttributeProxy


TensorSink = Callable[[str, torch.Tensor, Mapping[str, Any]], None]


class DirectPrimitiveEndpointObserver:
    """Bind one primitive endpoint in the *current* generated execution.

    This is deliberately narrower than ``SameDtypeSemanticCandidateObserver``:
    it is used only by the SEUP replay when a frozen compiler release cannot be
    reused after recompilation.  Binding is still structural, never by tensor
    values: rsqrt uses its exact output shape and occurrence, while bmm uses
    the exact occurrence of the external bmm call.  The observer records the
    live generated symbol and all pointer operands so the resulting certificate
    remains auditable.
    """

    def __init__(
        self, *, modules: Iterable[Any], task: Mapping[str, Any], sink: TensorSink,
        operation: str, target_shape: Sequence[int] | None = None,
        target_occurrence: int = 0,
    ) -> None:
        self.modules = list(modules)
        self.task = dict(task)
        self.sink = sink
        self.operation = str(operation)
        self.target_shape = tuple(int(x) for x in target_shape) if target_shape is not None else None
        self.target_occurrence = int(target_occurrence)
        self.occurrences = 0
        self.delivered = 0
        self.restores: list[tuple[Any, str, Any, bool]] = []
        self.nontriton_restores: list[Callable[[], None]] = []

    @staticmethod
    def _pointers(kernel: Any, args: tuple[Any, ...]) -> dict[str, torch.Tensor]:
        names = [
            str(name) for name, annotation in runtime_signature(kernel)
            if str(annotation).startswith("*")
        ]
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if len(names) != len(tensors):
            raise RuntimeError(
                f"dynamic endpoint pointer ABI changed: expected {len(names)}, got {len(tensors)}"
            )
        return dict(zip(names, tensors))

    def _emit(self, value: torch.Tensor, metadata: dict[str, Any]) -> None:
        task_id = str(self.task["task_id"])
        self.sink(task_id, value, metadata)
        self.delivered += 1

    def _maybe_emit_triton(self, symbol: str, kernel: Any, args: tuple[Any, ...]) -> None:
        pointers = self._pointers(kernel, args)
        formal = str(self.task["formal_pointer"])
        if formal not in pointers:
            return
        value = pointers[formal]
        if self.operation != "rsqrt":
            return
        if self.target_shape is not None and tuple(value.shape) != self.target_shape:
            return
        occurrence = self.occurrences
        self.occurrences += 1
        if occurrence != self.target_occurrence:
            return
        runtime_pointers = {
            name: tensor.detach().clone() for name, tensor in pointers.items()
        }
        self._emit(value, {
            "candidate_region_id": self.task.get("candidate_region_id"),
            "symbol": symbol,
            "formal_pointer": formal,
            "exact_aot_endpoint_id": self.task.get("exact_aot_endpoint_id"),
            "binding_rule": "current_triton_symbol+output_shape+ordinal",
            "shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
            "runtime_pointers": runtime_pointers,
        })

    def _install_triton(self) -> None:
        seen: set[int] = set()
        for module in self.modules:
            for symbol, kernel in vars(module).items():
                if id(kernel) in seen or not callable(getattr(kernel, "run", None)):
                    continue
                seen.add(id(kernel))
                original = kernel.run

                def wrapped(*args: Any, _symbol: str = str(symbol), _kernel: Any = kernel,
                            _original: Any = original, **kwargs: Any) -> Any:
                    # Capture pre-call buffers before an in-place Triton store.
                    pointers = self._pointers(_kernel, args)
                    result = _original(*args, **kwargs)
                    if self.operation == "rsqrt":
                        # Reuse the pre-call pointers by temporarily feeding the
                        # same structural matcher; it clones only on a match.
                        formal = str(self.task["formal_pointer"])
                        value = pointers.get(formal)
                        if value is not None and (
                            self.target_shape is None or tuple(value.shape) == self.target_shape
                        ):
                            occurrence = self.occurrences
                            self.occurrences += 1
                            if occurrence == self.target_occurrence:
                                runtime_pointers = {
                                    name: tensor.detach().clone() for name, tensor in pointers.items()
                                }
                                self._emit(value, {
                                    "candidate_region_id": self.task.get("candidate_region_id"),
                                    "symbol": _symbol, "formal_pointer": formal,
                                    "exact_aot_endpoint_id": self.task.get("exact_aot_endpoint_id"),
                                    "binding_rule": "current_triton_symbol+output_shape+ordinal",
                                    "shape": list(value.shape), "stride": list(value.stride()),
                                    "dtype": str(value.dtype), "runtime_pointers": runtime_pointers,
                                })
                    return result

                kernel.run = wrapped
                self.restores.append((kernel, "run", original, True))

    def _install_bmm(self) -> None:
        if self.operation != "bmm":
            return
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = getattr(namespace, "bmm", None)
            if not callable(original):
                continue

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                result = _original(*args, **kwargs)
                output = kwargs.get("out", result)
                if not isinstance(output, torch.Tensor):
                    return result
                occurrence = self.occurrences
                self.occurrences += 1
                if occurrence == self.target_occurrence:
                    pointers: dict[str, torch.Tensor] = {}
                    tensor_args = [value for value in args if isinstance(value, torch.Tensor)]
                    if tensor_args:
                        pointers["input_0"] = tensor_args[0].detach().clone()
                    if len(tensor_args) > 1:
                        pointers["input_1"] = tensor_args[1].detach().clone()
                    pointers["output_0"] = output.detach().clone()
                    self._emit(output, {
                        "candidate_region_id": self.task.get("candidate_region_id"),
                        "symbol": "extern_kernels.bmm", "formal_pointer": "output_0",
                        "exact_aot_endpoint_id": self.task.get("exact_aot_endpoint_id"),
                        "binding_rule": "current_extern_bmm_ordinal",
                        "shape": list(output.shape), "stride": list(output.stride()),
                        "dtype": str(output.dtype), "runtime_pointers": pointers,
                    })
                return result

            setattr(namespace, "bmm", wrapped)
            self.nontriton_restores.append(
                lambda ns=namespace, value=original: setattr(ns, "bmm", value)
            )

    def __enter__(self) -> "DirectPrimitiveEndpointObserver":
        self._install_triton()
        self._install_bmm()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for kernel, name, original, _had_run in reversed(self.restores):
            setattr(kernel, name, original)
        self.restores.clear()
        for restore in reversed(self.nontriton_restores):
            restore()
        self.nontriton_restores.clear()

    def validate(self) -> None:
        if self.delivered != 1:
            raise RuntimeError(
                f"dynamic primitive endpoint census incomplete: delivered={self.delivered}, "
                f"operation={self.operation}, occurrences={self.occurrences}, "
                f"target_occurrence={self.target_occurrence}"
            )


class SameDtypeSemanticCandidateObserver:
    """Observe an unmodified candidate at exact AOT-semantic output ports."""

    def __init__(
        self, *, modules: Iterable[Any], campaign_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]], sink: TensorSink,
        inventory_rows: Sequence[Mapping[str, Any]] = (),
        include_unresolved_tasks: bool = False,
        allow_missing_symbols: bool = False,
    ) -> None:
        self.modules = list(modules)
        self.allow_missing_symbols = bool(allow_missing_symbols)
        by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in campaign_rows:
            by_symbol[str(row["symbol"])].append(row)
        self.rows_by_symbol = dict(by_symbol)
        self.tasks_by_region: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in task_rows:
            if include_unresolved_tasks or row.get(
                "exact_semantic_endpoint_id", row.get("exact_aot_endpoint_id")
            ) is not None:
                self.tasks_by_region[str(row["candidate_region_id"])].append(row)
        self.expected_task_ids = {
            str(row["task_id"])
            for rows in self.tasks_by_region.values() for row in rows
        }
        self.sink = sink
        self.symbol_counts: dict[str, int] = {}
        self.task_counts: dict[str, int] = defaultdict(int)
        self.restores: list[tuple[Any, bool, Any]] = []
        supported = {"EXTERN", "DIRECT_ATEN", "DIRECT_TORCH_OP", "DIRECT_TENSOR_METHOD"}
        self.nontriton_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in inventory_rows:
            kind = str(row.get("implementation_kind_or_helper_role"))
            region_id = str(row.get("compute_region_id"))
            if (
                row.get("category") == "COMPUTE"
                and kind in supported
                and region_id in self.tasks_by_region
            ):
                self.nontriton_rows[(kind, str(row["source_line_sha256"]))].append(row)
        self.nontriton_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.nontriton_restores: list[Callable[[], None]] = []

    @staticmethod
    def _source_identity() -> tuple[str, int, str]:
        caller = sys._getframe(2)
        line = int(caller.f_lineno)
        source = linecache.getline(caller.f_code.co_filename, line).strip()
        return caller.f_code.co_filename, line, hashlib.sha256(source.encode()).hexdigest()

    def _take_nontriton(self, kind: str, digest: str) -> Mapping[str, Any]:
        key = (kind, digest)
        index = self.nontriton_counts[key]
        choices = self.nontriton_rows.get(key, ())
        if index >= len(choices):
            raise RuntimeError(f"candidate non-Triton call outside frozen census: {key}:{index}")
        self.nontriton_counts[key] += 1
        return choices[index]

    def _emit_nontriton(
        self, row: Mapping[str, Any], value: torch.Tensor, endpoint: str,
    ) -> None:
        region_id = str(row["compute_region_id"])
        for task in self.tasks_by_region.get(region_id, ()):
            if str(task["formal_pointer"]) != endpoint:
                continue
            self.sink(str(task["task_id"]), value, {
                "candidate_region_id": region_id,
                "implementation_kind": row["implementation_kind_or_helper_role"],
                "function": row["function"],
                "endpoint": endpoint,
                "exact_aot_endpoint_id": task.get("exact_aot_endpoint_id"),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
            })
            self.task_counts[str(task["task_id"])] += 1

    def _install_externals(self) -> None:
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            for symbol in ("mm", "bmm", "addmm", "convolution"):
                original = getattr(namespace, symbol, None)
                if not callable(original):
                    continue

                def wrapped(
                    *args: Any, _symbol: str = symbol, _original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    _filename, _line, digest = self._source_identity()
                    key = ("EXTERN", digest)
                    if key not in self.nontriton_rows:
                        return _original(*args, **kwargs)
                    row = self._take_nontriton(*key)
                    result = _original(*args, **kwargs)
                    value = kwargs.get("out", result)
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError("external candidate endpoint is not a tensor")
                    self._emit_nontriton(row, value, "output_0")
                    return result

                setattr(namespace, symbol, wrapped)
                self.nontriton_restores.append(
                    lambda ns=namespace, name=symbol, value=original: setattr(ns, name, value)
                )

    def _install_direct_aten(self) -> None:
        for module in self.modules:
            namespace = getattr(module, "aten", None)
            original = getattr(namespace, "index_put_", None) if namespace is not None else None
            if not callable(original):
                continue

            def wrapped(
                buffer: torch.Tensor, indices: Sequence[torch.Tensor | None],
                values: torch.Tensor, accumulate: bool = False,
                _original: Any = original,
            ) -> Any:
                _filename, _line, digest = self._source_identity()
                key = ("DIRECT_ATEN", digest)
                if key not in self.nontriton_rows:
                    return _original(buffer, indices, values, accumulate)
                row = self._take_nontriton(*key)
                result = _original(buffer, indices, values, accumulate)
                self._emit_nontriton(row, buffer, "mutated_output_0")
                return result

            module.aten = _AttributeProxy(namespace, {"index_put_": wrapped})
            self.nontriton_restores.append(
                lambda target=module, value=namespace: setattr(target, "aten", value)
            )

    def _install_convolution_backward(self) -> None:
        if not any(kind == "DIRECT_TORCH_OP" for kind, _digest in self.nontriton_rows):
            return
        for module in self.modules:
            torch_namespace = getattr(module, "torch", None)
            if torch_namespace is None:
                continue
            original = torch_namespace.ops.aten.convolution_backward.default

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _filename, _line, digest = self._source_identity()
                key = ("DIRECT_TORCH_OP", digest)
                if key not in self.nontriton_rows:
                    return _original(*args, **kwargs)
                row = self._take_nontriton(*key)
                result = _original(*args, **kwargs)
                for index, value in enumerate(result):
                    if isinstance(value, torch.Tensor):
                        self._emit_nontriton(row, value, f"output_{index}")
                return result

            default_proxy = _AttributeProxy(
                torch_namespace.ops.aten.convolution_backward, {"default": wrapped}
            )
            convolution_proxy = _AttributeProxy(
                torch_namespace.ops.aten, {"convolution_backward": default_proxy}
            )
            ops_proxy = _AttributeProxy(torch_namespace.ops, {"aten": convolution_proxy})
            module.torch = _AttributeProxy(torch_namespace, {"ops": ops_proxy})
            self.nontriton_restores.append(
                lambda target=module, value=torch_namespace: setattr(target, "torch", value)
            )

    def _install_tensor_copy(self) -> None:
        if not any(kind == "DIRECT_TENSOR_METHOD" for kind, _digest in self.nontriton_rows):
            return
        original = torch.Tensor.copy_

        def wrapped(
            target: torch.Tensor, source: torch.Tensor,
            non_blocking: bool = False,
        ) -> torch.Tensor:
            _filename, _line, digest = self._source_identity()
            key = ("DIRECT_TENSOR_METHOD", digest)
            if key not in self.nontriton_rows:
                return original(target, source, non_blocking)
            row = self._take_nontriton(*key)
            result = original(target, source, non_blocking)
            self._emit_nontriton(row, target, "mutated_output_0")
            return result

        torch.Tensor.copy_ = wrapped
        self.nontriton_restores.append(
            lambda value=original: setattr(torch.Tensor, "copy_", value)
        )

    def __enter__(self) -> "SameDtypeSemanticCandidateObserver":
        if self.allow_missing_symbols:
            available = {
                str(symbol)
                for module in self.modules
                for symbol, kernel in vars(module).items()
                if callable(getattr(kernel, "run", None))
            }
            used: set[str] = set()
            remapped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for expected, rows in self.rows_by_symbol.items():
                actual = expected if expected in available else None
                if actual is None:
                    normal = re.sub(r"_\d+$", "", expected)
                    candidates = sorted(
                        value for value in available - used
                        if re.sub(r"_\d+$", "", value) == normal
                    )
                    if candidates:
                        actual = candidates[0]
                if actual is not None:
                    remapped[actual].extend(rows)
                    used.add(actual)
            self.rows_by_symbol = dict(remapped)
        self._install_externals()
        self._install_direct_aten()
        self._install_convolution_backward()
        self._install_tensor_copy()
        found_symbols = set()
        seen_kernels = set()
        for module in self.modules:
            for symbol, kernel in vars(module).items():
                if symbol not in self.rows_by_symbol or id(kernel) in seen_kernels:
                    continue
                if not callable(getattr(kernel, "run", None)):
                    continue
                seen_kernels.add(id(kernel))
                found_symbols.add(symbol)
                had_run = "run" in vars(kernel)
                previous = vars(kernel).get("run")
                original = kernel.run

                def wrapped(
                    *args: Any, _symbol: str = symbol, _kernel: Any = kernel,
                    _original: Any = original, **kwargs: Any,
                ) -> Any:
                    index = self.symbol_counts.get(_symbol, 0)
                    self.symbol_counts[_symbol] = index + 1
                    rows = self.rows_by_symbol[_symbol]
                    if index >= len(rows):
                        raise RuntimeError(f"candidate invocation outside frozen census: {_symbol}:{index}")
                    region = rows[index]
                    pointer_names = [
                        str(name) for name, annotation in runtime_signature(_kernel)
                        if str(annotation).startswith("*")
                    ]
                    tensors = [value for value in args if isinstance(value, torch.Tensor)]
                    if len(pointer_names) != len(tensors):
                        raise RuntimeError("candidate pointer ABI changed")
                    pointers = dict(zip(pointer_names, tensors))
                    # Keep pre-invocation operands available to a mainline
                    # semantic repair.  In-place ``in_out_ptr`` buffers are
                    # overwritten by the generated kernel, so observing the
                    # post-call tensor alone is not sufficient to reconstruct
                    # an exact primitive reference.
                    runtime_pointers = {
                        name: value.detach().clone() for name, value in pointers.items()
                    }
                    result = _original(*args, **kwargs)
                    for task in self.tasks_by_region.get(str(region["region_id"]), ()):
                        formal = str(task["formal_pointer"])
                        if formal not in pointers:
                            raise RuntimeError(f"semantic endpoint pointer is absent: {task['task_id']}")
                        value = pointers[formal]
                        self.sink(str(task["task_id"]), value, {
                            "candidate_region_id": region["region_id"],
                            "symbol": _symbol,
                            "formal_pointer": formal,
                            "exact_aot_endpoint_id": task["exact_aot_endpoint_id"],
                            "shape": list(value.shape),
                            "stride": list(value.stride()),
                            "dtype": str(value.dtype),
                            "runtime_pointers": runtime_pointers,
                        })
                        self.task_counts[str(task["task_id"])] += 1
                    return result

                self.restores.append((kernel, had_run, previous))
                kernel.run = wrapped
        missing_symbols = set(self.rows_by_symbol) - found_symbols
        if missing_symbols and not self.allow_missing_symbols:
            raise RuntimeError(f"frozen candidate symbols are absent: {sorted(missing_symbols)[:8]}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for kernel, had_run, previous in reversed(self.restores):
            if had_run:
                kernel.run = previous
            else:
                delattr(kernel, "run")
        self.restores.clear()
        for restore in reversed(self.nontriton_restores):
            restore()
        self.nontriton_restores.clear()

    def validate(self) -> None:
        missing = self.expected_task_ids - set(self.task_counts)
        repeated = {
            task_id: count for task_id, count in self.task_counts.items()
            if count != 1
        }
        incomplete_symbols = {
            symbol: {"expected": len(rows), "observed": self.symbol_counts.get(symbol, 0)}
            for symbol, rows in self.rows_by_symbol.items()
            if self.symbol_counts.get(symbol, 0) != len(rows)
        }
        incomplete_nontriton = {
            key: {"expected": len(rows), "observed": self.nontriton_counts.get(key, 0)}
            for key, rows in self.nontriton_rows.items()
            if self.nontriton_counts.get(key, 0) != len(rows)
        }
        if missing or repeated or incomplete_symbols or incomplete_nontriton:
            raise RuntimeError(
                "same-dtype candidate endpoint census incomplete: "
                f"missing={sorted(missing)[:8]} repeated={repeated} "
                f"symbols={dict(list(incomplete_symbols.items())[:8])} "
                f"nontriton={dict(list(incomplete_nontriton.items())[:8])}"
            )
