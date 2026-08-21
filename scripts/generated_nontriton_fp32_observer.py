"""FP32-storage counterfactuals for generated non-Triton compute calls."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import linecache
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from scripts.generated_fp32_observer import (
    nonfinite_aware_metrics,
    promoted_pointer_arguments,
    tensor_runtime_contract,
)


def _source_identity() -> tuple[str, int, str]:
    caller = sys._getframe(2)
    line = int(caller.f_lineno)
    source = linecache.getline(caller.f_code.co_filename, line).strip()
    return caller.f_code.co_filename, line, hashlib.sha256(source.encode()).hexdigest()


def _promote_args(args: Sequence[Any]) -> list[Any]:
    promoted, _ = promoted_pointer_arguments(args)
    return promoted


def _promote_call(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    names = list(kwargs)
    combined = [*args, *(kwargs[name] for name in names)]
    promoted, _ = promoted_pointer_arguments(combined)
    split = len(args)
    return promoted[:split], dict(zip(names, promoted[split:]))


def fp32_external_reference(
    symbol: str, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> torch.Tensor:
    """Execute the declared external operation with floating storages promoted."""

    values, promoted_kwargs = _promote_call(args, kwargs)
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    with torch.no_grad():
        if symbol == "mm":
            return torch.mm(tensors[0], tensors[1])
        if symbol == "bmm":
            return torch.bmm(tensors[0], tensors[1])
        if symbol == "addmm":
            return torch.addmm(
                tensors[0], tensors[1], tensors[2],
                beta=promoted_kwargs.get("beta", 1), alpha=promoted_kwargs.get("alpha", 1),
            )
        if symbol == "convolution":
            bias = tensors[2] if len(tensors) == 3 else promoted_kwargs.get("bias")
            return torch.ops.aten.convolution.default(
                tensors[0], tensors[1], bias,
                list(promoted_kwargs["stride"]), list(promoted_kwargs["padding"]),
                list(promoted_kwargs["dilation"]), bool(promoted_kwargs["transposed"]),
                list(promoted_kwargs["output_padding"]), int(promoted_kwargs["groups"]),
            )
    raise ValueError(f"unsupported external symbol: {symbol}")


class _AttributeProxy:
    def __init__(self, original: Any, replacements: Mapping[str, Any]) -> None:
        self._original = original
        self._replacements = dict(replacements)

    def __getattr__(self, name: str) -> Any:
        if name in self._replacements:
            return self._replacements[name]
        return getattr(self._original, name)


class GeneratedNonTritonFP32Observer:
    """Bind and replay every generated external/direct compute invocation."""

    def __init__(
        self,
        *,
        modules: Iterable[Any],
        inventory_rows: Sequence[Mapping[str, Any]],
        sample_size: int = 64,
        metric_chunk_elements: int = 1_048_576,
    ) -> None:
        self.modules = list(modules)
        self.sample_size = sample_size
        self.metric_chunk_elements = metric_chunk_elements
        supported = {"EXTERN", "DIRECT_ATEN", "DIRECT_TORCH_OP", "DIRECT_TENSOR_METHOD"}
        rows = [
            dict(row) for row in inventory_rows
            if row.get("category") == "COMPUTE"
            and row.get("implementation_kind_or_helper_role") in supported
        ]
        allowed_functions = {
            "EXTERN": {
                "extern_kernels.mm", "extern_kernels.bmm",
                "extern_kernels.addmm", "extern_kernels.convolution",
            },
            "DIRECT_ATEN": {"aten.index_put_"},
            "DIRECT_TORCH_OP": {
                "torch.ops.aten.convolution_backward.default",
                "torch.ops.aten.masked_scatter_backward.default",
            },
            "DIRECT_TENSOR_METHOD": set(),
        }
        unsupported = [
            (row["implementation_kind_or_helper_role"], row["function"])
            for row in rows
            if not (
                row["implementation_kind_or_helper_role"] == "DIRECT_TENSOR_METHOD"
                and str(row["function"]).endswith(".copy_")
            )
            and row["function"] not in allowed_functions[row["implementation_kind_or_helper_role"]]
        ]
        if unsupported:
            raise RuntimeError(f"non-Triton FP32 observer lacks exact dispatch: {unsupported}")
        self.expected = len(rows)
        self.expected_kinds = {
            str(row["implementation_kind_or_helper_role"]) for row in rows
        }
        self.rows: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["implementation_kind_or_helper_role"]),
                str(row["source_path"]),
                int(row["source_line"]),
                str(row["source_line_sha256"]),
            )
            self.rows[key].append(row)
        duplicate_callsites = {key: value for key, value in self.rows.items() if len(value) != 1}
        if duplicate_callsites:
            raise RuntimeError(
                "static non-Triton inventory has ambiguous exact callsites: "
                f"{sorted((key, len(value)) for key, value in duplicate_callsites.items())}"
            )
        # The capture script numbers copied wrappers in PyCodeCache insertion order.
        # Reconstruct that bijection for the independently compiled replay modules.
        captured_by_ordinal: dict[int, str] = {}
        for row in rows:
            match = re.search(r"(?:^|/)model__(\d+)_", str(row["source_path"]))
            if match is not None:
                ordinal = int(match.group(1))
                previous = captured_by_ordinal.setdefault(ordinal, str(row["source_path"]))
                if previous != str(row["source_path"]):
                    raise RuntimeError(f"ambiguous captured wrapper ordinal: {ordinal}")
        wrapper_modules = []
        for module in self.modules:
            source = Path(module.__file__).resolve()
            header = source.read_text(errors="ignore")[:512]
            if re.search(r"# AOT ID: \['\d+_(?:forward|backward|inference)'\]", header):
                wrapper_modules.append(module)
        if captured_by_ordinal and max(captured_by_ordinal) >= len(wrapper_modules):
            raise RuntimeError("replay compiled fewer modules than the frozen callsite inventory")
        self.runtime_to_captured_path = {
            str(Path(module.__file__).resolve()): captured_by_ordinal[ordinal]
            for ordinal, module in enumerate(wrapper_modules)
            if ordinal in captured_by_ordinal
        }
        self.counts: dict[tuple[str, str, int, str], int] = defaultdict(int)
        self.records: list[dict[str, Any]] = []
        self.unmatched_generated_copy_calls: list[dict[str, Any]] = []
        self.restores: list[Callable[[], None]] = []
        self.installed_kinds: set[str] = set()

    def _take(self, kind: str, filename: str, line: int, digest: str) -> dict[str, Any]:
        runtime_path = str(Path(filename).resolve())
        captured_path = self.runtime_to_captured_path.get(runtime_path)
        if captured_path is None:
            raise RuntimeError(f"runtime module absent from frozen inventory: {runtime_path}")
        key = (kind, captured_path, int(line), digest)
        choices = self.rows.get(key, [])
        if len(choices) != 1:
            raise RuntimeError(f"runtime non-Triton call absent or ambiguous: {key}")
        self.counts[key] += 1
        return choices[0]

    def _metric(self, candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
        return nonfinite_aware_metrics(
            candidate, reference, sample_size=self.sample_size,
            metric_chunk_elements=self.metric_chunk_elements,
        )

    def _record(
        self, row: Mapping[str, Any], filename: str, line: int,
        metrics: Mapping[str, Any],
        runtime_operands: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.records.append({
            "region_id": str(row["compute_region_id"]),
            "phase": str(row["phase"]),
            "implementation_kind": str(row["implementation_kind_or_helper_role"]),
            "function": str(row["function"]),
            "source_path": str(row["source_path"]),
            "runtime_filename": filename,
            "runtime_line": line,
            "runtime_invocation_ordinal": len(self.records),
            "callsite_execution_ordinal": self.counts[(
                str(row["implementation_kind_or_helper_role"]),
                str(row["source_path"]), int(row["source_line"]),
                str(row["source_line_sha256"]),
            )] - 1,
            "source_line_sha256": str(row["source_line_sha256"]),
            "reference_role": "PRECISION_ONLY_SAME_DECLARED_OP_FP32_STORAGE_COUNTERFACTUAL",
            "runtime_operand_contracts": {
                name: tensor_runtime_contract(value)
                for name, value in sorted((runtime_operands or {}).items())
            },
            "endpoint_metrics": dict(metrics),
        })

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
                self.installed_kinds.add("EXTERN")

                def wrapped(
                    *args: Any, _symbol: str = symbol, _original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    filename, line, digest = _source_identity()
                    row = self._take("EXTERN", filename, line, digest)
                    reference = fp32_external_reference(_symbol, args, kwargs)
                    result = _original(*args, **kwargs)
                    candidate = kwargs.get("out", result)
                    if not isinstance(candidate, torch.Tensor):
                        raise TypeError(f"external {_symbol} produced no tensor")
                    operands = {
                        **{
                            f"arg{position}": value
                            for position, value in enumerate(args)
                            if isinstance(value, torch.Tensor)
                        },
                        **{
                            f"kw:{name}": value
                            for name, value in kwargs.items()
                            if isinstance(value, torch.Tensor)
                        },
                    }
                    self._record(
                        row, filename, line,
                        {"output": self._metric(candidate, reference)},
                        runtime_operands=operands,
                    )
                    return result

                setattr(namespace, symbol, wrapped)
                self.restores.append(
                    lambda ns=namespace, name=symbol, value=original: setattr(ns, name, value)
                )

    def _install_direct_aten(self) -> None:
        for module in self.modules:
            namespace = getattr(module, "aten", None)
            original = getattr(namespace, "index_put_", None) if namespace is not None else None
            if not callable(original):
                continue
            self.installed_kinds.add("DIRECT_ATEN")

            def wrapped(
                buffer: torch.Tensor, indices: Sequence[torch.Tensor | None],
                values: torch.Tensor, accumulate: bool = False,
                _original: Any = original,
            ) -> Any:
                filename, line, digest = _source_identity()
                row = self._take("DIRECT_ATEN", filename, line, digest)
                promoted = _promote_args((buffer, *indices, values))
                reference_buffer = promoted[0]
                reference_indices = promoted[1:1 + len(indices)]
                reference_values = promoted[-1]
                result = _original(buffer, indices, values, accumulate)
                torch.ops.aten.index_put_.default(
                    reference_buffer, reference_indices, reference_values, accumulate,
                )
                self._record(
                    row, filename, line,
                    {"mutated_buffer": self._metric(buffer, reference_buffer)},
                )
                return result

            proxy = _AttributeProxy(namespace, {"index_put_": wrapped})
            module.aten = proxy
            self.restores.append(lambda target=module, value=namespace: setattr(target, "aten", value))

    def _install_direct_torch_ops(self) -> None:
        for module in self.modules:
            torch_namespace = getattr(module, "torch", None)
            if torch_namespace is None:
                continue
            self.installed_kinds.add("DIRECT_TORCH_OP")
            replacements = {}
            for symbol in ("convolution_backward", "masked_scatter_backward"):
                overload_namespace = getattr(torch_namespace.ops.aten, symbol)
                original = overload_namespace.default

                def wrapped(
                    *args: Any, _symbol: str = symbol, _original: Any = original,
                    **kwargs: Any,
                ) -> Any:
                    filename, line, digest = _source_identity()
                    runtime_path = str(Path(filename).resolve())
                    captured_path = self.runtime_to_captured_path.get(runtime_path)
                    key = ("DIRECT_TORCH_OP", captured_path, line, digest)
                    if key not in self.rows:
                        return _original(*args, **kwargs)
                    row = self._take("DIRECT_TORCH_OP", filename, line, digest)
                    promoted = _promote_args(args)
                    result = _original(*args, **kwargs)
                    reference_fn = getattr(torch.ops.aten, _symbol).default
                    reference = reference_fn(*promoted, **kwargs)
                    if isinstance(result, torch.Tensor):
                        metrics = {"output": self._metric(result, reference)}
                    else:
                        metrics = {
                            f"output_{index}": self._metric(candidate, expected)
                            for index, (candidate, expected) in enumerate(zip(result, reference))
                            if isinstance(candidate, torch.Tensor)
                        }
                    self._record(row, filename, line, metrics)
                    return result

                replacements[symbol] = _AttributeProxy(
                    overload_namespace, {"default": wrapped}
                )
            aten_proxy = _AttributeProxy(torch_namespace.ops.aten, replacements)
            ops_proxy = _AttributeProxy(torch_namespace.ops, {"aten": aten_proxy})
            module.torch = _AttributeProxy(torch_namespace, {"ops": ops_proxy})
            self.restores.append(
                lambda target=module, value=torch_namespace: setattr(target, "torch", value)
            )

    def _install_tensor_copy(self) -> None:
        original = torch.Tensor.copy_

        def wrapped(target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
            filename, line, digest = _source_identity()
            runtime_path = str(Path(filename).resolve())
            captured_path = self.runtime_to_captured_path.get(runtime_path)
            key = ("DIRECT_TENSOR_METHOD", captured_path, line, digest)
            if key not in self.rows:
                if "output_code.py" in filename:
                    self.unmatched_generated_copy_calls.append({
                        "filename": filename, "line": line,
                        "source_line_sha256": digest,
                        "source": linecache.getline(filename, line).strip(),
                    })
                return original(target, source, non_blocking)
            row = self._take("DIRECT_TENSOR_METHOD", filename, line, digest)
            promoted = _promote_args((target, source))
            result = original(target, source, non_blocking)
            original(promoted[0], promoted[1], non_blocking)
            self._record(row, filename, line, {"mutated_target": self._metric(target, promoted[0])})
            return result

        torch.Tensor.copy_ = wrapped
        self.installed_kinds.add("DIRECT_TENSOR_METHOD")
        self.restores.append(lambda value=original: setattr(torch.Tensor, "copy_", value))

    def __enter__(self) -> "GeneratedNonTritonFP32Observer":
        self._install_externals()
        self._install_direct_aten()
        self._install_direct_torch_ops()
        self._install_tensor_copy()
        missing_hooks = self.expected_kinds - self.installed_kinds
        if missing_hooks:
            self.__exit__(None, None, None)
            raise RuntimeError(f"required non-Triton hooks were not installed: {sorted(missing_hooks)}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()

    def summary(self) -> dict[str, Any]:
        observed = len(self.records)
        missing = []
        for key, rows in sorted(self.rows.items()):
            if self.counts.get(key, 0) == 0:
                row = rows[0]
                missing.append({
                    "kind": key[0], "source_line_sha256": key[3],
                    "function": row["function"], "source_path": row["source_path"],
                    "source_line": row["source_line"],
                })
        executed_static_calls = self.expected - len(missing)
        accounted = executed_static_calls + len(missing) == self.expected
        identities = [
            (
                row["implementation_kind"], row["source_line_sha256"],
                row["region_id"], row["runtime_invocation_ordinal"],
                row["callsite_execution_ordinal"], tuple(sorted(row["endpoint_metrics"])),
            )
            for row in self.records
        ]
        return {
            "schema": "kernel-analyzer-generated-nontriton-fp32-observer-v2",
            "status": (
                "COMPLETE_RUNTIME_NONTRITON_FP32_REPLAY_WITH_STATIC_DISPOSITION"
                if accounted and not self.unmatched_generated_copy_calls
                else "UNRESOLVED_NONTRITON_FP32_REPLAY"
            ),
            "denominator": {
                "static_generated_compute_calls": self.expected,
                "static_calls_executed_in_measured_step": executed_static_calls,
                "actual_invocations_in_measured_step": observed,
                "static_calls_not_executed_in_measured_step": len(missing),
            },
            "expected_kinds": sorted(self.expected_kinds),
            "installed_kinds": sorted(self.installed_kinds),
            "missing_rows": missing,
            "missing_rows_disposition": "NOT_EXECUTED_IN_THIS_MEASURED_STEP",
            "runtime_identity": identities,
            "unmatched_generated_copy_calls": list(self.unmatched_generated_copy_calls),
            "records": list(self.records),
        }
