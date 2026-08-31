"""Observe or repair one exact generated external callsite."""

from __future__ import annotations

import hashlib
from functools import lru_cache
import linecache
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping

import torch

from forkcert.directional_error_sketch import fixed_flat_coordinate_indices
from kernel_analyzer.short_persistence import count_sketch_chunks
from scripts.generated_nontriton_fp32_observer import fp32_external_reference


def _source_identity() -> tuple[str, int, str]:
    caller = sys._getframe(2)
    line = int(caller.f_lineno)
    source = linecache.getline(caller.f_code.co_filename, line).strip()
    return caller.f_code.co_filename, line, hashlib.sha256(source.encode()).hexdigest()


COUNT_SKETCH_V2_SEED = 20260831


def _count_sketch(
    value: torch.Tensor,
    *,
    dimension: int = 4096,
    seed: int = COUNT_SKETCH_V2_SEED,
) -> torch.Tensor:
    """Return the v2 value-blind sketch without periodic coordinate aliasing."""

    flat = value.detach().float().reshape(-1)
    chunks = (
        flat[start:min(flat.numel(), start + 1_000_000)].cpu().numpy()
        for start in range(0, flat.numel(), 1_000_000)
    )
    result, coordinate_count = count_sketch_chunks(
        chunks, projection_dim=dimension, seed=seed,
    )
    if coordinate_count != flat.numel():
        raise RuntimeError("CountSketch did not consume the complete vector")
    # short_persistence normalizes its persistence sketch by sqrt(dimension).
    # Restore the standard CountSketch scale here because this path also stores
    # effect and repair norms. Ratios remain invariant either way.
    result *= dimension ** 0.5
    return torch.from_numpy(result).float()


class TargetedExternalIntervention:
    """Bind one call by source SHA and optionally replace its output with FP32."""

    MODES = {"OBSERVE", "SHAM", "REPAIR"}

    def __init__(
        self, *, modules: Iterable[Any], target: Mapping[str, Any], mode: str,
        capture_tensors: bool = True,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported intervention mode: {mode}")
        if target["implementation_kind"] != "EXTERN":
            raise ValueError("targeted intervention currently requires an EXTERN call")
        function = str(target["function"])
        if function not in {"extern_kernels.mm", "extern_kernels.bmm", "extern_kernels.addmm"}:
            raise ValueError(f"unsupported targeted function: {function}")
        self.modules = list(modules)
        self.target = dict(target)
        self.mode = mode
        self.capture_tensors = capture_tensors
        self.count = 0
        self.records: list[dict[str, Any]] = []
        self.restores: list[Callable[[], None]] = []

    def __enter__(self) -> "TargetedExternalIntervention":
        symbol = str(self.target["function"]).rsplit(".", 1)[-1]
        seen: set[int] = set()
        installed = 0
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = getattr(namespace, symbol, None)
            if not callable(original):
                continue
            installed += 1

            def wrapped(
                *args: Any, _original: Any = original, _symbol: str = symbol,
                **kwargs: Any,
            ) -> Any:
                filename, line, digest = _source_identity()
                if digest != self.target["source_line_sha256"]:
                    return _original(*args, **kwargs)
                self.count += 1
                if self.count != 1:
                    raise RuntimeError("target callsite executed more than once")
                reference = fp32_external_reference(_symbol, args, kwargs)
                result = _original(*args, **kwargs)
                candidate = kwargs.get("out", result)
                if not isinstance(candidate, torch.Tensor):
                    raise TypeError("target external call produced no tensor")
                reference_cast = reference.to(dtype=candidate.dtype)
                error = candidate.detach().float() - reference.detach().float()
                cast_delta = candidate.detach().float() - reference_cast.detach().float()
                compact_metrics = {
                    "full_coordinate_count": int(error.numel()),
                    "rms": float(torch.mean(error.double().square()).sqrt()),
                    "max_abs": float(error.abs().max()),
                    "reference_cast_changed_coordinates": int(torch.count_nonzero(cast_delta)),
                    "reference_cast_max_abs_change": float(cast_delta.abs().max()),
                }
                # Keep a value-blind, fixed-coordinate view of large GEMM/BMM
                # outputs.  This is small enough for 32-state formation tests
                # and, unlike RMS alone, preserves signed direction.
                positions = fixed_flat_coordinate_indices(
                    int(candidate.numel()), sample_size=256,
                ).to(candidate.device)
                candidate_sample = candidate.detach().reshape(-1)[positions].float().cpu()
                reference_cast_sample = reference_cast.detach().reshape(-1)[positions].float().cpu()
                compact_metrics["same_dtype_directional_sketch"] = {
                    "selection_rule": "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES",
                    "tensor_numel": int(candidate.numel()),
                    "flat_coordinate_indices": positions.cpu().tolist(),
                    "candidate_values": candidate_sample.tolist(),
                    "reference_values": reference_cast_sample.tolist(),
                    "signed_delta_values": (candidate_sample - reference_cast_sample).tolist(),
                }
                compact_metrics["same_dtype_count_sketch"] = {
                    "schema": "SPLITMIX64_COUNT_SKETCH_V2",
                    "selection_rule": "FIXED_VALUE_BLIND_HASH_BEFORE_READING_VALUES",
                    "projection_dimension": 4096,
                    "projection_seed": COUNT_SKETCH_V2_SEED,
                    "tensor_numel": int(candidate.numel()),
                    "effect": _count_sketch(candidate.detach().float() - reference_cast.detach().float()).tolist(),
                    "repair": _count_sketch(reference_cast).tolist(),
                }
                candidate_before = candidate.detach().float().cpu().clone() if self.capture_tensors else None
                reference_cpu = reference.detach().float().cpu().clone() if self.capture_tensors else None
                reference_cast_cpu = reference_cast.detach().float().cpu().clone() if self.capture_tensors else None
                if self.mode == "REPAIR":
                    candidate.copy_(reference_cast)
                elif self.mode == "SHAM":
                    candidate.copy_(candidate.clone())
                self.records.append({
                    "runtime_filename": filename,
                    "runtime_line": line,
                    "source_line_sha256": digest,
                    "candidate_dtype": str(candidate.dtype),
                    "candidate_shape": list(candidate.shape),
                    "reference_dtype": str(reference.dtype),
                    "candidate_before": candidate_before,
                    "reference": reference_cpu,
                    "reference_cast": reference_cast_cpu,
                    "delivered": candidate.detach().float().cpu().clone() if self.capture_tensors else None,
                    "compact_metrics": compact_metrics,
                    "delivered_matches_reference_cast": bool(torch.equal(candidate, reference_cast)),
                })
                return result

            setattr(namespace, symbol, wrapped)
            self.restores.append(
                lambda ns=namespace, name=symbol, value=original: setattr(ns, name, value)
            )
        if installed == 0:
            raise RuntimeError(f"no generated namespace exposes extern_kernels.{symbol}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()

    def summary(self) -> dict[str, Any]:
        if self.count != 1 or len(self.records) != 1:
            raise RuntimeError(f"target callsite execution count is {self.count}, expected 1")
        row = self.records[0]
        if not self.capture_tensors:
            return {
                "mode": self.mode,
                "source_line_sha256": row["source_line_sha256"],
                "runtime_filename": row["runtime_filename"],
                "runtime_line": row["runtime_line"],
                "candidate_dtype": row["candidate_dtype"],
                "candidate_shape": row["candidate_shape"],
                "reference_dtype": row["reference_dtype"],
                **row["compact_metrics"],
                "signed_error": None,
                "delivered_matches_reference_cast": row["delivered_matches_reference_cast"],
            }
        error = row["candidate_before"] - row["reference"]
        cast_delta = row["candidate_before"] - row["reference_cast"]
        delivered = row["delivered"]
        return {
            "mode": self.mode,
            "source_line_sha256": row["source_line_sha256"],
            "runtime_filename": row["runtime_filename"],
            "runtime_line": row["runtime_line"],
            "candidate_dtype": row["candidate_dtype"],
            "candidate_shape": row["candidate_shape"],
            "reference_dtype": row["reference_dtype"],
            "full_coordinate_count": error.numel(),
            "signed_error": error.reshape(-1).double().tolist(),
            "rms": float(torch.mean(error.double().square()).sqrt()),
            "max_abs": float(error.abs().max()),
            "reference_cast_changed_coordinates": int(torch.count_nonzero(cast_delta)),
            "reference_cast_max_abs_change": float(cast_delta.abs().max()),
            "delivered_matches_reference_cast": bool(torch.equal(delivered, row["reference_cast"])),
        }


@lru_cache(maxsize=None)
def _runtime_phase(filename: str) -> str:
    header = Path(filename).read_text(errors="ignore")[:512]
    match = re.search(r"# AOT ID: \['\d+_(forward|backward|inference)'\]", header)
    if match is None:
        return "UNKNOWN"
    value = match.group(1).upper()
    return "FORWARD" if value == "INFERENCE" else value


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "device_type": value.device.type,
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "shape": list(value.shape),
        "storage_offset": int(value.storage_offset()),
        "stride": list(value.stride()),
    }


class ExternalCallDiscovery:
    """Record runtime contracts for generated mm/bmm/addmm callsites."""

    SYMBOLS = ("mm", "bmm", "addmm")

    def __init__(self, modules: Iterable[Any]) -> None:
        self.modules = list(modules)
        self.records: list[dict[str, Any]] = []
        self.restores: list[Callable[[], None]] = []

    def __enter__(self) -> "ExternalCallDiscovery":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            for symbol in self.SYMBOLS:
                original = getattr(namespace, symbol, None)
                if not callable(original):
                    continue

                def wrapped(
                    *args: Any, _original: Any = original, _symbol: str = symbol,
                    **kwargs: Any,
                ) -> Any:
                    filename, line, digest = _source_identity()
                    result = _original(*args, **kwargs)
                    contracts: dict[str, Any] = {}
                    for index, value in enumerate(args):
                        if isinstance(value, torch.Tensor):
                            contracts[f"arg{index}"] = _tensor_contract(value)
                    for name, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            contracts[f"kw:{name}"] = _tensor_contract(value)
                    output = kwargs.get("out", result)
                    if isinstance(output, torch.Tensor) and "kw:out" not in contracts:
                        contracts["output"] = _tensor_contract(output)
                    self.records.append({
                        "implementation_kind": "EXTERN",
                        "function": f"extern_kernels.{_symbol}",
                        "phase": _runtime_phase(filename),
                        "runtime_filename": filename,
                        "runtime_line": line,
                        "source_line_sha256": digest,
                        "operand_contracts": contracts,
                    })
                    return result

                setattr(namespace, symbol, wrapped)
                self.restores.append(
                    lambda ns=namespace, name=symbol, value=original: setattr(ns, name, value)
                )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()

    def select(self, exact_payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        expected = dict(exact_payload["operand_contracts"])
        matches = [
            row for row in self.records
            if row["function"] == exact_payload["operation"]
            and row["phase"] == exact_payload["phase"]
            and row["operand_contracts"] == expected
        ]
        if not matches:
            raise RuntimeError(
                "external target contract is absent from the fresh full backward"
            )
        matches.sort(key=lambda row: (
            row["runtime_filename"], row["runtime_line"], row["source_line_sha256"]
        ))
        return matches[0], len(matches)
