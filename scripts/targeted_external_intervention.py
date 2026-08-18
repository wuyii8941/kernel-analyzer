"""Observe or repair one exact generated external callsite."""

from __future__ import annotations

import hashlib
import linecache
import sys
from typing import Any, Callable, Iterable, Mapping

import torch

from scripts.generated_nontriton_fp32_observer import fp32_external_reference


def _source_identity() -> tuple[str, int, str]:
    caller = sys._getframe(2)
    line = int(caller.f_lineno)
    source = linecache.getline(caller.f_code.co_filename, line).strip()
    return caller.f_code.co_filename, line, hashlib.sha256(source.encode()).hexdigest()


class TargetedExternalIntervention:
    """Bind one call by source SHA and optionally replace its output with FP32."""

    MODES = {"OBSERVE", "SHAM", "REPAIR"}

    def __init__(
        self, *, modules: Iterable[Any], target: Mapping[str, Any], mode: str,
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
                candidate_before = candidate.detach().float().cpu().clone()
                reference_cpu = reference.detach().float().cpu().clone()
                reference_cast_cpu = reference.to(dtype=candidate.dtype).detach().float().cpu().clone()
                if self.mode == "REPAIR":
                    candidate.copy_(reference.to(dtype=candidate.dtype))
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
                    "delivered": candidate.detach().float().cpu().clone(),
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
