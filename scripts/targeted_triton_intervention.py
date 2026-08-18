"""Observe or repair one exact output of one generated Triton invocation."""

from __future__ import annotations

from pathlib import Path
import ast
import hashlib
from typing import Any, Callable, Iterable, Mapping

import torch

from forkcert.generated_compute_dataflow_audit import _triton_access_modes
from scripts.generated_fp32_observer import (
    promoted_pointer_arguments, runtime_signature, validate_compiled_triton_replay_abi,
)


class TargetedTritonIntervention:
    MODES = {"OBSERVE", "SHAM", "REPAIR"}

    def __init__(
        self, *, modules: Iterable[Any], target: Mapping[str, Any], mode: str,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported intervention mode: {mode}")
        if target["implementation_kind"] != "TRITON":
            raise ValueError("targeted Triton intervention requires a TRITON target")
        self.modules = list(modules)
        self.target = dict(target)
        self.mode = mode
        self.invocation_count = 0
        self.target_count = 0
        self.record: dict[str, Any] | None = None
        self.restores: list[tuple[Any, bool, Any]] = []

    def _validate_program(self) -> None:
        symbol = self.target["symbol"]
        observed = set()
        for module in self.modules:
            tree = ast.parse(Path(module.__file__).read_text())
            modes = _triton_access_modes(tree)
            if symbol in modes:
                observed.add(str(modes[symbol]["embedded_program_sha256"]))
        expected = str(self.target["embedded_program_sha256"])
        if observed != {expected}:
            raise RuntimeError(
                f"target Triton program identity mismatch: expected={expected} observed={observed}"
            )

    def __enter__(self) -> "TargetedTritonIntervention":
        self._validate_program()
        symbol = str(self.target["symbol"])
        found = []
        seen: set[int] = set()
        for module in self.modules:
            kernel = getattr(module, symbol, None)
            if kernel is None or id(kernel) in seen or not callable(getattr(kernel, "run", None)):
                continue
            seen.add(id(kernel))
            found.append(kernel)
        if len(found) != 1:
            raise RuntimeError(f"target Triton symbol resolved to {len(found)} runtime kernels")
        kernel = found[0]
        had_run = "run" in vars(kernel)
        previous = vars(kernel).get("run")
        original = kernel.run

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            index = self.invocation_count
            self.invocation_count += 1
            if index != int(self.target.get("invocation_index", 0)):
                return original(*args, **kwargs)
            self.target_count += 1
            if self.target_count != 1:
                raise RuntimeError("target Triton invocation executed more than once")
            pointer_names = [
                name for name, annotation in runtime_signature(kernel)
                if str(annotation).startswith("*")
            ]
            tensor_args = [value for value in args if isinstance(value, torch.Tensor)]
            if len(pointer_names) != len(tensor_args):
                raise RuntimeError("target Triton pointer signature mismatch")
            promoted_args, promoted_tensors = promoted_pointer_arguments(args)
            validate_compiled_triton_replay_abi(kernel, args, promoted_args)
            candidate_pointers = dict(zip(pointer_names, tensor_args))
            reference_pointers = dict(zip(pointer_names, promoted_tensors))
            output_name = str(self.target["endpoint"])
            if output_name not in candidate_pointers:
                raise RuntimeError(f"target output pointer absent: {output_name}")
            result = original(*args, **kwargs)
            original(*promoted_args, **kwargs)
            candidate = candidate_pointers[output_name]
            reference = reference_pointers[output_name]
            candidate_before = candidate.detach().float().cpu().clone()
            reference_cpu = reference.detach().float().cpu().clone()
            reference_cast = reference.to(dtype=candidate.dtype).detach().float().cpu().clone()
            if self.mode == "REPAIR":
                candidate.copy_(reference.to(dtype=candidate.dtype))
            elif self.mode == "SHAM":
                candidate.copy_(candidate.clone())
            self.record = {
                "candidate_dtype": str(candidate.dtype),
                "candidate_shape": list(candidate.shape),
                "candidate_before": candidate_before,
                "reference": reference_cpu,
                "reference_cast": reference_cast,
                "delivered": candidate.detach().float().cpu().clone(),
            }
            return result

        kernel.run = wrapped
        self.restores.append((kernel, had_run, previous))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for kernel, had_run, previous in reversed(self.restores):
            if had_run:
                kernel.run = previous
            else:
                delattr(kernel, "run")
        self.restores.clear()

    def summary(self) -> dict[str, Any]:
        if self.target_count != 1 or self.record is None:
            raise RuntimeError(f"target Triton execution count is {self.target_count}, expected 1")
        row = self.record
        error = row["candidate_before"] - row["reference"]
        cast_delta = row["candidate_before"] - row["reference_cast"]
        flat_error = error.reshape(-1).double().contiguous()
        summary = {
            "mode": self.mode,
            "symbol": self.target["symbol"],
            "region_id": self.target["region_id"],
            "endpoint": self.target["endpoint"],
            "candidate_dtype": row["candidate_dtype"],
            "candidate_shape": row["candidate_shape"],
            "full_coordinate_count": int(error.numel()),
            "signed_error_sha256": hashlib.sha256(flat_error.numpy().tobytes()).hexdigest(),
            "rms": float(torch.mean(error.double().square()).sqrt()),
            "max_abs": float(error.abs().max()),
            "reference_cast_changed_coordinates": int(torch.count_nonzero(cast_delta)),
            "reference_cast_max_abs_change": float(cast_delta.abs().max()),
            "delivered_matches_reference_cast": bool(
                torch.equal(row["delivered"], row["reference_cast"])
            ),
        }
        if flat_error.numel() <= 4096:
            summary["signed_error"] = flat_error.tolist()
        else:
            summary["signed_error_storage"] = "HASH_ONLY_LARGE_ENDPOINT"
        return summary
