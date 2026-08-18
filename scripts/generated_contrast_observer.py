"""Batched full-coordinate precision and same-dtype generated-call observer."""

from __future__ import annotations

import hashlib
import linecache
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from scripts.generated_nontriton_fp32_observer import (
    _AttributeProxy,
    _promote_args,
    fp32_external_reference,
)


TensorSink = Callable[[str, str, str, torch.Tensor, Mapping[str, Any]], None]


def _source_identity() -> tuple[str, int, str]:
    caller = sys._getframe(2)
    line = int(caller.f_lineno)
    source = linecache.getline(caller.f_code.co_filename, line).strip()
    return caller.f_code.co_filename, line, hashlib.sha256(source.encode()).hexdigest()


def _semantic_hash(function: str) -> str:
    return hashlib.sha256(("DECLARED_OP\0" + function).encode()).hexdigest()


def _program_hash(role: str, function: str, source_sha: str = "") -> str:
    return hashlib.sha256((role + "\0" + function + "\0" + source_sha).encode()).hexdigest()


def low_external_reference(
    symbol: str, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> torch.Tensor:
    tensors = [value for value in args if isinstance(value, torch.Tensor)]
    with torch.no_grad():
        if symbol == "mm":
            return torch.mm(tensors[0], tensors[1])
        if symbol == "bmm":
            return torch.bmm(tensors[0], tensors[1])
        if symbol == "addmm":
            return torch.addmm(
                tensors[0], tensors[1], tensors[2],
                beta=kwargs.get("beta", 1), alpha=kwargs.get("alpha", 1),
            )
        if symbol == "convolution":
            bias = tensors[2] if len(tensors) == 3 else kwargs.get("bias")
            return torch.ops.aten.convolution.default(
                tensors[0], tensors[1], bias, list(kwargs["stride"]),
                list(kwargs["padding"]), list(kwargs["dilation"]),
                bool(kwargs["transposed"]), list(kwargs["output_padding"]),
                int(kwargs["groups"]),
            )
    raise ValueError("unsupported external symbol: %s" % symbol)


class BatchedGeneratedContrastObserver:
    """Observe selected generated regions without retaining full tensors in RAM."""

    def __init__(
        self, *, modules: Iterable[Any], targets: Sequence[Mapping[str, Any]], sink: TensorSink,
    ) -> None:
        self.modules = list(modules)
        self.targets = {
            str(row["exact_generated_call"]["source_line_sha256"]): dict(row)
            for row in targets
        }
        if len(self.targets) != len(targets):
            raise ValueError("candidate source identities are not unique within a cell")
        self.sink = sink
        self.counts = {key: 0 for key in self.targets}
        self.restores: list[Callable[[], None]] = []

    def _emit(
        self, target: Mapping[str, Any], candidate: torch.Tensor,
        low: torch.Tensor, high: torch.Tensor, endpoint: str,
    ) -> None:
        call = target["exact_generated_call"]
        metadata = {
            "candidate_dtype": str(candidate.dtype),
            "low_dtype": str(low.dtype),
            "high_dtype": str(high.dtype),
            "shape": list(candidate.shape),
            "semantic_boundary_exact": True,
            "semantic_program_sha256": _semantic_hash(str(call["function"])),
            "candidate_program_sha256": _program_hash(
                "candidate", str(call["function"]), str(call["source_line_sha256"])
            ),
            "low_arm_program_sha256": _program_hash("low", str(call["function"])),
            "high_arm_program_sha256": _program_hash("high", str(call["function"])),
        }
        candidate_f = candidate.detach().float()
        low_f = low.detach().float()
        high_f = high.detach().float()
        self.sink(target["candidate_id"], "PRECISION", endpoint, low_f - high_f, metadata)
        self.sink(target["candidate_id"], "OPTIMIZATION", endpoint, candidate_f - low_f, metadata)
        self.sink(target["candidate_id"], "TOTAL", endpoint, candidate_f - high_f, metadata)

    def __enter__(self) -> "BatchedGeneratedContrastObserver":
        self._install_externals()
        self._install_convolution_backward()
        return self

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

                def wrapped(*args: Any, _original: Any = original,
                            _symbol: str = symbol, **kwargs: Any) -> Any:
                    _, _, digest = _source_identity()
                    target = self.targets.get(digest)
                    if target is None:
                        return _original(*args, **kwargs)
                    high = fp32_external_reference(_symbol, args, kwargs)
                    low = low_external_reference(_symbol, args, kwargs)
                    result = _original(*args, **kwargs)
                    candidate = kwargs.get("out", result)
                    if not isinstance(candidate, torch.Tensor):
                        raise TypeError("generated external call produced no tensor")
                    endpoint = str(target["sampled_t1"]["endpoint"])
                    self._emit(target, candidate, low, high, endpoint)
                    self.counts[digest] += 1
                    return result

                setattr(namespace, symbol, wrapped)
                self.restores.append(
                    lambda ns=namespace, name=symbol, value=original: setattr(ns, name, value)
                )

    def _install_convolution_backward(self) -> None:
        for module in self.modules:
            torch_namespace = getattr(module, "torch", None)
            if torch_namespace is None:
                continue
            original = torch_namespace.ops.aten.convolution_backward.default

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _, _, digest = _source_identity()
                target = self.targets.get(digest)
                if target is None:
                    return _original(*args, **kwargs)
                promoted = _promote_args(args)
                with torch.no_grad():
                    high = torch.ops.aten.convolution_backward.default(*promoted, **kwargs)
                    low = torch.ops.aten.convolution_backward.default(*args, **kwargs)
                result = _original(*args, **kwargs)
                endpoint = str(target["sampled_t1"]["endpoint"])
                index = int(endpoint.rsplit("_", 1)[-1])
                self._emit(target, result[index], low[index], high[index], endpoint)
                self.counts[digest] += 1
                return result

            proxy = _AttributeProxy(
                torch_namespace.ops.aten.convolution_backward, {"default": wrapped}
            )
            convolution_proxy = _AttributeProxy(
                torch_namespace.ops.aten, {"convolution_backward": proxy}
            )
            ops_proxy = _AttributeProxy(torch_namespace.ops, {"aten": convolution_proxy})
            module.torch = _AttributeProxy(torch_namespace, {"ops": ops_proxy})
            self.restores.append(
                lambda target=module, value=torch_namespace: setattr(target, "torch", value)
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()

    def validate(self) -> None:
        missing = [key for key, count in self.counts.items() if count != 1]
        if missing:
            raise RuntimeError("selected generated regions did not execute exactly once: %s" % missing)
