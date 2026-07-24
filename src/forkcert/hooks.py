from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class TensorSummary:
    name: str
    shape: list[int]
    dtype: str
    max_abs: float
    mean_abs: float
    sum_abs: float
    l2: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairSummary:
    name: str
    ref: TensorSummary
    alt: TensorSummary
    diff: TensorSummary
    rel_l2: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapturedActivation:
    name: str
    invocation: int
    tensor: Any


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("hooks require torch") from exc


def summarize_tensor(name: str, tensor: Any) -> TensorSummary:
    torch = _require_torch()
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if not torch.is_tensor(tensor):
        raise TypeError(f"module output for {name} is not a tensor or tensor tuple")
    detached = tensor.detach().float()
    abs_value = detached.abs()
    return TensorSummary(
        name=name,
        shape=list(detached.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        max_abs=float(abs_value.max().item()) if detached.numel() else 0.0,
        mean_abs=float(abs_value.mean().item()) if detached.numel() else 0.0,
        sum_abs=float(abs_value.sum().item()) if detached.numel() else 0.0,
        l2=float(torch.linalg.vector_norm(detached).item()) if detached.numel() else 0.0,
    )


def summarize_pair(name: str, ref: Any, alt: Any) -> PairSummary:
    torch = _require_torch()
    ref_tensor = ref[0] if isinstance(ref, (tuple, list)) else ref
    alt_tensor = alt[0] if isinstance(alt, (tuple, list)) else alt
    if not torch.is_tensor(ref_tensor) or not torch.is_tensor(alt_tensor):
        raise TypeError(f"pair output for {name} is not tensor-like")
    diff = alt_tensor.detach().float() - ref_tensor.detach().float()
    ref_summary = summarize_tensor(f"{name}.ref", ref_tensor)
    alt_summary = summarize_tensor(f"{name}.alt", alt_tensor)
    diff_summary = summarize_tensor(f"{name}.diff", diff)
    rel_l2 = diff_summary.l2 / ref_summary.l2 if ref_summary.l2 > 0 else 0.0
    return PairSummary(name=name, ref=ref_summary, alt=alt_summary, diff=diff_summary, rel_l2=rel_l2)


class ActivationRecorder:
    """Records compact activation summaries from selected modules."""

    def __init__(self, name_filter: Callable[[str, Any], bool] | None = None, max_modules: int | None = None):
        self.name_filter = name_filter or (lambda _name, _module: True)
        self.max_modules = max_modules
        self.records: list[TensorSummary] = []
        self._handles: list[Any] = []

    def attach(self, model: Any) -> "ActivationRecorder":
        count = 0
        for name, module in model.named_modules():
            if not name or not self.name_filter(name, module):
                continue
            if self.max_modules is not None and count >= self.max_modules:
                break
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
            count += 1
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                self.records.append(summarize_tensor(name, output))
            except TypeError:
                return

        return hook

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


class ActivationTensorRecorder:
    """Captures selected module outputs on CPU for exact paired tensor differences."""

    def __init__(self, name_filter: Callable[[str, Any], bool] | None = None, max_modules: int | None = None):
        self.name_filter = name_filter or (lambda _name, _module: True)
        self.max_modules = max_modules
        self.records: list[CapturedActivation] = []
        self._handles: list[Any] = []
        self._counts: dict[str, int] = {}

    def attach(self, model: Any) -> "ActivationTensorRecorder":
        count = 0
        for name, module in model.named_modules():
            normalized = name.removeprefix("_orig_mod.")
            if not normalized or not self.name_filter(normalized, module):
                continue
            if self.max_modules is not None and count >= self.max_modules:
                break
            self._handles.append(module.register_forward_hook(self._make_hook(normalized)))
            count += 1
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            torch = _require_torch()
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                return
            invocation = self._counts.get(name, 0)
            self._counts[name] = invocation + 1
            self.records.append(CapturedActivation(name, invocation, tensor.detach().to("cpu")))

        return hook


def force_dtype_roundtrip_hook(dtype: Any):
    torch = _require_torch()

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            return output.to(dtype).to(output.dtype)
        if isinstance(output, tuple):
            return tuple(item.to(dtype).to(item.dtype) if torch.is_tensor(item) else item for item in output)
        return output

    return hook


def attach_dtype_roundtrip(model: Any, dtype: Any, name_filter: Callable[[str, Any], bool] | None = None) -> list[Any]:
    filt = name_filter or (lambda _name, _module: True)
    handles = []
    for name, module in model.named_modules():
        if name and filt(name, module):
            handles.append(module.register_forward_hook(force_dtype_roundtrip_hook(dtype)))
    return handles


def attach_bf16_roundtrip(model: Any, name_filter: Callable[[str, Any], bool] | None = None) -> list[Any]:
    torch = _require_torch()
    return attach_dtype_roundtrip(model, torch.bfloat16, name_filter)


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()
