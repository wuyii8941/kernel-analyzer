"""Capture and compare MoE routing without changing model execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import torch


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class MoERoutingRecorder:
    """Record top-k decisions and margins from router-logit modules."""

    def __init__(self, routers: Mapping[str, torch.nn.Module], top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.routers = dict(routers)
        self.top_k = top_k
        self.rows: list[dict[str, Any]] = []
        self._handles: list[Any] = []

    def __enter__(self) -> "MoERoutingRecorder":
        for name, module in sorted(self.routers.items()):
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *unused: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor) or logits.shape[-1] <= self.top_k:
                raise RuntimeError(f"router {name} output is not valid logits")
            probabilities = torch.softmax(logits.detach().float(), dim=-1)
            values, indices = torch.topk(probabilities, self.top_k + 1, dim=-1)
            selected = indices[..., : self.top_k].cpu()
            margin = (values[..., self.top_k - 1] - values[..., self.top_k]).cpu()
            loads = torch.bincount(
                selected.reshape(-1), minlength=logits.shape[-1],
            ).cpu()
            row = {
                "router": name,
                "invocation": sum(item["router"] == name for item in self.rows),
                "top_k": self.top_k,
                "expert_count": int(logits.shape[-1]),
                "token_shape": list(logits.shape[:-1]),
                "selected_experts": selected.tolist(),
                "topk_margin": margin.tolist(),
                "expert_load": loads.tolist(),
            }
            row["routing_digest"] = _digest(row)
            self.rows.append(row)
        return capture

    def certificate(self) -> dict[str, Any]:
        payload = {
            "schema": "kernel-analyzer-moe-routing-capture-v1",
            "status": "COMPLETE_ROUTING_CAPTURE",
            "rows": self.rows,
            "candidate_tensor_values_used_for_case_selection": False,
        }
        payload["result_sha256"] = _digest(payload)
        return payload


def compare_routing(
    candidate_rows: Iterable[Mapping[str, Any]],
    repair_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = list(candidate_rows)
    repair = list(repair_rows)
    if len(candidate) != len(repair):
        raise ValueError("candidate and repair routing invocation counts differ")
    total = 0
    flipped = 0
    first = None
    layers = []
    for left, right in zip(candidate, repair):
        identity = (left["router"], left["invocation"])
        if identity != (right["router"], right["invocation"]):
            raise ValueError("candidate and repair router identities differ")
        left_ids = torch.tensor(left["selected_experts"], dtype=torch.int64)
        right_ids = torch.tensor(right["selected_experts"], dtype=torch.int64)
        if left_ids.shape != right_ids.shape:
            raise ValueError("candidate and repair routing shapes differ")
        token_flip = (left_ids != right_ids).any(dim=-1)
        layer_flips = int(token_flip.sum())
        layer_total = int(token_flip.numel())
        if layer_flips and first is None:
            index = torch.nonzero(token_flip, as_tuple=False)[0].tolist()
            first = {"router": identity[0], "invocation": identity[1], "token_index": index}
        total += layer_total
        flipped += layer_flips
        layers.append({
            "router": identity[0], "invocation": identity[1],
            "flipped_tokens": layer_flips, "token_denominator": layer_total,
            "candidate_expert_load": left["expert_load"],
            "repair_expert_load": right["expert_load"],
        })
    return {
        "schema": "kernel-analyzer-moe-routing-comparison-v1",
        "routing_regime": "SAME_ROUTE_SMOOTH_TRANSPORT" if flipped == 0 else "DISCRETE_ROUTING_REGIME",
        "flipped_tokens": flipped,
        "token_denominator": total,
        "hamming_rate": flipped / total if total else 0.0,
        "first_divergence": first,
        "layers": layers,
        "removed_from_coverage_denominator": False,
    }
