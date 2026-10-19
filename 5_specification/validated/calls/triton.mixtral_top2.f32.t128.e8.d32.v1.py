"""Direct Triton 3.6 float32 Mixtral top-2 routing and supplied-expert combination."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


TOKENS, EXPERTS, WIDTH = 128, 8, 32


@triton.jit
def _mixtral_top2(logits_ptr, experts_ptr, selected_ptr, weights_ptr, out_ptr,
                  EXPERTS: tl.constexpr, WIDTH: tl.constexpr):
    token = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    coordinate = tl.arange(0, WIDTH)
    logits = tl.load(logits_ptr + token * EXPERTS + expert)
    first_index = tl.argmax(logits, axis=0)
    first_value = tl.max(logits, axis=0)
    remaining = tl.where(expert == first_index, -float("inf"), logits)
    second_index = tl.argmax(remaining, axis=0)
    second_value = tl.max(remaining, axis=0)
    second_mass = tl.exp(second_value - first_value)
    first_weight = 1.0 / (1.0 + second_mass)
    second_weight = second_mass / (1.0 + second_mass)
    selected = (expert == first_index) | (expert == second_index)
    weights = tl.where(expert == first_index, first_weight,
                       tl.where(expert == second_index, second_weight, 0.0))
    values = tl.load(
        experts_ptr + token * EXPERTS * WIDTH + expert[:, None] * WIDTH + coordinate[None, :]
    )
    out = tl.sum(weights[:, None] * values, axis=0)
    tl.store(selected_ptr + token * EXPERTS + expert, selected)
    tl.store(weights_ptr + token * EXPERTS + expert, weights)
    tl.store(out_ptr + token * WIDTH + coordinate, out)


def call(inputs):
    required = {"router_logits", "expert_outputs"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    logits, experts = inputs["router_logits"], inputs["expert_outputs"]
    expected = {"router_logits": (TOKENS, EXPERTS), "expert_outputs": (TOKENS, EXPERTS, WIDTH)}
    for name, tensor in (("router_logits", logits), ("expert_outputs", experts)):
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != expected[name] or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {expected[name]}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")
    ordered = torch.sort(logits, dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("router_logits must be pairwise distinct within every token")
    selected = torch.empty_like(logits, dtype=torch.bool)
    weights = torch.empty_like(logits)
    out = torch.empty((TOKENS, WIDTH), dtype=torch.float32, device=logits.device)
    _mixtral_top2[(TOKENS,)](
        logits, experts, selected, weights, out, EXPERTS=EXPERTS, WIDTH=WIDTH, num_warps=1,
    )
    return {"selected": selected, "weights": weights, "out": out}
