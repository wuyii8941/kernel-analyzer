from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenGradContribution:
    case_id: str
    token_index: int
    branch_active: bool
    contribution_norm: float


def ppo_token_surrogate(logp_new: Any, old_logp: Any, advantage: Any, eps: float) -> Any:
    """Return PPO clipped surrogate for one token/sample.

    The returned value is a maximization objective. Training losses usually negate it.
    """
    torch = _require_torch()
    ratio = torch.exp(logp_new - old_logp)
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    return torch.minimum(ratio * advantage, clipped * advantage)


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("gradient contribution requires torch") from exc


def global_grad_norm(parameters: Any) -> float:
    torch = _require_torch()
    total = torch.zeros((), dtype=torch.float32)
    for param in parameters:
        if param.grad is None:
            continue
        total = total + torch.sum(param.grad.detach().float() ** 2)
    return float(torch.sqrt(total).item())


def branch_gradient_expected_zero(clip_active: bool) -> bool:
    return bool(clip_active)

