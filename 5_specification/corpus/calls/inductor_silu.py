"""Public PyTorch 2.10 call proposed for the ACT contract-binding review."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _silu(x):
    return F.silu(x)


_compiled = torch.compile(_silu, fullgraph=True, backend="inductor")


def call(inputs):
    """Run one compiled SiLU call and return its complete public output."""
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if not torch.is_tensor(x) or not x.is_floating_point():
        raise ValueError("x must be a floating tensor")
    return {"out": _compiled(x)}
