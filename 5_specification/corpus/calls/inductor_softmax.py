"""Public PyTorch 2.10 call proposed for the SOFTMAX contract-binding review."""
from __future__ import annotations

import torch


def _softmax(x):
    return torch.softmax(x, dim=-1)


_compiled = torch.compile(_softmax, fullgraph=True, backend="inductor")


def call(inputs):
    """Run one fixed rank-2 softmax and return its complete public output."""
    required = {"x", "dim", "half_to_float"}
    if set(inputs) != required:
        raise ValueError(f"expected inputs {sorted(required)}, received {sorted(inputs)}")
    x = inputs["x"]
    if not torch.is_tensor(x) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a rank-2 floating tensor")
    if inputs["dim"] != -1 or inputs["half_to_float"] is not False:
        raise ValueError("this candidate fixes dim=-1 and half_to_float=False")
    return {"out": _compiled(x)}
