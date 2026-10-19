"""PyTorch 2.10 Inductor exact-form GELU forward and complete input VJP."""
from __future__ import annotations

import torch
import torch.nn.functional as F


ROWS, COLS = 32, 128
APPROXIMATE = "none"


def _loss_with_output(x, upstream):
    y = F.gelu(x, approximate=APPROXIMATE)
    return (y * upstream).sum(), y


def _forward_and_vjp(x, upstream):
    dx, (_, y) = torch.func.grad_and_value(
        _loss_with_output, argnums=0, has_aux=True
    )(x, upstream)
    return y, dx


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "upstream", "approximate"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["approximate"] != APPROXIMATE:
        raise ValueError("this call fixes approximate='none'")
    shape = (ROWS, COLS)
    for name in ("x", "upstream"):
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
    y, dx = _compiled(inputs["x"], inputs["upstream"])
    return {"y": y, "dx": dx}
