"""PyTorch 2.10 Inductor sigmoid forward and complete input VJP."""
from __future__ import annotations

import torch


ROWS, COLS = 32, 128


def _loss_with_output(x, upstream):
    y = torch.sigmoid(x)
    return (y * upstream).sum(), y


def _forward_and_vjp(x, upstream):
    dx, (_, y) = torch.func.grad_and_value(
        _loss_with_output, argnums=0, has_aux=True
    )(x, upstream)
    return y, dx


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "upstream"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    shape = (ROWS, COLS)
    for name in ("x", "upstream"):
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32 or not tensor.is_cuda
                or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
    if not bool((inputs["x"].abs() <= 100).all()):
        raise ValueError("x must satisfy the MATHFN real profile bound |x| <= 100")
    y, dx = _compiled(inputs["x"], inputs["upstream"])
    return {"y": y, "dx": dx}
