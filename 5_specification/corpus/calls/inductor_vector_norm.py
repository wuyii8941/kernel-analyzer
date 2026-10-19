"""PyTorch 2.10 Inductor public vector 2-norm and complete input VJP."""
from __future__ import annotations

import torch


ROWS, COLS, DIM = 16, 128, 1
ORD, KEEPDIM = 2, False


def _loss_with_output(x, upstream):
    y = torch.linalg.vector_norm(x, ord=ORD, dim=DIM, keepdim=KEEPDIM)
    return (y * upstream).sum(), y


def _forward_and_vjp(x, upstream):
    dx, (_, y) = torch.func.grad_and_value(
        _loss_with_output, argnums=0, has_aux=True
    )(x, upstream)
    return y, dx


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "upstream", "ord", "dim", "keepdim"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if (inputs["ord"] != ORD or inputs["dim"] != DIM
            or inputs["keepdim"] is not KEEPDIM):
        raise ValueError("this call fixes ord=2, dim=1 and keepdim=False")
    tensors = {
        "x": ((ROWS, COLS), inputs["x"]),
        "upstream": ((ROWS,), inputs["upstream"]),
    }
    for name, (shape, tensor) in tensors.items():
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32 or not tensor.is_cuda
                or not tensor.is_contiguous()):
            raise ValueError(
                f"{name} must be a contiguous CUDA float32 tensor of shape {shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
    y, dx = _compiled(inputs["x"], inputs["upstream"])
    return {"y": y, "dx": dx}
