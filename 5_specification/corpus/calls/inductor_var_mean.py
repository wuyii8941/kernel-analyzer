"""PyTorch 2.10 Inductor variance/mean pair and complete combined input VJP."""
from __future__ import annotations

import torch


ROWS, COLS, DIM = 16, 128, 1
CORRECTION, KEEPDIM = 1, False


def _loss_with_output(x, upstream_var, upstream_mean):
    variance, mean = torch.var_mean(
        x, dim=DIM, correction=CORRECTION, keepdim=KEEPDIM
    )
    loss = (variance * upstream_var + mean * upstream_mean).sum()
    return loss, (variance, mean)


def _forward_and_vjp(x, upstream_var, upstream_mean):
    dx, (_, (variance, mean)) = torch.func.grad_and_value(
        _loss_with_output, argnums=0, has_aux=True
    )(x, upstream_var, upstream_mean)
    return variance, mean, dx


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {
        "x", "upstream_var", "upstream_mean", "dim", "correction", "keepdim"
    }
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if (inputs["dim"] != DIM or inputs["correction"] != CORRECTION
            or inputs["keepdim"] is not KEEPDIM):
        raise ValueError("this call fixes dim=1, correction=1 and keepdim=False")
    tensors = {
        "x": ((ROWS, COLS), inputs["x"]),
        "upstream_var": ((ROWS,), inputs["upstream_var"]),
        "upstream_mean": ((ROWS,), inputs["upstream_mean"]),
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
    variance, mean, dx = _compiled(
        inputs["x"], inputs["upstream_var"], inputs["upstream_mean"]
    )
    return {"variance": variance, "mean": mean, "dx": dx}
