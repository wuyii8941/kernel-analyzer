"""PyTorch 2.10 Inductor RMSNorm forward and complete public VJP."""
from __future__ import annotations

import torch
import torch.nn.functional as F


BATCH, WIDTH, EPS = 16, 128, 1e-5


def _loss_with_output(x, weight, upstream):
    y = F.rms_norm(x, (WIDTH,), weight, EPS)
    return (y * upstream).sum(), y


def _forward_and_vjp(x, weight, upstream):
    gradients, (_, y) = torch.func.grad_and_value(
        _loss_with_output, argnums=(0, 1), has_aux=True
    )(x, weight, upstream)
    return y, gradients[0], gradients[1]


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "weight", "upstream", "normalized_shape", "eps"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["normalized_shape"] != [WIDTH] or inputs["eps"] != EPS:
        raise ValueError("this call fixes normalized_shape=[128] and eps=1e-5")
    expected = {
        "x": (BATCH, WIDTH),
        "weight": (WIDTH,),
        "upstream": (BATCH, WIDTH),
    }
    for name, shape in expected.items():
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")
    y, dx, dweight = _compiled(inputs["x"], inputs["weight"], inputs["upstream"])
    return {"y": y, "dx": dx, "dweight": dweight}
