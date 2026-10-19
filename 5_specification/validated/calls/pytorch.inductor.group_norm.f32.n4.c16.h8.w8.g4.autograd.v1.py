"""PyTorch 2.10 Inductor GroupNorm forward and complete public VJP."""
from __future__ import annotations

import torch
import torch.nn.functional as F


BATCH, CHANNELS, HEIGHT, WIDTH = 4, 16, 8, 8
GROUPS, EPS = 4, 1e-5


def _loss_with_output(x, weight, bias, upstream):
    y = F.group_norm(x, GROUPS, weight, bias, EPS)
    return (y * upstream).sum(), y


def _forward_and_vjp(x, weight, bias, upstream):
    gradients, (_, y) = torch.func.grad_and_value(
        _loss_with_output, argnums=(0, 1, 2), has_aux=True
    )(x, weight, bias, upstream)
    return y, gradients[0], gradients[1], gradients[2]


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "weight", "bias", "upstream", "num_groups", "eps"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["num_groups"] != GROUPS or inputs["eps"] != EPS:
        raise ValueError("this call fixes num_groups=4 and eps=1e-5")
    data_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    expected = {
        "x": data_shape,
        "weight": (CHANNELS,),
        "bias": (CHANNELS,),
        "upstream": data_shape,
    }
    for name, shape in expected.items():
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")
    y, dx, dweight, dbias = _compiled(
        inputs["x"], inputs["weight"], inputs["bias"], inputs["upstream"]
    )
    return {"y": y, "dx": dx, "dweight": dweight, "dbias": dbias}
