"""Direct Triton 3.6 float32 BatchNorm evaluation values."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


N, C, H, W = 2, 8, 8, 8
SIZE = N * C * H * W
BLOCK = 256
EPS = 0.125


@triton.jit
def _batch_norm_eval_kernel(x_ptr, mean_ptr, variance_ptr, weight_ptr, bias_ptr, out_ptr,
                            SIZE: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
                            EPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    channel = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    variance = tl.load(variance_ptr + channel, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
    value = (x - mean) * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(out_ptr + offsets, value, mask=mask)


def call(inputs):
    required = {"x", "running_mean", "running_var", "weight", "bias"}
    if set(inputs) != required:
        raise ValueError(f"expected inputs {sorted(required)}, received {sorted(inputs)}")
    tensors = (
        ("x", inputs["x"], (N, C, H, W)),
        ("running_mean", inputs["running_mean"], (C,)),
        ("running_var", inputs["running_var"], (C,)),
        ("weight", inputs["weight"], (C,)),
        ("bias", inputs["bias"], (C,)),
    )
    for name, value, shape in tensors:
        if (not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    running_var = inputs["running_var"]
    if bool((running_var < 0.25).any()) or bool((running_var > 4).any()):
        raise ValueError("running_var must contain values in [0.25, 4]")

    out = torch.empty_like(inputs["x"])
    _batch_norm_eval_kernel[(triton.cdiv(SIZE, BLOCK),)](
        inputs["x"], inputs["running_mean"], running_var, inputs["weight"], inputs["bias"], out,
        SIZE=SIZE, C=C, HW=H * W, EPS=EPS, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
