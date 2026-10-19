"""Direct Triton 3.6 float32 warm-buffer Nesterov SGD state transition."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 4096, 256
LR, MOMENTUM, WEIGHT_DECAY = 0.125, 0.75, 0.25


@triton.jit
def _sgd_kernel(theta_ptr, gradient_ptr, buffer_ptr,
                SIZE: tl.constexpr, LR: tl.constexpr, MOMENTUM: tl.constexpr,
                WEIGHT_DECAY: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    theta = tl.load(theta_ptr + offsets, mask=active, other=0.0)
    gradient = tl.load(gradient_ptr + offsets, mask=active, other=0.0)
    old_buffer = tl.load(buffer_ptr + offsets, mask=active, other=0.0)
    processed = -gradient + WEIGHT_DECAY * theta
    new_buffer = MOMENTUM * old_buffer + processed
    direction = processed + MOMENTUM * new_buffer
    new_theta = theta - LR * direction
    tl.store(theta_ptr + offsets, new_theta, mask=active)
    tl.store(buffer_ptr + offsets, new_buffer, mask=active)


def call(inputs):
    if set(inputs) != {"theta", "gradient", "b"}:
        raise ValueError(f"expected b, gradient and theta, received {sorted(inputs)}")
    theta, gradient, buffer = inputs["theta"], inputs["gradient"], inputs["b"]
    for name, value in (("theta", theta), ("gradient", gradient), ("b", buffer)):
        if (not torch.is_tensor(value) or value.shape != (SIZE,) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")

    _sgd_kernel[(triton.cdiv(SIZE, BLOCK),)](
        theta, gradient, buffer, SIZE=SIZE, LR=LR, MOMENTUM=MOMENTUM,
        WEIGHT_DECAY=WEIGHT_DECAY, BLOCK=BLOCK, num_warps=4)
    return {"theta": theta, "b": buffer}
