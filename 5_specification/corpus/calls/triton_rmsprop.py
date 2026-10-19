"""Direct Triton 3.6 float32 warm-state centered RMSprop transition."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 4096, 256
LR, ALPHA, EPS, WEIGHT_DECAY, MOMENTUM = 0.125, 0.75, 0.25, 0.25, 0.5


@triton.jit
def _rmsprop_kernel(theta_ptr, gradient_ptr, v_ptr, gave_ptr, buffer_ptr,
                    SIZE: tl.constexpr, LR: tl.constexpr, ALPHA: tl.constexpr, EPS: tl.constexpr,
                    WEIGHT_DECAY: tl.constexpr, MOMENTUM: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    theta = tl.load(theta_ptr + offsets, mask=active, other=0.0)
    gradient = tl.load(gradient_ptr + offsets, mask=active, other=0.0)
    old_v = tl.load(v_ptr + offsets, mask=active, other=0.0)
    old_gave = tl.load(gave_ptr + offsets, mask=active, other=0.0)
    old_buffer = tl.load(buffer_ptr + offsets, mask=active, other=0.0)

    processed = -gradient + WEIGHT_DECAY * theta
    new_v = ALPHA * old_v + (1.0 - ALPHA) * processed * processed
    new_gave = ALPHA * old_gave + (1.0 - ALPHA) * processed
    adjusted = new_v - new_gave * new_gave
    new_buffer = MOMENTUM * old_buffer + processed / (tl.sqrt(adjusted) + EPS)
    new_theta = theta - LR * new_buffer

    tl.store(theta_ptr + offsets, new_theta, mask=active)
    tl.store(v_ptr + offsets, new_v, mask=active)
    tl.store(gave_ptr + offsets, new_gave, mask=active)
    tl.store(buffer_ptr + offsets, new_buffer, mask=active)


@triton.jit
def _increment_step_kernel(step_ptr):
    old_step = tl.load(step_ptr)
    tl.store(step_ptr, old_step + 1)


def call(inputs):
    required = {"theta", "gradient", "v", "gave", "b", "t"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    theta, gradient, v, gave, buffer, step = (
        inputs[name] for name in ("theta", "gradient", "v", "gave", "b", "t"))
    for name, value in (("theta", theta), ("gradient", gradient), ("v", v), ("gave", gave), ("b", buffer)):
        if (not torch.is_tensor(value) or value.shape != (SIZE,) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    if bool((v < 0).any()):
        raise ValueError("v must be nonnegative")
    processed = -gradient + WEIGHT_DECAY * theta
    next_v = ALPHA * v + (1.0 - ALPHA) * processed.square()
    next_gave = ALPHA * gave + (1.0 - ALPHA) * processed
    if bool((next_v - next_gave.square() < 0).any()):
        raise ValueError("the selected centered square estimate must be nonnegative")
    if (not torch.is_tensor(step) or step.shape != () or step.dtype != torch.int64
            or not step.is_cuda or not step.is_contiguous() or int(step.item()) != 4):
        raise ValueError("t must be a contiguous CUDA int64 scalar equal to 4")

    _rmsprop_kernel[(triton.cdiv(SIZE, BLOCK),)](
        theta, gradient, v, gave, buffer, SIZE=SIZE, LR=LR, ALPHA=ALPHA, EPS=EPS,
        WEIGHT_DECAY=WEIGHT_DECAY, MOMENTUM=MOMENTUM, BLOCK=BLOCK, num_warps=4)
    _increment_step_kernel[(1,)](step)
    return {"theta": theta, "v": v, "gave": gave, "b": buffer, "t": step}
