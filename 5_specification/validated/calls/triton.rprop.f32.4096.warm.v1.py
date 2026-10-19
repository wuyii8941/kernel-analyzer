"""Direct Triton 3.6 float32 warm-state Rprop transition."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 4096, 256
LR, ETAMINUS, ETAPLUS, STEP_MIN, STEP_MAX = 0.125, 0.5, 1.5, 0.03125, 0.75


@triton.jit
def _rprop_kernel(theta_ptr, gradient_ptr, prev_ptr, eta_ptr,
                  SIZE: tl.constexpr, ETAMINUS: tl.constexpr, ETAPLUS: tl.constexpr,
                  STEP_MIN: tl.constexpr, STEP_MAX: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    theta = tl.load(theta_ptr + offsets, mask=active, other=0.0)
    received = tl.load(gradient_ptr + offsets, mask=active, other=0.0)
    previous = tl.load(prev_ptr + offsets, mask=active, other=0.0)
    old_eta = tl.load(eta_ptr + offsets, mask=active, other=0.0)

    processed = -received
    product = previous * processed
    factor = tl.where(product > 0.0, ETAPLUS, tl.where(product < 0.0, ETAMINUS, 1.0))
    new_eta = tl.minimum(tl.maximum(old_eta * factor, STEP_MIN), STEP_MAX)
    effective = tl.where(product < 0.0, 0.0, processed)
    direction = tl.where(effective > 0.0, 1.0, tl.where(effective < 0.0, -1.0, 0.0))
    new_theta = theta - new_eta * direction

    tl.store(theta_ptr + offsets, new_theta, mask=active)
    tl.store(prev_ptr + offsets, effective, mask=active)
    tl.store(eta_ptr + offsets, new_eta, mask=active)


@triton.jit
def _increment_step(step_ptr):
    tl.store(step_ptr, tl.load(step_ptr) + 1)


def call(inputs):
    required = {"theta", "gradient", "prev", "eta", "t"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    theta, gradient, prev, eta, step = (inputs[name] for name in ("theta", "gradient", "prev", "eta", "t"))
    for name, value in (("theta", theta), ("gradient", gradient), ("prev", prev), ("eta", eta)):
        if (not torch.is_tensor(value) or value.shape != (SIZE,) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    if bool((eta < 0).any()):
        raise ValueError("eta must be nonnegative")
    if (not torch.is_tensor(step) or step.shape != () or step.dtype != torch.int64
            or not step.is_cuda or not step.is_contiguous() or int(step.item()) != 4):
        raise ValueError("t must be a contiguous CUDA int64 scalar equal to 4")

    _rprop_kernel[(triton.cdiv(SIZE, BLOCK),)](
        theta, gradient, prev, eta, SIZE=SIZE, ETAMINUS=ETAMINUS, ETAPLUS=ETAPLUS,
        STEP_MIN=STEP_MIN, STEP_MAX=STEP_MAX, BLOCK=BLOCK, num_warps=4)
    _increment_step[(1,)](step)
    return {"theta": theta, "prev": prev, "eta": eta, "t": step}
