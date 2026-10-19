"""Direct Triton 3.6 float32 warm-state coupled Adam/AMSGrad transition."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 4096, 256
LR, BETA1, BETA2, EPS, WEIGHT_DECAY = 0.125, 0.5, 0.75, 0.25, 0.25
BIAS_CORRECTION1, BIAS_CORRECTION2 = 31.0 / 32.0, 781.0 / 1024.0


@triton.jit
def _adam_kernel(theta_ptr, gradient_ptr, m_ptr, v_ptr, vmax_ptr,
                 SIZE: tl.constexpr, LR: tl.constexpr, BETA1: tl.constexpr, BETA2: tl.constexpr,
                 EPS: tl.constexpr, WEIGHT_DECAY: tl.constexpr, BIAS_CORRECTION1: tl.constexpr,
                 BIAS_CORRECTION2: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    theta = tl.load(theta_ptr + offsets, mask=active, other=0.0)
    gradient = tl.load(gradient_ptr + offsets, mask=active, other=0.0)
    old_m = tl.load(m_ptr + offsets, mask=active, other=0.0)
    old_v = tl.load(v_ptr + offsets, mask=active, other=0.0)
    old_vmax = tl.load(vmax_ptr + offsets, mask=active, other=0.0)

    processed = -gradient + WEIGHT_DECAY * theta
    new_m = BETA1 * old_m + (1.0 - BETA1) * processed
    new_v = BETA2 * old_v + (1.0 - BETA2) * processed * processed
    new_vmax = tl.maximum(old_vmax, new_v)
    mhat = new_m / BIAS_CORRECTION1
    vhat = new_vmax / BIAS_CORRECTION2
    new_theta = theta - LR * mhat / (tl.sqrt(vhat) + EPS)

    tl.store(theta_ptr + offsets, new_theta, mask=active)
    tl.store(m_ptr + offsets, new_m, mask=active)
    tl.store(v_ptr + offsets, new_v, mask=active)
    tl.store(vmax_ptr + offsets, new_vmax, mask=active)


@triton.jit
def _increment_step_kernel(step_ptr):
    old_step = tl.load(step_ptr)
    tl.store(step_ptr, old_step + 1)


def call(inputs):
    required = {"theta", "gradient", "m", "v", "vmax", "t"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    theta, gradient, m, v, vmax, step = (inputs[name] for name in ("theta", "gradient", "m", "v", "vmax", "t"))
    for name, value in (("theta", theta), ("gradient", gradient), ("m", m), ("v", v), ("vmax", vmax)):
        if (not torch.is_tensor(value) or value.shape != (SIZE,) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    if bool((v < 0).any()) or bool((vmax < 0).any()):
        raise ValueError("v and vmax must be nonnegative")
    if (not torch.is_tensor(step) or step.shape != () or step.dtype != torch.int64
            or not step.is_cuda or not step.is_contiguous() or int(step.item()) != 4):
        raise ValueError("t must be a contiguous CUDA int64 scalar equal to 4")

    _adam_kernel[(triton.cdiv(SIZE, BLOCK),)](
        theta, gradient, m, v, vmax, SIZE=SIZE, LR=LR, BETA1=BETA1, BETA2=BETA2, EPS=EPS,
        WEIGHT_DECAY=WEIGHT_DECAY, BIAS_CORRECTION1=BIAS_CORRECTION1,
        BIAS_CORRECTION2=BIAS_CORRECTION2, BLOCK=BLOCK, num_warps=4)
    _increment_step_kernel[(1,)](step)
    return {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": step}
