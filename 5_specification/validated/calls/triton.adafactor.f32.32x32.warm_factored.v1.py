"""Direct Triton 3.6 float32 warm-state factored Adafactor transition."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLS = 32, 32
LR, EPS1, EPS2, CLIP, WEIGHT_DECAY = 0.125, 0.25, 0.25, 1.0, 0.25


@triton.jit
def _adafactor_kernel(theta_ptr, gradient_ptr, row_ptr, col_ptr, step_ptr,
                      ROWS: tl.constexpr, COLS: tl.constexpr, LR: tl.constexpr,
                      EPS1: tl.constexpr, EPS2: tl.constexpr, CLIP: tl.constexpr,
                      WEIGHT_DECAY: tl.constexpr):
    row_ids = tl.arange(0, ROWS)[:, None]
    col_ids = tl.arange(0, COLS)[None, :]
    offsets = row_ids * COLS + col_ids
    theta = tl.load(theta_ptr + offsets)
    received_gradient = tl.load(gradient_ptr + offsets)
    old_row = tl.load(row_ptr + row_ids)
    old_col = tl.load(col_ptr + col_ids)
    old_step = tl.load(step_ptr)

    gradient = -received_gradient
    step = old_step + 1
    one_minus_beta2 = 1.0 / step
    rho = tl.minimum(LR, 1.0 / tl.sqrt(step.to(tl.float32)))
    theta_rms = tl.sqrt(tl.sum(tl.sum(theta * theta, axis=1), axis=0) / (ROWS * COLS))
    alpha = tl.maximum(EPS2, theta_rms) * rho

    squared = gradient * gradient
    row_target = tl.expand_dims(tl.sum(squared, axis=1) / COLS, 1)
    col_target = tl.expand_dims(tl.sum(squared, axis=0) / ROWS, 0)
    new_row = old_row + one_minus_beta2 * (row_target - old_row)
    new_col = old_col + one_minus_beta2 * (col_target - old_col)
    row_mean = tl.sum(new_row, axis=0) / ROWS
    estimate = new_row * new_col / tl.maximum(row_mean, EPS1)
    update = gradient / tl.sqrt(tl.maximum(estimate, EPS1 * EPS1))
    update_rms = tl.sqrt(tl.sum(tl.sum(update * update, axis=1), axis=0) / (ROWS * COLS))
    scale = tl.maximum(1.0, update_rms / CLIP)
    new_theta = theta * (1.0 - LR * WEIGHT_DECAY) - alpha * update / scale

    tl.store(theta_ptr + offsets, new_theta)
    tl.store(row_ptr + row_ids, new_row)
    tl.store(col_ptr + col_ids, new_col)
    tl.store(step_ptr, step)


def call(inputs):
    required = {"theta", "gradient", "row", "col", "t"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    theta, gradient, row, col, step = (inputs[name] for name in ("theta", "gradient", "row", "col", "t"))
    for name, value, shape in (
        ("theta", theta, (ROWS, COLS)), ("gradient", gradient, (ROWS, COLS)),
        ("row", row, (ROWS, 1)), ("col", col, (1, COLS)),
    ):
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    if bool((row < 0).any()) or bool((col < 0).any()):
        raise ValueError("row and col variance state must be nonnegative")
    if (not torch.is_tensor(step) or step.shape != () or step.dtype != torch.int64
            or not step.is_cuda or not step.is_contiguous() or int(step.item()) != 3):
        raise ValueError("t must be a contiguous CUDA int64 scalar equal to 3")

    _adafactor_kernel[(1,)](
        theta, gradient, row, col, step, ROWS=ROWS, COLS=COLS, LR=LR, EPS1=EPS1,
        EPS2=EPS2, CLIP=CLIP, WEIGHT_DECAY=WEIGHT_DECAY, num_warps=8)
    return {"theta": theta, "row": row, "col": col, "t": step}
