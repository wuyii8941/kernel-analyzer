"""Direct Triton 3.6 float32 native dropout backward with an explicit Boolean mask."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 1024, 256
SCALE = 1.25


@triton.jit
def _dropout_backward_kernel(grad_ptr, mask_ptr, out_ptr,
                             SIZE: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
    keep = tl.load(mask_ptr + offsets, mask=active, other=False)
    value = tl.where(keep, grad * SCALE, 0.0)
    tl.store(out_ptr + offsets, value, mask=active)


def call(inputs):
    if set(inputs) != {"grad_output", "mask"}:
        raise ValueError(f"expected grad_output and mask, received {sorted(inputs)}")
    grad, mask = inputs["grad_output"], inputs["mask"]
    if (not torch.is_tensor(grad) or grad.shape != (SIZE,) or grad.dtype != torch.float32
            or not grad.is_cuda or not grad.is_contiguous()):
        raise ValueError(f"grad_output must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
    if not bool(torch.isfinite(grad).all()) or bool((grad.abs() > 16).any()):
        raise ValueError("grad_output must contain finite values in [-16, 16]")
    if (not torch.is_tensor(mask) or mask.shape != (SIZE,) or mask.dtype != torch.bool
            or not mask.is_cuda or not mask.is_contiguous()):
        raise ValueError(f"mask must be a contiguous CUDA bool tensor of shape {(SIZE,)}")

    out = torch.empty_like(grad)
    _dropout_backward_kernel[(triton.cdiv(SIZE, BLOCK),)](
        grad, mask, out, SIZE=SIZE, SCALE=SCALE, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
