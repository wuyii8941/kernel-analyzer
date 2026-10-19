"""Direct Triton 3.6 float32 in-place gradient value clipping for one buffer."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 4096, 256
CLIP_VALUE = 0.75


@triton.jit
def _clip_grad_value_kernel(grad_ptr, SIZE: tl.constexpr, CLIP_VALUE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
    clipped = tl.minimum(tl.maximum(grad, -CLIP_VALUE), CLIP_VALUE)
    tl.store(grad_ptr + offsets, clipped, mask=active)


def call(inputs):
    if set(inputs) != {"grad"}:
        raise ValueError(f"expected grad, received {sorted(inputs)}")
    grad = inputs["grad"]
    if (not torch.is_tensor(grad) or grad.shape != (SIZE,) or grad.dtype != torch.float32
            or not grad.is_cuda or not grad.is_contiguous()):
        raise ValueError(f"grad must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
    if not bool(torch.isfinite(grad).all()) or bool((grad.abs() > 16).any()):
        raise ValueError("grad must contain finite values in [-16, 16]")

    _clip_grad_value_kernel[(triton.cdiv(SIZE, BLOCK),)](
        grad, SIZE=SIZE, CLIP_VALUE=CLIP_VALUE, BLOCK=BLOCK, num_warps=4)
    return {"out": grad}
