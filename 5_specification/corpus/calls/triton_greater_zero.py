"""Direct Triton 3.6 float32 comparison for one reviewed LOGIC binding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 1024, 256


@triton.jit
def _greater_zero_kernel(x_ptr, out_ptr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values > 0.0, mask=mask)


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (SIZE,) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    out = torch.empty((SIZE,), dtype=torch.bool, device=x.device)
    _greater_zero_kernel[(triton.cdiv(SIZE, BLOCK),)](
        x, out, SIZE=SIZE, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
