"""Direct Triton 3.6 element-wise multiply-add for one reviewed ARITH binding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SIZE, BLOCK = 1024, 256


@triton.jit
def _multiply_add_kernel(x_ptr, y_ptr, z_ptr, out_ptr,
                         SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * y + z, mask=mask)


def call(inputs):
    if set(inputs) != {"x", "y", "z"}:
        raise ValueError(f"expected inputs x, y and z, received {sorted(inputs)}")
    values = [inputs[name] for name in ("x", "y", "z")]
    for name, value in zip(("x", "y", "z"), values):
        if (not torch.is_tensor(value) or value.shape != (SIZE,) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {(SIZE,)}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    out = torch.empty_like(values[0])
    _multiply_add_kernel[(triton.cdiv(SIZE, BLOCK),)](
        values[0], values[1], values[2], out, SIZE=SIZE, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
