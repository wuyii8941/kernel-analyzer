"""Direct Triton 3.6 float32 matrix transpose for one reviewed LAYOUT binding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLS, BLOCK = 32, 64, 256
SIZE = ROWS * COLS


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, SIZE: tl.constexpr,
                      ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    out_row = offsets // ROWS
    out_col = offsets % ROWS
    source = out_col * COLS + out_row
    values = tl.load(x_ptr + source, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (ROWS, COLS) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(ROWS, COLS)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    out = torch.empty((COLS, ROWS), dtype=x.dtype, device=x.device)
    _transpose_kernel[(triton.cdiv(SIZE, BLOCK),)](
        x, out, SIZE=SIZE, ROWS=ROWS, COLS=COLS, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
