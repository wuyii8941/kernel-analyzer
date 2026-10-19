"""Direct Triton 3.6 float32 row selection for one reviewed INDEX binding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLS, SELECTED, BLOCK = 128, 32, 64, 256
SIZE = SELECTED * COLS


@triton.jit
def _index_select_kernel(x_ptr, index_ptr, out_ptr, SIZE: tl.constexpr,
                         COLS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    output_row = offsets // COLS
    column = offsets % COLS
    source_row = tl.load(index_ptr + output_row, mask=mask, other=0)
    values = tl.load(x_ptr + source_row * COLS + column, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


def call(inputs):
    if set(inputs) != {"x", "index"}:
        raise ValueError(f"expected inputs x and index, received {sorted(inputs)}")
    x, index = inputs["x"], inputs["index"]
    if (not torch.is_tensor(x) or x.shape != (ROWS, COLS) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(ROWS, COLS)}")
    if (not torch.is_tensor(index) or index.shape != (SELECTED,) or index.dtype != torch.int64
            or not index.is_cuda or not index.is_contiguous()):
        raise ValueError(f"index must be a contiguous CUDA int64 tensor of shape {(SELECTED,)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    if bool((index < 0).any()) or bool((index >= ROWS).any()):
        raise ValueError(f"index values must be in [0, {ROWS})")
    out = torch.empty((SELECTED, COLS), dtype=x.dtype, device=x.device)
    _index_select_kernel[(triton.cdiv(SIZE, BLOCK),)](
        x, index, out, SIZE=SIZE, COLS=COLS, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
