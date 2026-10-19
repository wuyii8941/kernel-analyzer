"""Direct Triton 3.6 row sum for one reviewed REDUCE binding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLUMNS = 32, 256


@triton.jit
def _row_sum_kernel(x_ptr, out_ptr,
                    ROWS: tl.constexpr, COLUMNS: tl.constexpr,
                    BLOCK_COLUMNS: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_COLUMNS)
    values = tl.load(x_ptr + row * COLUMNS + columns,
                     mask=columns < COLUMNS, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr + row, total, mask=row < ROWS)


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (ROWS, COLUMNS)
            or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(ROWS, COLUMNS)}")
    out = torch.empty((ROWS,), dtype=torch.float32, device=x.device)
    _row_sum_kernel[(ROWS,)](
        x, out, ROWS=ROWS, COLUMNS=COLUMNS, BLOCK_COLUMNS=COLUMNS, num_warps=8)
    return {"out": out}
