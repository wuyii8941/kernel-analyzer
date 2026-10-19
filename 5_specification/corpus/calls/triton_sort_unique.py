"""Direct Triton 3.6 ascending row sort with complete original indices."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLS = 32, 32


@triton.jit
def _sort_unique_kernel(x_ptr, values_ptr, indices_ptr, COLS: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, COLS)
    source = tl.load(x_ptr + row * COLS + columns)
    ordered = tl.sort(source, dim=0, descending=False)
    matches = ordered[:, None] == source[None, :]
    original = tl.argmax(matches.to(tl.int32), axis=1)
    tl.store(values_ptr + row * COLS + columns, ordered)
    tl.store(indices_ptr + row * COLS + columns, original)


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (ROWS, COLS) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(ROWS, COLS)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    ordered = torch.sort(x, dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("every input row must contain pairwise distinct values")
    values = torch.empty_like(x)
    indices = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    _sort_unique_kernel[(ROWS,)](x, values, indices, COLS=COLS, num_warps=4)
    return {"values": values, "indices": indices}
