"""Direct Triton 3.6 float32 BSR softmax values core for a fixed canonical pattern."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


MATRIX_ROWS, MATRIX_COLUMNS = 128, 64
BLOCK_ROWS, BLOCK_COLUMNS = 16, 16
BLOCK_ROW_COUNT, BLOCKS_PER_ROW = 8, 2
NNZ_BLOCKS = BLOCK_ROW_COUNT * BLOCKS_PER_ROW
CROW = [2 * row for row in range(BLOCK_ROW_COUNT + 1)]
COL = [column for row in range(BLOCK_ROW_COUNT) for column in ((0, 2) if row % 2 == 0 else (1, 3))]


@triton.jit
def _bsr_softmax_values(values_ptr, out_ptr,
                        BLOCK_ROWS: tl.constexpr, BLOCK_COLUMNS: tl.constexpr,
                        BLOCKS_PER_ROW: tl.constexpr):
    matrix_row = tl.program_id(0)
    block_row = matrix_row // BLOCK_ROWS
    row_in_block = matrix_row % BLOCK_ROWS
    position = tl.arange(0, BLOCKS_PER_ROW * BLOCK_COLUMNS)
    entry = block_row * BLOCKS_PER_ROW + position // BLOCK_COLUMNS
    column = position % BLOCK_COLUMNS
    offsets = entry * BLOCK_ROWS * BLOCK_COLUMNS + row_in_block * BLOCK_COLUMNS + column
    values = tl.load(values_ptr + offsets)
    maximum = tl.max(values, axis=0)
    numerators = tl.exp(values - maximum)
    result = numerators / tl.sum(numerators, axis=0)
    tl.store(out_ptr + offsets, result)


def call(inputs):
    if set(inputs) != {"values"}:
        raise ValueError(f"expected ['values'], received {sorted(inputs)}")
    values = inputs["values"]
    expected = (NNZ_BLOCKS, BLOCK_ROWS, BLOCK_COLUMNS)
    if (not torch.is_tensor(values) or tuple(values.shape) != expected or values.dtype != torch.float32
            or not values.is_cuda or not values.is_contiguous()):
        raise ValueError(f"values must be a contiguous CUDA float32 tensor of shape {expected}")
    if not bool(torch.isfinite(values).all()) or bool((values.abs() > 8).any()):
        raise ValueError("values must contain finite values in [-8, 8]")
    out = torch.empty_like(values)
    _bsr_softmax_values[(MATRIX_ROWS,)](
        values, out, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLUMNS=BLOCK_COLUMNS,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW, num_warps=1,
    )
    return {"out": out}
