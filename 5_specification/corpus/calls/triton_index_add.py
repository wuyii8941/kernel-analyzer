"""Direct Triton 3.6 float32 out-of-place index_add for one fixed reviewed index family."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, COLS, SOURCE_ROWS = 128, 32, 64
SIZE, BLOCK = ROWS * COLS, 256
ALPHA = 0.5


def make_even_index(seed, shape, dtype):
    if tuple(shape) != (SOURCE_ROWS,) or dtype != "int64":
        raise ValueError("index generator requires shape [64] and dtype int64")
    return torch.arange(SOURCE_ROWS, dtype=torch.int64, device="cuda") * 2


@triton.jit
def _index_add_kernel(self_ptr, source_ptr, out_ptr,
                      SIZE: tl.constexpr, COLS: tl.constexpr,
                      ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SIZE
    row = offsets // COLS
    col = offsets % COLS
    base = tl.load(self_ptr + offsets, mask=active, other=0.0)
    selected = row % 2 == 0
    source_offset = (row // 2) * COLS + col
    source = tl.load(source_ptr + source_offset, mask=active & selected, other=0.0)
    tl.store(out_ptr + offsets, base + source * ALPHA, mask=active)


def call(inputs):
    if set(inputs) != {"self", "source", "index"}:
        raise ValueError(f"expected self, source and index, received {sorted(inputs)}")
    self_tensor, source, index = inputs["self"], inputs["source"], inputs["index"]
    tensors = (("self", self_tensor, (ROWS, COLS)), ("source", source, (SOURCE_ROWS, COLS)))
    for name, value, shape in tensors:
        if (not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 16).any()):
            raise ValueError(f"{name} must contain finite values in [-16, 16]")
    if (not torch.is_tensor(index) or index.shape != (SOURCE_ROWS,) or index.dtype != torch.int64
            or not index.is_cuda or not index.is_contiguous()):
        raise ValueError("index must be a contiguous CUDA int64 tensor of shape [64]")
    expected_index = torch.arange(SOURCE_ROWS, dtype=torch.int64, device=index.device) * 2
    if not bool(torch.equal(index, expected_index)):
        raise ValueError("index must contain the fixed even rows 0, 2, ..., 126")

    out = torch.empty_like(self_tensor)
    _index_add_kernel[(triton.cdiv(SIZE, BLOCK),)](
        self_tensor, source, out, SIZE=SIZE, COLS=COLS, ALPHA=ALPHA, BLOCK=BLOCK, num_warps=4)
    return {"out": out}
