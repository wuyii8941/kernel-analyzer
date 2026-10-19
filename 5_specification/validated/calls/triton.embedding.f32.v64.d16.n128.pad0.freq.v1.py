"""Direct Triton 3.6 float32 embedding forward and deterministic dense weight VJP."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS, WIDTH, POSITIONS, PADDING_IDX = 64, 16, 128, 0


def make_balanced_indices(seed, shape, dtype):
    if tuple(shape) != (POSITIONS,) or dtype != "int64":
        raise ValueError("balanced index generator requires shape [128] and dtype int64")
    position = torch.arange(POSITIONS, dtype=torch.int64, device="cuda")
    return (position * 29 + int(seed)) % ROWS


@triton.jit
def _embedding_forward(weight_ptr, index_ptr, out_ptr, WIDTH: tl.constexpr):
    position = tl.program_id(0)
    feature = tl.arange(0, WIDTH)
    row = tl.load(index_ptr + position)
    values = tl.load(weight_ptr + row * WIDTH + feature)
    tl.store(out_ptr + position * WIDTH + feature, values)


@triton.jit
def _embedding_dense_vjp(index_ptr, upstream_ptr, dweight_ptr,
                         WIDTH: tl.constexpr, POSITIONS: tl.constexpr, PADDING_IDX: tl.constexpr):
    row = tl.program_id(0)
    position = tl.arange(0, POSITIONS)
    feature = tl.arange(0, WIDTH)
    indices = tl.load(index_ptr + position)
    selected = indices == row
    received = tl.load(upstream_ptr + position[:, None] * WIDTH + feature[None, :])
    total = tl.sum(tl.where(selected[:, None], received, 0.0), axis=0)
    count = tl.sum(selected.to(tl.float32), axis=0)
    denominator = tl.where(count == 0, 1.0, count)
    result = total / denominator
    result = tl.where(row == PADDING_IDX, 0.0, result)
    tl.store(dweight_ptr + row * WIDTH + feature, result)


def call(inputs):
    required = {"weight", "indices", "upstream"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    weight, indices, upstream = (inputs[name] for name in ("weight", "indices", "upstream"))
    expected = {"weight": (ROWS, WIDTH), "indices": (POSITIONS,), "upstream": (POSITIONS, WIDTH)}
    for name, tensor in (("weight", weight), ("indices", indices), ("upstream", upstream)):
        dtype = torch.int64 if name == "indices" else torch.float32
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != expected[name] or tensor.dtype != dtype
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA {dtype} tensor of shape {expected[name]}")
    if bool(((indices < 0) | (indices >= ROWS)).any()):
        raise ValueError("indices must lie in [0, ROWS)")
    for name, tensor in (("weight", weight), ("upstream", upstream)):
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")

    out = torch.empty((POSITIONS, WIDTH), dtype=torch.float32, device=weight.device)
    dweight = torch.empty((ROWS, WIDTH), dtype=torch.float32, device=weight.device)
    _embedding_forward[(POSITIONS,)](weight, indices, out, WIDTH=WIDTH, num_warps=1)
    _embedding_dense_vjp[(ROWS,)](
        indices, upstream, dweight, WIDTH=WIDTH, POSITIONS=POSITIONS,
        PADDING_IDX=PADDING_IDX, num_warps=8,
    )
    return {"out": out, "dweight": dweight}
