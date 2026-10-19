"""PyTorch 2.10 Inductor functional gather along dimension one."""
from __future__ import annotations

import torch


ROWS, COLS, SELECTED, DIM = 32, 64, 24, 1
SPARSE_GRAD = False


def _gather(x, index):
    return torch.gather(x, DIM, index, sparse_grad=SPARSE_GRAD)


_compiled = torch.compile(_gather, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "index", "dim", "sparse_grad"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["dim"] != DIM or inputs["sparse_grad"] is not SPARSE_GRAD:
        raise ValueError("this call fixes dim=1 and sparse_grad=False")
    x, index = inputs["x"], inputs["index"]
    if (not torch.is_tensor(x) or tuple(x.shape) != (ROWS, COLS)
            or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be contiguous CUDA float32 with shape {(ROWS, COLS)}")
    if (not torch.is_tensor(index) or tuple(index.shape) != (ROWS, SELECTED)
            or index.dtype != torch.int64 or not index.is_cuda or not index.is_contiguous()):
        raise ValueError(
            f"index must be contiguous CUDA int64 with shape {(ROWS, SELECTED)}"
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain finite values")
    if bool((index < 0).any()) or bool((index >= COLS).any()):
        raise ValueError(f"index values must be in [0, {COLS})")
    return {"out": _compiled(x, index)}
