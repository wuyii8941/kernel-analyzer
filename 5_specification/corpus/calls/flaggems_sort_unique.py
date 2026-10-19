"""FlagGems 5.0.2 public ascending row sort in the reviewed unique-row range."""
from __future__ import annotations

import numpy as np
import torch

import flag_gems


ROWS, COLUMNS = 32, 32


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (ROWS, COLUMNS)
            or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(
            f"x must be a contiguous CUDA float32 tensor of shape {(ROWS, COLUMNS)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    host = x.detach().cpu().numpy()
    if any(np.unique(row).size != COLUMNS for row in host):
        raise ValueError("every row must contain pairwise distinct physical float32 values")

    with flag_gems.use_gems(include=["sort"]):
        values, indices = torch.sort(x, dim=1, descending=False)
    return {"values": values, "indices": indices}
