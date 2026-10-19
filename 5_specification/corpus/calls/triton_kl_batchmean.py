"""Direct Triton 3.6 PyTorch-style probability-target KL batchmean and input VJP."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BATCH, SUPPORT, SIZE = 8, 64, 8 * 64


@triton.jit
def _kl_batchmean_kernel(input_ptr, target_ptr, loss_ptr, grad_ptr,
                         SIZE: tl.constexpr, BATCH: tl.constexpr,
                         BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < SIZE
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=1.0)
    terms = target * (tl.log(target) - values)
    total = tl.sum(tl.where(mask, terms, 0.0), axis=0) / BATCH
    grad = -target / BATCH
    tl.store(loss_ptr, total)
    tl.store(grad_ptr + offsets, grad, mask=mask)


def call(inputs):
    if set(inputs) != {"input", "target"}:
        raise ValueError(f"expected input and target, received {sorted(inputs)}")
    values, target = inputs["input"], inputs["target"]
    for name, value in (("input", values), ("target", target)):
        if (not torch.is_tensor(value) or value.shape != (BATCH, SUPPORT)
                or value.dtype != torch.float32 or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(
                f"{name} must be a contiguous CUDA float32 tensor of shape {(BATCH, SUPPORT)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    if bool((values.abs() > 16).any()):
        raise ValueError("input must lie in the reviewed measurement range [-16, 16]")
    if bool((target <= 0).any()) or bool((target > 4).any()):
        raise ValueError("target must lie in the reviewed measurement range (0, 4]")

    loss = torch.empty((), dtype=torch.float32, device=values.device)
    grad = torch.empty_like(values)
    _kl_batchmean_kernel[(1,)](
        values, target, loss, grad, SIZE=SIZE, BATCH=BATCH, BLOCK=SIZE, num_warps=8)
    return {"loss": loss, "grad": grad}
