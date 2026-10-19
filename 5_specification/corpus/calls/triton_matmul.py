"""Direct Triton 3.6.0 matrix product proposed for MATMUL binding review."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


M, K, N = 16, 32, 16


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, out_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + rows[:, None] * K + inner[None, :],
                mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
    b = tl.load(b_ptr + inner[:, None] * N + cols[None, :],
                mask=(inner[:, None] < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, input_precision="ieee")
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def call(inputs):
    if set(inputs) != {"a", "b"}:
        raise ValueError(f"expected inputs a and b, received {sorted(inputs)}")
    a, b = inputs["a"], inputs["b"]
    for name, value, shape in (("a", a, (M, K)), ("b", b, (K, N))):
        if not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32 or not value.is_cuda:
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    out = torch.empty((M, N), dtype=torch.float32, device=a.device)
    _matmul_kernel[(1,)](a, b, out, M, N, K, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, num_warps=4)
    return {"out": out}
