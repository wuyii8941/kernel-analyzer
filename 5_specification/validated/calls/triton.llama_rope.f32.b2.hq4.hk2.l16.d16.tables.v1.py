"""Direct Triton 3.6 float32 Transformers-Llama supplied-table RoPE and q/k VJP."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BATCH, QUERY_HEADS, KEY_HEADS, LENGTH, WIDTH = 2, 4, 2, 16, 16


@triton.jit
def _rope_value_vjp(x_ptr, upstream_ptr, cos_ptr, sin_ptr, out_ptr, grad_ptr,
                    HEADS: tl.constexpr, LENGTH: tl.constexpr, WIDTH: tl.constexpr):
    program = tl.program_id(0)
    token = program % LENGTH
    head = (program // LENGTH) % HEADS
    batch = program // (LENGTH * HEADS)
    feature = tl.arange(0, WIDTH)
    half = WIDTH // 2
    partner = tl.where(feature < half, feature + half, feature - half)
    base = ((batch * HEADS + head) * LENGTH + token) * WIDTH
    table = (batch * LENGTH + token) * WIDTH
    x = tl.load(x_ptr + base + feature)
    x_partner = tl.load(x_ptr + base + partner)
    upstream = tl.load(upstream_ptr + base + feature)
    upstream_partner = tl.load(upstream_ptr + base + partner)
    cos = tl.load(cos_ptr + table + feature)
    sin = tl.load(sin_ptr + table + feature)
    sin_partner = tl.load(sin_ptr + table + partner)
    rotated = tl.where(feature < half, -x_partner, x_partner)
    out = x * cos + rotated * sin
    grad = upstream * cos + tl.where(feature < half, upstream_partner * sin_partner,
                                     -upstream_partner * sin_partner)
    tl.store(out_ptr + base + feature, out)
    tl.store(grad_ptr + base + feature, grad)


def call(inputs):
    required = {"q", "k", "cos", "sin", "upstream_q", "upstream_k"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    q, k, cos, sin, upstream_q, upstream_k = (inputs[name] for name in
                                               ("q", "k", "cos", "sin", "upstream_q", "upstream_k"))
    expected = {
        "q": (BATCH, QUERY_HEADS, LENGTH, WIDTH),
        "k": (BATCH, KEY_HEADS, LENGTH, WIDTH),
        "cos": (BATCH, LENGTH, WIDTH), "sin": (BATCH, LENGTH, WIDTH),
        "upstream_q": (BATCH, QUERY_HEADS, LENGTH, WIDTH),
        "upstream_k": (BATCH, KEY_HEADS, LENGTH, WIDTH),
    }
    for name, tensor in (("q", q), ("k", k), ("cos", cos), ("sin", sin),
                         ("upstream_q", upstream_q), ("upstream_k", upstream_k)):
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != expected[name] or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {expected[name]}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")
    q_out, dq = torch.empty_like(q), torch.empty_like(q)
    k_out, dk = torch.empty_like(k), torch.empty_like(k)
    _rope_value_vjp[(BATCH * QUERY_HEADS * LENGTH,)](
        q, upstream_q, cos, sin, q_out, dq, HEADS=QUERY_HEADS, LENGTH=LENGTH, WIDTH=WIDTH, num_warps=1)
    _rope_value_vjp[(BATCH * KEY_HEADS * LENGTH,)](
        k, upstream_k, cos, sin, k_out, dk, HEADS=KEY_HEADS, LENGTH=LENGTH, WIDTH=WIDTH, num_warps=1)
    return {"q_out": q_out, "k_out": k_out, "dq": dq, "dk": dk}
