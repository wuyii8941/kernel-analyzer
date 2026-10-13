"""Inputs and specification of the flash attention case (audit_fa.py, per_program.py)."""
from __future__ import annotations


def make_inputs(dtype, Z, H, N, D, seed, scale=1.0):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [(torch.randn(Z, H, N, D, device="cuda", generator=g) * scale).to(dtype) for _ in range(3)]


def spec_attention(q, k, v, sm_scale, causal):
    """The specification: softmax(q k^T * sm_scale) v in float64 (torch)."""
    import torch
    q, k, v = (t.double() for t in (q, k, v))
    s = q @ k.transpose(-1, -2) * sm_scale
    if causal:
        N = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=s.device), 1), float("-inf"))
    return torch.softmax(s, -1) @ v
