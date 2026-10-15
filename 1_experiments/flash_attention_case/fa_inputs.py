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


def reference_o(q, k, v, sm_scale, causal):
    """The reference program for O: the torch operation the kernel replaces, in float64."""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=causal, scale=sm_scale)


def reference_m(q, k, sm_scale, causal):
    """The reference program for M: the logsumexp of the scaled scores in base 2 (the tutorial's kernel stores
    m_i + log2(l_i) with the scores in log2 units), in float64."""
    import math

    import torch
    s = q.double() @ k.double().transpose(-1, -2) * sm_scale
    if causal:
        N = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=s.device), 1), float("-inf"))
    return torch.logsumexp(s, -1) / math.log(2)
