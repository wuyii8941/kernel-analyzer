"""Focused Triton variants for the Ministral fused RoPE arithmetic.

This module changes only the implementation used for sine and cosine.  It is
an experimental source intervention, not a general RoPE implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tl_math_kernel(query, frequency, position, output, n: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    dim = index % 128
    token = (index // 128) % 128
    base = index - dim
    value = tl.load(query + index, mask=mask, other=0.0).to(tl.float32)
    partner_dim = tl.where(dim < 64, dim + 64, dim - 64)
    partner = tl.load(query + base + partner_dim, mask=mask, other=0.0).to(tl.float32)
    rotated = tl.where(dim < 64, -partner, partner)
    phase = tl.load(frequency + (dim % 64)).to(tl.float32) * tl.load(position + token).to(tl.float32)
    cosine = tl.cos(phase)
    sine = tl.sin(phase)
    scale = 1.0 + 0.1 * tl.log(1.0 + tl.floor(tl.load(position + token).to(tl.float32) / 16384.0))
    tl.store(output + index, (value * cosine + rotated * sine) * scale, mask=mask)

@triton.jit
def _libdevice_kernel(query, frequency, position, output, n: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < n
    dim = index % 128
    token = (index // 128) % 128
    base = index - dim
    value = tl.load(query + index, mask=mask, other=0.0).to(tl.float32)
    partner_dim = tl.where(dim < 64, dim + 64, dim - 64)
    partner = tl.load(query + base + partner_dim, mask=mask, other=0.0).to(tl.float32)
    rotated = tl.where(dim < 64, -partner, partner)
    phase = tl.load(frequency + (dim % 64)).to(tl.float32) * tl.load(position + token).to(tl.float32)
    cosine = libdevice.cos(phase)
    sine = libdevice.sin(phase)
    scale = 1.0 + 0.1 * tl.log(1.0 + tl.floor(tl.load(position + token).to(tl.float32) / 16384.0))
    tl.store(output + index, (value * cosine + rotated * sine) * scale, mask=mask)

def rotary_variant(
    query: torch.Tensor,
    frequency: torch.Tensor,
    position: torch.Tensor,
    *,
    trig: str,
) -> torch.Tensor:
    """Run one same-storage RoPE implementation on contiguous [32,128,128]."""
    if trig not in {"tl_math", "libdevice"}:
        raise ValueError("trig must be 'tl_math' or 'libdevice'")
    if (
        query.shape != (32, 128, 128)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or frequency.shape != (64,)
        or frequency.dtype != torch.float32
        or position.shape != (128,)
        or position.dtype != torch.int64
        or query.device != frequency.device
        or query.device != position.device
        or query.device.type != "cuda"
    ):
        raise ValueError("Ministral RoPE variant inputs differ from the declared boundary")
    output = torch.empty_like(query)
    kernel = _tl_math_kernel if trig == "tl_math" else _libdevice_kernel
    block = 256
    kernel[(triton.cdiv(query.numel(), block),)](
        query,
        frequency,
        position,
        output,
        n=query.numel(),
        block=block,
    )
    return output


def fp64_reference(
    query: torch.Tensor, frequency: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    """Higher-precision reference with the same formula and BF16 output."""
    query64 = query.double()
    phase = position.double()[:, None] * frequency.double()[None, :]
    cosine = torch.cat((torch.cos(phase), torch.cos(phase)), dim=-1)
    sine = torch.cat((torch.sin(phase), torch.sin(phase)), dim=-1)
    rotated = torch.cat((-query64[..., 64:], query64[..., :64]), dim=-1)
    scale = 1.0 + 0.1 * torch.log(1.0 + torch.floor(position.double() / 16384.0))
    return ((query64 * cosine[None] + rotated * sine[None]) * scale[None, :, None]).to(
        torch.bfloat16
    )
