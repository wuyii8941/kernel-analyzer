"""Registry adapter for reviewed Ministral attention position scaling."""

from __future__ import annotations

import torch

from .attention_position_scaling_source import check_source
from .dense_pointer_view import dense_pointer_view


def _fused_rotary_reference(metadata, candidate, contract):
    pointers = metadata.get("runtime_pointers") or {}
    expected = {"in_ptr0", "in_ptr1", "in_ptr2", "out_ptr0", "out_ptr1"}
    if set(pointers) != expected:
        raise ValueError("Unexpected fused position-scaling pointer set")
    query, frequencies, positions = (
        pointers["in_ptr0"], pointers["in_ptr1"], pointers["in_ptr2"])
    for name, value, dtype, size in (
        ("in_ptr0", query, torch.bfloat16, 32 * 128 * 128),
        ("in_ptr1", frequencies, torch.float32, 64),
        ("in_ptr2", positions, torch.int64, 128),
    ):
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.device != candidate.device or value.numel() != size):
            raise ValueError("Fused position-scaling pointer differs: " + name)
    for name in ("out_ptr0", "out_ptr1"):
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16
                or value.device != candidate.device or value.numel() != contract["elements"]):
            raise ValueError("Fused position-scaling pointer differs: " + name)
    if not torch.isfinite(query).all() or not torch.isfinite(frequencies).all():
        raise ValueError("Nonfinite fused position-scaling input")

    # in_ptr0 has logical strides (128, 4096, 1) for [head, position, dim]
    # while each output is contiguous [head, position, dim].  Reconstruct the
    # source loads from pointer order rather than trusting the Python view.
    query_pointer = dense_pointer_view(query, (128, 32, 128))
    query_hpd = query_pointer.permute(1, 0, 2).float().clone()
    frequency = dense_pointer_view(frequencies, (64,)).float().clone()
    position = dense_pointer_view(positions, (128,)).float().clone()
    phase = position[:, None] * frequency[None, :]
    cosine = torch.cat((torch.cos(phase), torch.cos(phase)), dim=-1)
    sine = torch.cat((torch.sin(phase), torch.sin(phase)), dim=-1)
    rotated = torch.cat((-query_hpd[..., 64:], query_hpd[..., :64]), dim=-1)
    scale = 1.0 + float(contract["beta"]) * torch.log(
        1.0 + torch.floor(position / float(contract["original_context_length"])))
    result = (query_hpd * cosine[None, :, :] + rotated * sine[None, :, :])
    result = result * scale[None, :, None]
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite fused position-scaling reference arithmetic")
    return result.to(torch.bfloat16).reshape(candidate.shape)


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Position-scaling symbol differs")
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("Position-scaling output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported position-scaling input/output alias")
    if (not isinstance(candidate, torch.Tensor)
            or candidate.dtype != torch.bfloat16
            or candidate.numel() != contract.get("elements")
            or not candidate.is_contiguous()):
        raise ValueError("Position-scaling runtime layout differs")
    if contract.get("layout") == "FUSED_ROTARY_QUERY_SCALING":
        return _fused_rotary_reference(metadata, candidate, contract)
    positions = torch.arange(
        candidate.numel(), device=candidate.device, dtype=torch.float32,
    )
    result = 1.0 + float(contract["beta"]) * torch.log(
        1.0 + torch.floor(positions / float(contract["original_context_length"]))
    )
    return result.to(candidate.dtype).reshape(candidate.shape)
