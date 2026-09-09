"""Registry adapter for the reviewed in-place BF16 channel-bias addition."""

import torch

from kernel_analyzer.channel_bias_source import check_source as _check_source


def check_source(source, symbol):
    contract = _check_source(source, symbol)
    return dict(contract, output_pointer="in_out_ptr0")


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Channel-bias symbol differs")
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("Channel-bias output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported channel-bias input/output alias")
    pointers = metadata.get("runtime_pointers") or {}
    if set(pointers) != {"in_out_ptr0", "in_ptr0"}:
        raise ValueError("Channel-bias pointer ABI differs")
    channels, length = contract["channels"], contract["length"]
    base, bias = pointers["in_out_ptr0"], pointers["in_ptr0"]
    if (not isinstance(candidate, torch.Tensor)
            or any(not isinstance(value, torch.Tensor) for value in (base, bias))
            or candidate.dtype != torch.bfloat16 or base.dtype != torch.bfloat16
            or bias.dtype != torch.bfloat16
            or candidate.numel() != channels * length
            or base.numel() != channels * length or bias.numel() != channels
            or not candidate.is_contiguous() or not base.is_contiguous()
            or not bias.is_contiguous()
            or any(value.device != candidate.device for value in (base, bias))
            or not torch.isfinite(base).all() or not torch.isfinite(bias).all()):
        raise ValueError("Channel-bias runtime layout differs")
    result = base.reshape(channels, length).float() + bias.reshape(channels, 1).float()
    return result.to(torch.bfloat16).reshape(candidate.shape)
