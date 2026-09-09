"""Registry adapter for the reviewed forward state-recurrence final output."""

import torch

from kernel_analyzer.forward_state_recurrence_reference import decode_inputs, evaluate
from kernel_analyzer.forward_state_recurrence_source import check_source as _check_source


def check_source(source, symbol):
    contract = _check_source(source, symbol)
    outputs = [f"out_ptr{index}" for index in range(contract["steps"])]
    return dict(contract, output_pointers=outputs, output_pointer=outputs[-1])


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Forward recurrence symbol differs")
    formal = metadata.get("formal_pointer")
    if formal != contract.get("output_pointer"):
        raise ValueError("Forward recurrence final-output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported forward recurrence input/output alias")
    pointers = metadata.get("runtime_pointers") or {}
    selected = {name: pointers.get(name) for name in (f"in_ptr{i}" for i in range(5))}
    decoded = decode_inputs(
        selected,
        steps=contract["steps"],
        channels=contract["channels"],
        state_width=contract["state_width"],
        packed_width=contract["packed_width"],
        state_offset=contract["state_offset"],
    )
    final = evaluate(**decoded)["states"][-1].to(torch.bfloat16)
    if (not isinstance(candidate, torch.Tensor) or candidate.dtype != torch.bfloat16
            or candidate.numel() != final.numel() or not candidate.is_contiguous()
            or candidate.device != final.device):
        raise ValueError("Forward recurrence final-output layout differs")
    return final.reshape(candidate.shape)
