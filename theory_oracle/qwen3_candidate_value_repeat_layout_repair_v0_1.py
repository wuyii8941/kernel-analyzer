#!/usr/bin/env python
"""Repair value-head repeat/layout/cast generated-kernel representatives."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    if destination.ndim != 4 or destination.shape[1] % 2 or destination.shape[-1] != 128:
        raise RuntimeError("unexpected value-repeat destination layout")
    batch, repeated_heads, sequence, width = destination.shape
    heads = repeated_heads // 2
    if source.numel() != batch * sequence * heads * width:
        raise RuntimeError("value-repeat source numel mismatch")
    with torch.no_grad():
        logical = source.reshape(batch, sequence, heads, width).permute(0, 2, 1, 3)
        repeated = logical.unsqueeze(2).expand(batch, heads, 2, sequence, width).clone().reshape(destination.shape)
        destination.copy_(repeated.to(destination.dtype))


if __name__ == "__main__":
    run_experiment({"description": __doc__, "schema_version": "forkcert.qwen3-candidate-value-repeat-layout-repair.v0.1", "valid_status": "VALID_ORIGINAL_CANDIDATE_VALUE_REPEAT_LAYOUT_REPAIR", "kernel_family": "triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_view_10", "expected_family_calls": 28, "selected_call_indices": [0, 14, 27], "claim_limits": ["value repeat/layout/cast generated family only", "one state", "repair only", "no correctness claim"]}, repair)
