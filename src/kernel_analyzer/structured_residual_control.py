"""Controlled transformations of AdamW8bit first-moment residual feedback.

These classes are mechanism probes, not production optimizers.  The second
moment always keeps its ordinary compensation because arbitrary permutation of
an unsigned state residual can create invalid negative second moments.
"""
from __future__ import annotations

import torch

from kernel_analyzer.compensation_control import TensorScalarCompensationControl


def roll_within_blocks(value: torch.Tensor, block_size: int) -> torch.Tensor:
    """Cyclically move coordinates inside each quantization block."""
    flat = value.reshape(-1)
    if flat.numel() % block_size:
        raise ValueError("tensor size must be divisible by block_size")
    return flat.reshape(-1, block_size).roll(1, dims=1).reshape_as(value)


class ExternalFirstMomentResidualControl(TensorScalarCompensationControl):
    """Read a frozen first-moment residual sequence in a declared arrangement."""

    def __init__(self, *args, first_residual_reads, arrangement: str, **kwargs):
        if arrangement not in {"CORRECT", "COORDINATE_ROLL", "TIME_REVERSE"}:
            raise ValueError("unknown first-moment residual arrangement")
        super().__init__(*args, compensation_enabled=True, **kwargs)
        self.first_residual_reads = [value.detach().clone() for value in first_residual_reads]
        if arrangement == "TIME_REVERSE":
            # The initial all-zero read stays first.  The remaining observed
            # residuals are reversed, preserving their multiset and energy.
            self.first_residual_reads = [
                self.first_residual_reads[0], *reversed(self.first_residual_reads[1:])
            ]
        self.arrangement = arrangement
        self._first_read_index = 0

    def _read(self, state, compensation):
        from torchao.optim.subclass_8bit import OptimState8bit

        if isinstance(state, OptimState8bit) and state.signed:
            if self._first_read_index >= len(self.first_residual_reads):
                raise RuntimeError("first-moment residual schedule exhausted")
            residual = self.first_residual_reads[self._first_read_index].to(state.device)
            self._first_read_index += 1
            if self.arrangement == "COORDINATE_ROLL":
                residual = roll_within_blocks(residual, state.block_size)
            return state.dequantize() + residual.float()
        return super()._read(state, compensation)

