"""Reviewed mixed-precision optimizer-state variants for mechanism experiments."""

from __future__ import annotations

import torch
from torchao.optim import AdamW8bit


class AdamWFirstMomentFP32(AdamW8bit):
    """Keep the first moment in FP32 and the second moment blockwise 8-bit."""

    def _new_buffer(self, parameter: torch.Tensor, signed: bool):
        if signed:
            return torch.zeros_like(parameter)
        return super()._new_buffer(parameter, signed)


class AdamWSecondMomentFP32(AdamW8bit):
    """Keep the second moment in FP32 and the first moment blockwise 8-bit."""

    def _new_buffer(self, parameter: torch.Tensor, signed: bool):
        if not signed:
            return torch.zeros_like(parameter)
        return super()._new_buffer(parameter, signed)
