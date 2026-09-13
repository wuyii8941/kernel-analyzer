"""Online controls for first-moment residual coordinate and temporal structure."""
from __future__ import annotations

import torch

from kernel_analyzer.compensation_control import TensorScalarCompensationControl
from kernel_analyzer.structured_residual_control import roll_within_blocks


class OnlineFirstMomentResidualControl(TensorScalarCompensationControl):
    """Apply a deployable coordinate roll or one-step-extra temporal lag."""

    def __init__(self, *args, arrangement: str, **kwargs):
        if arrangement not in {"CORRECT", "COORDINATE_ROLL", "ONE_EXTRA_STEP_LAG"}:
            raise ValueError("unknown online residual arrangement")
        super().__init__(*args, compensation_enabled=True, **kwargs)
        self.arrangement = arrangement
        self._lagged = {}
        self._pending = {}

    def _read(self, state, compensation):
        from torchao.optim.subclass_8bit import OptimState8bit

        if isinstance(state, OptimState8bit) and state.signed:
            residual = compensation
            if self.arrangement == "COORDINATE_ROLL":
                residual = roll_within_blocks(residual, state.block_size)
            elif self.arrangement == "ONE_EXTRA_STEP_LAG":
                key = id(state)
                self._pending[key] = compensation.detach().clone()
                residual = self._lagged.get(key, torch.zeros_like(compensation))
            return state.dequantize() + residual.float()
        return super()._read(state, compensation)

    def _write(self, state, value, compensation):
        from torchao.optim.subclass_8bit import OptimState8bit

        if (isinstance(state, OptimState8bit) and state.signed
                and self.arrangement == "ONE_EXTRA_STEP_LAG"):
            key = id(state)
            prior = self._pending.pop(key)
            super()._write(state, value, compensation)
            self._lagged[key] = prior
            return
        super()._write(state, value, compensation)
