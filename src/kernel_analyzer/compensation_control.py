"""Same-code-path control; does not alter the frozen compensation implementation."""
from kernel_analyzer.adamw8bit_error_compensation import AdamW8bitErrorCompensated


class CompensationControl(AdamW8bitErrorCompensated):
    """Store residuals in both modes; only their contribution on read is toggled.

    Disabled mode intentionally retains memory/work, so this is a causal
    numerical comparison, not a performance implementation.
    """

    def __init__(self, *args, compensation_enabled: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.compensation_enabled = bool(compensation_enabled)

    def _read(self, state, compensation):
        from torchao.optim.subclass_8bit import OptimState8bit
        if isinstance(state, OptimState8bit):
            return state.dequantize() + compensation.float() * float(self.compensation_enabled)
        return state.float()


class TensorScalarCompensationControl(CompensationControl):
    """Development control matching TorchAO's CPU float32 lr/step arithmetic."""

    def __init__(self, *args, **kwargs):
        import torch
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            group['lr'] = torch.tensor(group['lr'], dtype=torch.float32)

    def step(self, closure=None):
        import torch
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and not self.state[parameter]:
                    state = self.state[parameter]
                    state['step'] = torch.tensor(0.0)
                    state['exp_avg'], state['exp_avg_compensation'] = self._new_state(parameter, signed=True)
                    state['exp_avg_sq'], state['exp_avg_sq_compensation'] = self._new_state(parameter, signed=False)
        return super().step(closure)
