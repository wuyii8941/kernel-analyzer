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
