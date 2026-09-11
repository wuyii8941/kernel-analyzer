"""AdamW8bit development variant with low-precision state-error compensation.

TorchAO AdamW8bit quantizes each updated moment and uses its dequantized value
at the next step.  This variant stores the quantization residual in BF16 and
adds it back before the next recurrence.  It is a mechanism experiment, not a
drop-in production optimizer: only dense FP32 parameters and non-AMSGrad
AdamW are supported.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW8bitErrorCompensated(Optimizer):
    """Block-quantized AdamW moments plus BF16 recurrence compensation."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        block_size: int = 256,
        compensation_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError("lr, eps, and weight_decay must be nonnegative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must lie in [0, 1)")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if compensation_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("unsupported compensation dtype")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay))
        self.block_size = int(block_size)
        self.compensation_dtype = compensation_dtype

    def _new_state(self, parameter: Tensor, *, signed: bool):
        from torchao.optim.subclass_8bit import OptimState8bit

        if parameter.numel() >= 4096 and parameter.numel() % self.block_size == 0:
            quantized = OptimState8bit.zeros(
                parameter.shape, signed=signed, block_size=self.block_size,
                device=parameter.device,
            )
            compensation = torch.zeros_like(
                parameter, dtype=self.compensation_dtype,
            )
            return quantized, compensation
        return torch.zeros_like(parameter), None

    @staticmethod
    def _read(state: Tensor, compensation: Tensor | None) -> Tensor:
        from torchao.optim.subclass_8bit import OptimState8bit

        if isinstance(state, OptimState8bit):
            value = state.dequantize()
            if compensation is not None:
                value = value + compensation.float()
            return value
        return state.float()

    @staticmethod
    def _write(state: Tensor, value: Tensor, compensation: Tensor | None) -> None:
        from torchao.optim.subclass_8bit import OptimState8bit

        if isinstance(state, OptimState8bit):
            state.copy_(value)
            if compensation is None:
                raise RuntimeError("quantized state has no compensation")
            reconstructed = state.dequantize()
            compensation.copy_((value - reconstructed).to(compensation.dtype))
        else:
            state.copy_(value)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype is not torch.float32 or parameter.grad.is_sparse:
                    raise ValueError("only dense FP32 parameters are supported")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"], state["exp_avg_compensation"] = self._new_state(
                        parameter, signed=True,
                    )
                    state["exp_avg_sq"], state["exp_avg_sq_compensation"] = self._new_state(
                        parameter, signed=False,
                    )
                state["step"] += 1
                gradient = parameter.grad.float()
                first_previous = self._read(
                    state["exp_avg"], state["exp_avg_compensation"]
                )
                second_previous = self._read(
                    state["exp_avg_sq"], state["exp_avg_sq_compensation"]
                )
                first = first_previous.lerp(gradient, 1.0 - beta1)
                second = second_previous.lerp(gradient.square(), 1.0 - beta2)
                self._write(state["exp_avg"], first, state["exp_avg_compensation"])
                self._write(state["exp_avg_sq"], second, state["exp_avg_sq_compensation"])

                step = state["step"]
                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                updated = parameter.float()
                if group["weight_decay"]:
                    updated = updated - group["lr"] * group["weight_decay"] * updated
                denominator = second.sqrt() / bias2**0.5 + group["eps"]
                updated = updated - group["lr"] * (first / bias1) / denominator
                parameter.copy_(updated)
        return loss


def optimizer_state_storage_bytes(optimizer: Optimizer) -> int:
    """Count stored tensor bytes, including quantization metadata and compensation."""

    from torchao.optim.subclass_8bit import OptimState8bit

    seen: set[int] = set()
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            tensors = []
            if isinstance(value, OptimState8bit):
                tensors = [value.codes, value.scale, value.qmap]
            elif isinstance(value, Tensor):
                tensors = [value]
            for tensor in tensors:
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    total += tensor.numel() * tensor.element_size()
    return total
