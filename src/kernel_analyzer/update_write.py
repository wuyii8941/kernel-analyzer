"""Measure optimizer proposals separately from stored-parameter writes."""

from __future__ import annotations

import torch


WRITE_PROTOCOL = {
    "version": "adamw-readback-v2",
    "implementation": "torch.optim.AdamW",
    "representation": "FP32_MASTER_INITIALIZED_FROM_STORED_PARAMETER",
    "foreach": False,
    "fused": False,
    "torch_version": torch.__version__,
    "measurement": "parameter_after_step_minus_parameter_before_step",
}


def parameter_write_delta(base: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
    """Measure a mixed-dtype addition, not an optimizer transition.

    Optimizer evidence must use :func:`adamw_parameter_write` instead.  Casting
    the proposal first introduces an additional rounding operation.
    """

    stored = base.detach()
    after = stored.clone()
    after.add_(proposed.to(device=stored.device))
    return after.float() - stored.float()


def legacy_parameter_write_delta(base: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
    """Historical v1 map, retained only to reproduce its extra-rounding results."""

    stored = base.detach()
    return ((stored + proposed.to(stored.dtype)) - stored).float()


def adamw_parameter_write(
    base: torch.Tensor, gradient: torch.Tensor, *,
    first: torch.Tensor | None = None, second: torch.Tensor | None = None,
    prior_step: int = 0, learning_rate: float = 1e-4,
    beta1: float = 0.9, beta2: float = 0.95, epsilon: float = 1e-8,
    weight_decay: float = 0.0, representation: str = "FP32_MASTER",
) -> torch.Tensor:
    """Clone a declared state, execute AdamW, and read back the parameter.

    FP32_MASTER starts from ``base.float()``; it does not reconstruct a lost
    historical master parameter. DIRECT_STORED uses the original dtype for
    parameters, gradients and moments. Neither mode mutates caller state.
    """
    if representation not in {"FP32_MASTER", "DIRECT_STORED"}:
        raise ValueError("unknown parameter representation")
    if prior_step < 0 or (prior_step and (first is None or second is None)):
        raise ValueError("warm replay requires both moments and a nonnegative step")
    dtype = torch.float32 if representation == "FP32_MASTER" else base.dtype
    before = base.detach().to(dtype=dtype).clone()
    if gradient.shape != before.shape:
        raise ValueError("gradient and parameter shapes differ")
    parameter = torch.nn.Parameter(before.clone())
    parameter.grad = gradient.detach().to(device=before.device, dtype=dtype).clone()
    optimizer = torch.optim.AdamW(
        [parameter], lr=learning_rate, betas=(beta1, beta2), eps=epsilon,
        weight_decay=weight_decay, foreach=False, fused=False,
    )
    if first is not None or second is not None:
        if first is None or second is None or first.shape != base.shape or second.shape != base.shape:
            raise ValueError("both matched moment tensors are required")
        optimizer.state[parameter] = {
            "step": torch.tensor(float(prior_step)),
            "exp_avg": first.detach().to(device=before.device, dtype=dtype).clone(),
            "exp_avg_sq": second.detach().to(device=before.device, dtype=dtype).clone(),
        }
    optimizer.step()
    return parameter.detach().float() - before.float()
