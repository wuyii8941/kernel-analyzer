"""PyTorch 2.10 Inductor mean-MSE forward and complete input/target VJP."""
from __future__ import annotations

import torch
import torch.nn.functional as F


ROWS, COLS = 32, 128


def _loss_with_output(x, target, upstream):
    loss = F.mse_loss(x, target, reduction="mean")
    return loss * upstream, loss


def _forward_and_vjp(x, target, upstream):
    (dx, dtarget), (_, loss) = torch.func.grad_and_value(
        _loss_with_output, argnums=(0, 1), has_aux=True
    )(x, target, upstream)
    return loss, dx, dtarget


_compiled = torch.compile(_forward_and_vjp, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"x", "target", "upstream", "reduction"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["reduction"] != "mean":
        raise ValueError("this call fixes reduction='mean'; weight is absent")
    for name in ("x", "target"):
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != (ROWS, COLS)
                or tensor.dtype != torch.float32 or not tensor.is_cuda
                or not tensor.is_contiguous()):
            raise ValueError(
                f"{name} must be a contiguous CUDA float32 tensor of shape {(ROWS, COLS)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
    upstream = inputs["upstream"]
    if (not torch.is_tensor(upstream) or tuple(upstream.shape) != ()
            or upstream.dtype != torch.float32 or not upstream.is_cuda
            or not upstream.is_contiguous()):
        raise ValueError("upstream must be a contiguous CUDA float32 scalar tensor")
    if not bool(torch.isfinite(upstream)):
        raise ValueError("upstream must be finite")
    loss, dx, dtarget = _compiled(inputs["x"], inputs["target"], upstream)
    return {"loss": loss, "dx": dx, "dtarget": dtarget}
