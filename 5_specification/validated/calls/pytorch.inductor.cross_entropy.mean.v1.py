"""Public PyTorch 2.10 call proposed for the CE contract-binding review.

The call returns the mean, unsmoothed class-index cross entropy and the full
VJP with upstream 1.  Both are produced by one ``torch.compile`` region.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _loss(logits, target):
    return F.cross_entropy(
        logits,
        target,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
    )


def _loss_and_gradient(logits, target):
    gradient, loss = torch.func.grad_and_value(_loss)(logits, target)
    return loss, gradient


_compiled = torch.compile(_loss_and_gradient, fullgraph=True, backend="inductor")


def call(inputs):
    """Run the fixed public boundary named by the candidate binding plan."""
    required = {"logits", "target", "weights", "eps", "ignore_index", "reduction", "upstream"}
    if set(inputs) != required:
        raise ValueError(f"expected inputs {sorted(required)}, received {sorted(inputs)}")
    logits, target = inputs["logits"], inputs["target"]
    if not torch.is_tensor(logits) or logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("logits must be a rank-2 floating tensor")
    if not torch.is_tensor(target) or target.ndim != 1 or target.dtype != torch.int64:
        raise ValueError("target must be a rank-1 int64 tensor")
    if target.shape[0] != logits.shape[0]:
        raise ValueError("target length must equal the number of logits rows")
    if inputs["weights"] is not None or inputs["eps"] != 0 or inputs["ignore_index"] != -100 or \
            inputs["reduction"] != "mean" or inputs["upstream"] != 1:
        raise ValueError("this candidate fixes weights=None, eps=0, ignore_index=-100, reduction=mean, upstream=1")
    loss, gradient = _compiled(logits, target)
    return {"loss": loss, "gradient": gradient}
