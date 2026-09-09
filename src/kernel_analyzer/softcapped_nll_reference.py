"""Local softcapped-logit loss reference; not a runtime support declaration.

Normalization is supplied from the compared execution. Float64 evaluates the
declared formula, not an absolute ground truth or a changed training objective.
"""
import math
import torch
from kernel_analyzer.selected_nll_reference import selected_nll_backward


def softcapped_nll_backward(logits, labels, row_max, log_exp_sum, total_weight,
                           grad_output, *, cap, ignore_index=-100):
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Finite positive softcap required')
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('Finite floating token-by-vocabulary logits required')
    t = torch.tanh(logits.double()/cap)
    gradient = selected_nll_backward(cap*t, labels, row_max, log_exp_sum,
                                    total_weight, grad_output, ignore_index)
    # d[c*tanh(x/c)]/dx = 1-tanh(x/c)^2. The source may materialize
    # multiply-by-c and divide-by-c separately; that rounding is not inherited.
    return gradient*(1-t.square())
