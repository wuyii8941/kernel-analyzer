"""Reference for selected-logit NLL using precomputed normalization terms.

This is a local formula, not a reference for upstream softmax or a runtime
binding. Labels have already been shifted by the caller. Negative labels other
than ignore_index are not accepted, even if generated indexing wraps them.
"""
import torch


def selected_nll(logits, labels, row_max, log_exp_sum, ignore_index=-100):
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError('Expected floating token-by-vocabulary logits')
    n, vocab = logits.shape
    if n == 0 or vocab == 0 or labels.shape != (n,) or labels.dtype != torch.int64:
        raise ValueError('Invalid labels or empty logits')
    if row_max.shape != (n,) or log_exp_sum.shape != (n,):
        raise ValueError('Normalization shape mismatch')
    if any(x.device != logits.device for x in (labels, row_max, log_exp_sum)):
        raise ValueError('Device mismatch')
    if not all(x.is_floating_point() and torch.isfinite(x).all()
               for x in (logits, row_max, log_exp_sum)):
        raise ValueError('Finite floating inputs required')
    valid = labels != ignore_index
    if ((labels[valid] < 0) | (labels[valid] >= vocab)).any():
        raise ValueError('Label outside declared vocabulary')
    count = valid.sum()
    if count == 0:
        raise ValueError('No valid labels; mean NLL undefined')
    selected = logits.double().gather(1, labels.masked_fill(~valid, 0)[:, None]).squeeze(1)
    losses = -(selected-row_max.double()-log_exp_sum.double())
    return losses.masked_fill(~valid, 0).sum()/count, count


def selected_nll_backward(logits, labels, row_max, log_exp_sum, total_weight,
                          grad_output, ignore_index=-100):
    """Local fused NLL/log-softmax derivative with declared saved inputs.

    Returns an unrounded float64 mathematical reference. The runtime adapter
    must declare output storage dtype and clone logits before in-place writes.
    Normalization terms are consumed as saved, not silently recomputed.
    """
    _, count = selected_nll(logits, labels, row_max, log_exp_sum, ignore_index)
    for name, value in (('total_weight', total_weight), ('grad_output', grad_output)):
        if (not isinstance(value, torch.Tensor) or value.numel() != 1
                or value.device != logits.device or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError('Invalid scalar ' + name)
    if total_weight.item() != count.item():
        raise ValueError('Saved total weight differs from unweighted valid-label count')
    probabilities = (logits.double()-row_max.double()[:, None]-log_exp_sum.double()[:, None]).exp()
    if not torch.isfinite(probabilities).all():
        raise ValueError('Nonfinite reconstructed probabilities')
    valid = labels != ignore_index
    gradient = probabilities.clone()
    rows = torch.arange(logits.shape[0], device=logits.device)[valid]
    gradient[rows, labels[valid]] -= 1
    gradient[~valid] = 0
    return gradient * (grad_output.double().reshape(())/total_weight.double().reshape(()))
