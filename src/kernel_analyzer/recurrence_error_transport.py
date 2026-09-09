"""Conditional error transport in a scalar/coordinatewise recurrence.

Given hC[t]=aC[t]*hC[t-1]+bC[t]+rounding[t] and the reference recurrence,
e[t]=aC[t]*e[t-1]+(aC[t]-aR[t])*hR[t-1]+(bC[t]-bR[t])+rounding[t].
This algebra does not imply that any term has nonzero population mean.
"""


def reconstruct(candidate_decay, reference_decay, reference_previous,
                injection_difference, rounding_remainder, initial_difference):
    import torch
    tensors = (candidate_decay, reference_decay, reference_previous,
               injection_difference, rounding_remainder)
    shape = candidate_decay.shape
    if (len(shape) < 1 or shape[0] == 0 or initial_difference.shape != shape[1:]
            or candidate_decay.dtype not in (torch.float32, torch.float64)
            or any(t.shape != shape or t.dtype != candidate_decay.dtype
                   or t.device != candidate_decay.device for t in tensors)
            or initial_difference.dtype != candidate_decay.dtype
            or initial_difference.device != candidate_decay.device
            or any(not torch.isfinite(t).all() for t in (*tensors, initial_difference))):
        raise ValueError('Incompatible finite recurrence records')
    coefficient = (candidate_decay-reference_decay)*reference_previous
    inherited, coefficient_effect, injection_effect, rounding_effect = [], [], [], []
    state = initial_difference
    c = torch.zeros_like(state); b = torch.zeros_like(state); r = torch.zeros_like(state)
    for i in range(shape[0]):
        state = candidate_decay[i]*state
        c = candidate_decay[i]*c + coefficient[i]
        b = candidate_decay[i]*b + injection_difference[i]
        r = candidate_decay[i]*r + rounding_remainder[i]
        inherited.append(state); coefficient_effect.append(c)
        injection_effect.append(b); rounding_effect.append(r)
    pieces = dict(initial=torch.stack(inherited), decay=torch.stack(coefficient_effect),
                  injection=torch.stack(injection_effect), rounding=torch.stack(rounding_effect))
    return dict(components=pieces, reconstructed_difference=sum(pieces.values()),
                population_bias_proved=False, actual_kernel_replayed=False)
