"""Measure consistency of observed saved scores and normalization statistics.

This is a mechanism diagnostic, not a bias test or a new reference policy.
Inputs must come from the same actual forward invocation.
"""


def evaluate(saved_scores, maximum, denominator, *, scale):
    import math
    import torch
    if (not isinstance(saved_scores, torch.Tensor) or saved_scores.ndim != 2
            or saved_scores.dtype != torch.bfloat16 or saved_scores.numel() == 0):
        raise ValueError('Expected BF16 saved score matrix')
    rows = saved_scores.shape[0]
    for value in (maximum, denominator):
        if (not isinstance(value, torch.Tensor) or value.shape != (rows,)
                or value.dtype != torch.float32 or value.device != saved_scores.device):
            raise ValueError('Expected same-call FP32 row statistics')
    if (type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0
            or any(not torch.isfinite(v).all() for v in (saved_scores, maximum, denominator))
            or not (denominator > 0).all()):
        raise ValueError('Invalid saved values or scale')
    # FP64 evaluation isolates consistency of the stored values from FP32 replay
    # noise. This is not claimed to reproduce the actual Triton instruction path.
    q = ((saved_scores.double() - maximum.double()[:, None]).exp()
         / denominator.double()[:, None])
    mass = q.sum(-1)
    response = scale * q * (1 - mass[:, None])
    if not torch.isfinite(q).all() or not torch.isfinite(response).all():
        raise ValueError('Nonfinite reconstructed diagnostic')
    return dict(reconstructed_probability=q, row_mass=mass,
                normalization_defect=mass-1,
                constant_cotangent_response=response,
                interpretation='CONDITIONAL_SAVED_STATE_CONSISTENCY_NOT_POPULATION_BIAS',
                actual_backward_replayed=False)
