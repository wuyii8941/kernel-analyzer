"""Declared elementwise baseline on actual same-input outputs, not Gram data."""
import math
import torch


def compare_outputs(candidate, reference, *, rtol, atol):
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError('Tolerances must be finite and nonnegative')
    if candidate.shape != reference.shape or candidate.numel() == 0:
        raise ValueError('Output shapes must match and be nonempty')
    if not candidate.is_floating_point() or not reference.is_floating_point():
        raise ValueError('Floating outputs required')
    c, r = candidate.detach().double(), reference.detach().double()
    result = {'rtol': rtol, 'atol': atol, 'elements': c.numel(),
              'rule': 'abs(candidate-reference) <= atol + rtol*abs(reference)',
              'comparison_dtype': 'float64', 'policy_origin': 'DECLARED_ANALYSIS_BASELINE_NOT_KERNEL_AUTHOR_TEST'}
    if not bool(torch.isfinite(c).all() & torch.isfinite(r).all()):
        return {**result, 'status': 'NONFINITE_OUTPUT', 'allclose': None}
    error = (c-r).abs()
    violations = int(torch.count_nonzero(error > atol + rtol*r.abs()))
    return {**result, 'status': 'FINITE_COMPARISON', 'allclose': violations == 0,
            'violating_coordinates': violations, 'max_absolute_error': float(error.max()),
            'max_absolute_reference': float(r.abs().max())}
