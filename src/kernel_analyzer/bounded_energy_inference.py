"""Conditional finite-sample Q test; requires independently justified bounds.

This does not replace the historical studentized protocol. A caller must establish
iid sampling, 0 < E[B], and population bounds 0 <= X <= x_max, 0 <= B <= b_max.
Sample maxima are not admissible substitutes for these population bounds.
"""
import math


def bounded_q_test(effect_energy, repair_energy, *, margin, x_max, b_max,
                   bound_provenance, alpha=0.05):
    x, b = list(effect_energy), list(repair_energy)
    if not bound_provenance or not isinstance(bound_provenance, str):
        raise ValueError('An external population-bound justification is required')
    if not x or len(x) != len(b):
        raise ValueError('Paired nonempty independent-unit summaries required')
    if (not all(math.isfinite(v) for v in (margin, x_max, b_max, alpha))
            or margin <= 0 or x_max < 0 or b_max <= 0 or not 0 < alpha < 1):
        raise ValueError('Invalid declared bounds or margin')
    if any(not math.isfinite(v) or v < 0 or v > x_max for v in x):
        raise ValueError('Effect energy violates the declared population bound')
    if any(not math.isfinite(v) or v < 0 or v > b_max for v in b):
        raise ValueError('Repair energy violates the declared population bound')
    # D = X - margin^2 B lies in [-margin^2 b_max, x_max].
    squared_margin = margin * margin
    if not math.isfinite(squared_margin) or squared_margin == 0:
        raise ValueError('Margin arithmetic overflow or underflow')
    scaled_bound = squared_margin * b_max
    width = x_max + scaled_bound
    if not math.isfinite(width) or scaled_bound == 0:
        raise ValueError('Bound arithmetic overflow or underflow')
    mean = math.fsum((xi - squared_margin * bi) / len(x) for xi, bi in zip(x, b))
    radius = width * math.sqrt(-math.log(alpha) / (2 * len(x)))
    upper, lower = mean + radius, mean - radius
    if not all(math.isfinite(v) for v in (mean, radius, upper, lower)) or radius == 0:
        raise ValueError('Inference arithmetic overflow or underflow')
    return dict(schema='bounded-energy-inference-candidate-v1',
        decision='EQUIVALENT' if upper < 0 else 'NON_EQUIVALENT' if lower > 0 else 'INCONCLUSIVE',
        mean_energy_contrast=mean, one_sided_upper=upper, one_sided_lower=lower,
        alpha_per_direction=alpha, interval_is_simultaneous=False,
        bound_provenance=bound_provenance, independent_units=len(x),
        assumptions=['IID_UNITS', 'POSITIVE_POPULATION_REPAIR_ENERGY',
                     'EXTERNALLY_JUSTIFIED_POPULATION_ENERGY_BOUNDS'],
        applicability_to_real_training='REQUIRES_BOUND_AND_SAMPLING_AUDIT',
        replaces_historical_protocol=False)
