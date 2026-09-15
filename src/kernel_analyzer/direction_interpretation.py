"""Compatibility entry point for the corrected positive-frequency statistic."""
from .population_direction import population_positive_direction_prevalence


def positive_direction_frequency(values, **kwargs):
    result = population_positive_direction_prevalence(values, **kwargs)
    result['interpretation_version'] = 'positive-frequency-v2'
    result['negative_direction_prevalence_assessed'] = False
    return result
