"""Correct interpretation of the frozen v1 positive-frequency statistic.

The v1 implementation is retained unchanged for historical source-bound replay.
Its lower-tail label must not be interpreted as evidence of negative values:
zero observations also belong to the nonpositive event.
"""
from .population_direction import population_positive_direction_prevalence


def positive_direction_frequency(values, **kwargs):
    result = population_positive_direction_prevalence(values, **kwargs)
    result['legacy_decision'] = result['decision']
    if result['decision'] == 'OPPOSITE_DIRECTION_PREVALENCE_CONFIRMED':
        result['decision'] = 'POSITIVE_DIRECTION_FREQUENCY_BELOW_NULL'
    result['interpretation_version'] = 'positive-frequency-v2'
    result['negative_direction_prevalence_assessed'] = False
    return result
