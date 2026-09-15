import numpy as np
import pytest
from scipy.stats import ttest_1samp

from kernel_analyzer.mean_inference import mean_inference_p, student_quantile
from kernel_analyzer.population_direction import population_positive_direction_prevalence
from kernel_analyzer.training_bias_profile import _branch_result


def test_small_sample_quantile():
    assert student_quantile(1, .975) == pytest.approx(12.7062047364)
    assert student_quantile(7, .975) == pytest.approx(2.3646242516)


def test_production_mean_matches_student_test():
    values = np.array([.1, .7, -.2, 1.2, .5, -.1, .9, .6])
    result = _branch_result(values, direction_must_repeat=False, draws=999, seed=4)
    assert mean_inference_p(result) == pytest.approx(ttest_1samp(values, 0).pvalue)


def test_constant_sample_does_not_establish_population_mean():
    result = _branch_result(np.ones(32), direction_must_repeat=False, draws=999, seed=4)
    assert result['raw_studentized_signflip_p'] < .05
    assert mean_inference_p(result) == 1
    assert not result['raw_confirmed']


def test_legacy_symmetry_p_cannot_substitute_for_mean():
    with pytest.raises(ValueError, match='recompute'):
        mean_inference_p({'raw_studentized_signflip_p': .0001})
    assert mean_inference_p({'raw_studentized_mean_p': .8, 'raw_studentized_signflip_p': .0001}) == .8


def test_direct_direction_entry_handles_zeros():
    assert population_positive_direction_prevalence([0.] * 32)['decision'] == 'POSITIVE_DIRECTION_FREQUENCY_BELOW_NULL'
