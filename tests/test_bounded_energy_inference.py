import math
import pytest
from kernel_analyzer.bounded_energy_inference import bounded_q_test


def test_rare_event_counterexample_is_not_equivalent():
    for n in (16, 64, 256, 1024):
        p = 1/(20*n)
        result = bounded_q_test([0.0]*n, [1.0]*n, margin=.01,
            x_max=.01**2/p, b_max=1, bound_provenance='Declared Bernoulli support')
        assert result['decision'] == 'INCONCLUSIVE'


def test_exact_binomial_boundary_false_equivalence():
    # Enumerate every outcome count, not an estimated Monte Carlo frequency.
    for n in (8, 32, 100):
        p = .2
        probability = 0
        for k in range(n+1):
            result = bounded_q_test([1.0]*k+[0.0]*(n-k), [1.0]*n,
                margin=math.sqrt(p), x_max=1, b_max=1,
                bound_provenance='Declared Bernoulli support')
            if result['decision'] == 'EQUIVALENT':
                probability += math.comb(n,k)*p**k*(1-p)**(n-k)
        assert probability <= .05


def test_small_effect_can_pass_with_enough_units():
    assert bounded_q_test([0.0]*1000, [1.0]*1000, margin=.1,
        x_max=.01,b_max=1,bound_provenance='External support')['decision'] == 'EQUIVALENT'


def test_bound_violation_fails_closed():
    with pytest.raises(ValueError, match='violates'):
        bounded_q_test([2], [1], margin=.1,x_max=1,b_max=1,bound_provenance='External')


@pytest.mark.parametrize('margin', [1e-200, 1e200])
def test_extreme_margin_is_not_a_certificate(margin):
    with pytest.raises(ValueError, match='arithmetic'):
        bounded_q_test([0], [1], margin=margin, x_max=1, b_max=1,
                       bound_provenance='External')


def test_tiny_alpha_has_finite_log_calculation():
    result = bounded_q_test([0], [1], margin=.1, x_max=1, b_max=1,
                            alpha=1e-320, bound_provenance='External')
    assert math.isfinite(result['one_sided_upper'])
    assert result['decision'] == 'INCONCLUSIVE'
