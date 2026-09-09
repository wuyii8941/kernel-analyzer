import hashlib
import pytest
from scripts.build_same_data_baselines import local_comparison_summary


def example():
    data = b'original capture'
    payload = dict(case_id='case', state_ids=['cal', 'confirm'], confirmation_state_ids=['confirm'])
    record = dict(case_id='case', raw_sha256=hashlib.sha256(data).hexdigest(), observations=[
        dict(state_id=s, rtol=1e-5, atol=1e-8, status='FINITE_COMPARISON', allclose=s != 'cal')
        for s in payload['state_ids']])
    return payload, data, record


def test_only_confirmation_is_summarized():
    result = local_comparison_summary(*example())
    assert result['local_allclose'] is True
    assert result['local_failed_states'] == 0
    assert result['local_tolerance_scope'] == 'DECLARED_BASELINE_NOT_KERNEL_AUTHOR_POLICY'


@pytest.mark.parametrize('error', ['digest', 'case', 'order', 'mixed', 'negative', 'unknown', 'not_bool', 'empty', 'absent', 'duplicate'])
def test_reject_unbound_or_invalid_comparison(error):
    p, data, r = example()
    if error == 'digest': r['raw_sha256'] = 'wrong'
    if error == 'case': r['case_id'] = 'other'
    if error == 'order': r['observations'].reverse()
    if error == 'mixed': r['observations'][0]['rtol'] = 0.1
    if error == 'negative':
        for o in r['observations']: o['atol'] = -1
    if error == 'unknown': r['observations'][0]['status'] = 'UNKNOWN'
    if error == 'not_bool': r['observations'][0]['allclose'] = 'False'
    if error == 'empty': p['confirmation_state_ids'] = []
    if error == 'absent': p['confirmation_state_ids'] = ['missing']
    if error == 'duplicate': p['confirmation_state_ids'] *= 2
    with pytest.raises(ValueError): local_comparison_summary(p, data, r)


def test_nonfinite_is_not_a_negative_or_pass():
    p, data, r = example()
    r['observations'][1].update(status='NONFINITE_OUTPUT', allclose=None)
    result = local_comparison_summary(p, data, r)
    assert result['local_allclose'] == 'NONFINITE_NOT_ASSESSED'
    assert result['local_nonfinite_states'] == 1
