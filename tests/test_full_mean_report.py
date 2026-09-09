import hashlib
import pytest
import torch
from kernel_analyzer.streaming_mean_profile import StreamingMeanProfile
from scripts.build_same_data_baselines import full_mean_summary


def example():
    collector=StreamingMeanProfile(2,2)
    for i in range(4): collector.append(torch.tensor([0.,1.]),torch.tensor([1.,0.]))
    result=collector.finish(); data=b'raw'
    p=dict(case_id='case',state_ids=['0','1','2','3'],calibration_state_ids=['0','1'],
           confirmation_state_ids=['2','3'],original_coordinate_statistics={'LOCAL':result['rows']})
    record={k:v for k,v in p.items() if k!='original_coordinate_statistics'}
    record.update(status='VERIFIED_IDENTICAL_RECAPTURE',raw_sha256=hashlib.sha256(data).hexdigest(),stages={'LOCAL':result})
    return p,data,record


def test_join_full_coordinate_mean():
    p,data,r=example(); out=full_mean_summary(p,data,r,'LOCAL')
    assert out['mean_relative_magnitude']==1
    assert out['residual_mean_relative_magnitude']==1
    assert out['mean_status']=='FULL_COORDINATE_STREAMING_FIXED_SUITE'


@pytest.mark.parametrize('bad',['hash','partition','scope','mean_bound','count','missing_mean','missing_residual'])
def test_reject_wrong_or_overclaimed_measurement(bad):
    p,data,r=example()
    if bad=='hash': r['raw_sha256']='other'
    if bad=='partition': r['confirmation_state_ids']=['3','2']
    if bad=='scope': r['stages']['LOCAL']['population_guarantee']=True
    if bad=='mean_bound': r['stages']['LOCAL']['confirmation_mean_relative_magnitude']=100
    if bad=='count': r['stages']['LOCAL']['confirmation_count']=3
    if bad=='missing_mean': r['stages']['LOCAL']['confirmation_mean_relative_magnitude']=None
    if bad=='missing_residual': r['stages']['LOCAL']['confirmation_residual_mean_relative_magnitude']=None
    with pytest.raises(ValueError): full_mean_summary(p,data,r,'LOCAL')
