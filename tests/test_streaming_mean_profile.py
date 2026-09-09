import pytest
import torch
from kernel_analyzer.streaming_mean_profile import StreamingMeanProfile


def collect(u,r,cal=3):
    recorder=StreamingMeanProfile(cal,len(u)-cal)
    for x,y in zip(u,r): recorder.append(x,y)
    return recorder.finish()


def test_full_vector_calculation_and_inputs_unchanged():
    gen=torch.Generator().manual_seed(873)
    u=torch.randn(8,37,generator=gen,dtype=torch.float64)
    r=torch.randn(8,37,generator=gen,dtype=torch.float64)
    original_u=u.clone(); original_r=r.clone()
    result=collect(u,r)
    scale=r[3:].square().sum(1).mean().sqrt()
    q=u-(u*r).sum(1,keepdim=True)/r.square().sum(1,keepdim=True)*r
    assert result['confirmation_mean_relative_magnitude']==pytest.approx(float(u[3:].mean(0).norm()/scale))
    assert result['confirmation_residual_mean_relative_magnitude']==pytest.approx(float(q[3:].mean(0).norm()/scale))
    direction=u[:3].mean(0); direction/=direction.norm()
    assert result['normalized_heldout_effect_projections']==pytest.approx((u[3:]@direction/scale).tolist())
    assert torch.equal(u,original_u) and torch.equal(r,original_r)


def test_unseen_orthogonal_direction_is_not_reported_as_zero_mean():
    u=torch.tensor([[0.,.001,0.],[0.,.001,0.],[0.,0.,20.],[0.,0.,20.]])
    r=torch.tensor([[1.,0.,0.]]*4)
    result=collect(u,r,2)
    assert result['confirmation_mean_relative_magnitude']==20
    assert result['normalized_heldout_effect_projections']==[0.,0.]
    assert result['confirmation_residual_mean_relative_magnitude']==20


def test_zero_calibration_direction_is_unavailable_not_zero_projection():
    u=torch.tensor([[0.,1.],[0.,-1.],[0.,3.],[0.,3.]])
    r=torch.tensor([[1.,0.]]*4)
    result=collect(u,r,2)
    assert result['normalized_heldout_effect_projections']==[None,None]
    assert result['confirmation_mean_relative_magnitude']==3


def test_small_repair_does_not_silently_drop_state():
    u=torch.ones(4,2); r=torch.ones(4,2); r[0].zero_()
    result=collect(u,r,2)
    assert result['confirmation_residual_mean_relative_magnitude'] is None
    assert len(result['rows'])==4
    assert result['confirmation_mean_relative_magnitude']==pytest.approx(1.)


def test_incomplete_and_changing_coordinates_rejected():
    rec=StreamingMeanProfile(1,1); rec.append(torch.ones(2),torch.ones(2))
    with pytest.raises(ValueError): rec.finish()
    with pytest.raises(ValueError): rec.append(torch.ones(3),torch.ones(3))


def test_no_population_or_equivalence_claim():
    result=collect(torch.zeros(4,2),torch.ones(4,2),2)
    assert not result['population_guarantee']
    assert result['decision_role']=='DESCRIPTIVE_NOT_USED_FOR_EQUIVALENCE'
