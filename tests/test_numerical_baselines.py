import numpy as np
import pytest
from kernel_analyzer.numerical_baselines import stage_baselines
from test_training_numerical_analysis import artifact


def test_scaling_and_mean_are_not_the_same_quantity():
    r=np.tile(np.array([[1.,0.],[-1.,0.]]),(8,1))
    row=stage_baselines(artifact(.02*r,r))[0]
    assert row['relative_rms']==pytest.approx(.02)
    assert row['aligned_ratio_of_sums']==pytest.approx(.02)
    assert row['mean_relative_magnitude']==pytest.approx(0)
    assert row['residual_mean_relative_magnitude']==pytest.approx(0)


def test_residual_mean_uses_statewise_projection():
    r=np.tile([1.,0.],(16,1)); u=np.tile([.02,.03],(16,1))
    row=stage_baselines(artifact(u,r))[0]
    assert row['residual_mean_relative_magnitude']==pytest.approx(.03)


def test_sketch_is_not_silently_full_space_mean():
    raw=artifact(np.ones((16,2)),np.ones((16,2)))
    raw['stages']['PARAMETER_WRITE']['COUNT_SKETCH_V2_0']=raw['stages']['PARAMETER_WRITE'].pop('EXACT')
    row=stage_baselines(raw)[0]
    assert row['mean_status']=='FULL_GRAM_UNAVAILABLE'
    assert 'mean_relative_magnitude' not in row


def test_gram_disagreement_is_reported():
    raw=artifact(np.ones((16,2)),np.ones((16,2)))
    raw['original_coordinate_statistics']['PARAMETER_WRITE'][0]['effect_energy']=4
    assert stage_baselines(raw)[0]['mean_status']=='GRAM_ORIGINAL_STATISTICS_DISAGREE'


def test_missing_local_does_not_erase_valid_update():
    raw=artifact(np.ones((16,2)),np.ones((16,2)))
    raw['original_coordinate_statistics']['LOCAL']=[]
    rows=stage_baselines(raw)
    assert rows[0]['status']=='VALID'
    assert rows[1]['status']=='ORIGINAL_STATISTICS_INCOMPLETE'


def test_single_state_driven_energy_is_visible_without_changing_statistics():
    r=np.ones((16,2)); u=np.zeros_like(r); u[-1,0]=1.
    row=stage_baselines(artifact(u,r))[0]
    assert row['state_count']==8
    assert row['nonzero_effect_states']==1
    assert row['largest_state_effect_energy_fraction']==1.
    assert row['relative_rms']==pytest.approx(.25)


def test_identity_has_no_dominant_effect_state():
    row=stage_baselines(artifact(np.zeros((16,2)),np.ones((16,2))))[0]
    assert row['nonzero_effect_states']==0
    assert row['largest_state_effect_energy_fraction']==0.
