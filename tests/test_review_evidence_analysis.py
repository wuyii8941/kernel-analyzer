import math
import pytest
from scripts.analyze_review_evidence import geometry


def test_rotation_is_not_norm_shrinkage():
    result = geometry(2, 1, -1)
    assert result['candidate_reference_rms_ratio'] == 1
    assert result['aggregate_cosine'] == 0
    assert result['aligned_coefficient'] == -1


def test_scalar_shrinkage():
    result = geometry(.01, 1, -.1)
    assert result['candidate_reference_rms_ratio'] == pytest.approx(.9)
    assert result['aggregate_cosine'] == pytest.approx(1)


def test_identity():
    assert geometry(0, 1, 0)['candidate_reference_rms_ratio'] == 1


@pytest.mark.parametrize('values', [(1,0,0),(-1,1,0),(1,1,2),(math.inf,1,0)])
def test_invalid_energy(values):
    with pytest.raises(ValueError): geometry(*values)
