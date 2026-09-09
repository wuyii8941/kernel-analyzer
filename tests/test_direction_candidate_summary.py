import pytest

from scripts.summarize_direction_candidates import interval_sign, replicated_sign


def test_interval_sign_requires_strict_separation_from_zero():
    assert interval_sign([1.,2.])==1
    assert interval_sign([-2.,-1.])==-1
    assert interval_sign([-1.,1.])==0
    assert interval_sign([0.,1.])==0
    with pytest.raises(ValueError):
        interval_sign([2.,1.])


def test_direction_must_reproduce_in_every_recorded_view():
    intervals={
        "a":{"additive":[1.,2.]},
        "b":{"additive":[2.,3.]},
    }
    assert replicated_sign(intervals,"additive")=="POSITIVE"
    intervals["b"]["additive"]=[-1.,1.]
    assert replicated_sign(intervals,"additive").startswith("NOT_REPRODUCED")
    intervals["b"]["additive"]=[-3.,-2.]
    assert replicated_sign(intervals,"additive").startswith("NOT_REPRODUCED")
