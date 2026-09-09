import numpy as np
import pytest
from scripts import run_training_bias_profile_v2_empirical as empirical
from kernel_analyzer.parallel_measurement import parallel_views


@pytest.mark.parametrize('size',[16,4096,4097,1000003])
def test_parallel_seeds_are_bitwise_identical_to_original(size):
    rng=np.random.default_rng(73)
    value=rng.normal(size=size).astype(np.float32)
    value[::5]*=1e-6
    original,n=empirical._compact_views(value)
    actual,m=parallel_views(value,seeds=empirical.SKETCH_SEEDS,dimension=empirical.SKETCH_DIMENSION,cache={},workers=3)
    assert n==m
    assert original.keys()==actual.keys()
    assert all(np.array_equal(original[k],actual[k]) for k in original)


def test_finite_bfloat16_mask_scale_does_not_overflow_sketch_storage():
    value=np.full(8192,-float(np.finfo(np.float32).max),dtype=np.float32)
    actual,n=parallel_views(value,seeds=(73,),dimension=4096,cache={},workers=1)
    assert n==value.size
    sketch=actual['COUNT_SKETCH_V3_FLOAT64_SEED_73']
    assert sketch.dtype==np.float64
    assert np.isfinite(sketch).all()
