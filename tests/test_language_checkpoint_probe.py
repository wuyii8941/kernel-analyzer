import pytest
import torch

from scripts.probe_liger_language_checkpoints import Moments


def test_streamed_full_space_matches_direct_computation():
    gen = torch.Generator().manual_seed(42)
    u = torch.randn(32, 19, generator=gen, dtype=torch.float64)
    r = torch.randn(32, 19, generator=gen, dtype=torch.float64)
    stats = Moments()
    for x, y in zip(u, r):
        stats.add(x, y)
    result = stats.result()
    assert result["effect_mean_energy"] == pytest.approx(float(u.mean(0).square().sum()))
    assert result["aligned_ratio_of_sums"] == pytest.approx(float((u*r).sum()/r.square().sum()))
    assert result["split_half_mean_inner_product"] == pytest.approx(float((u[:16].mean(0)*u[16:].mean(0)).sum()))
    assert result["lag1_inner_product_sum"] == pytest.approx(float((u[:-1]*u[1:]).sum()))


def test_nonfinite_probe_is_not_zero_or_negative_evidence():
    with pytest.raises(ValueError, match="Nonfinite"):
        Moments().add(torch.tensor([float("nan")]), torch.ones(1))
