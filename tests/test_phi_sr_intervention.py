import pytest
import torch

from scripts.run_phi_mm_sr_intervention import norm_match_update


def test_norm_match_preserves_direction_and_matches_energy():
    value = torch.tensor([3.0, 4.0])
    matched = norm_match_update(value, 10.0)
    assert torch.linalg.vector_norm(matched).item() == pytest.approx(10.0)
    assert torch.dot(value, matched).item() > 0.0
    assert matched[0].item() / matched[1].item() == pytest.approx(3.0 / 4.0)


def test_norm_match_rejects_zero_to_nonzero():
    with pytest.raises(ValueError):
        norm_match_update(torch.zeros(3), 1.0)
