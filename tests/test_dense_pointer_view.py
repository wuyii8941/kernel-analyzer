import pytest
import torch
from kernel_analyzer.dense_pointer_view import dense_pointer_view


def test_transposed_snapshot_preserves_physical_order():
    physical = torch.arange(24.).reshape(4, 6)
    snapshot = physical.T.unsqueeze(0).clone()
    assert not snapshot.is_contiguous()
    torch.testing.assert_close(dense_pointer_view(snapshot, (4, 6)), physical)
    assert not torch.equal(snapshot.reshape(4, 6), physical)


def test_nonzero_storage_offset():
    x = torch.arange(30.)[6:].reshape(4, 6).T
    torch.testing.assert_close(dense_pointer_view(x, (4, 6)), torch.arange(6., 30.).reshape(4, 6))


@pytest.mark.parametrize('value', [torch.zeros(4, 6)[:, ::2], torch.zeros(1, 6).expand(4, 6)])
def test_reject_gaps_or_overlaps(value):
    with pytest.raises(ValueError, match='dense'):
        dense_pointer_view(value, (value.numel(),))


def test_reject_wrong_extent():
    with pytest.raises(ValueError, match='extent'):
        dense_pointer_view(torch.zeros(12), (3, 5))
