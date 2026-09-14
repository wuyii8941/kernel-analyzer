import torch
import pytest

from kernel_analyzer.structured_residual_control import roll_within_blocks


def test_block_roll_preserves_energy_and_changes_coordinates():
    value = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    moved = roll_within_blocks(value, 4)
    assert torch.equal(moved.reshape(-1)[:4], torch.tensor([3., 0., 1., 2.]))
    assert torch.equal(value.square().sum(), moved.square().sum())


def test_block_roll_is_block_local():
    value = torch.arange(8, dtype=torch.float32)
    moved = roll_within_blocks(value, 4)
    assert set(moved[:4].tolist()) == set(value[:4].tolist())
    assert set(moved[4:].tolist()) == set(value[4:].tolist())


def test_unknown_residual_arrangement_is_rejected():
    from kernel_analyzer.online_residual_control import OnlineFirstMomentResidualControl
    parameter = torch.nn.Parameter(torch.zeros(4096))
    try:
        OnlineFirstMomentResidualControl([parameter], arrangement="UNKNOWN")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arrangement was accepted")


def test_online_correct_path_matches_existing_control():
    pytest.importorskip("torchao")
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.online_residual_control import OnlineFirstMomentResidualControl
    left = torch.nn.Parameter(torch.linspace(-1, 1, 4096))
    right = torch.nn.Parameter(left.detach().clone())
    settings = dict(lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    expected = TensorScalarCompensationControl([left], compensation_enabled=True, **settings)
    actual = OnlineFirstMomentResidualControl([right], arrangement="CORRECT", **settings)
    generator = torch.Generator().manual_seed(17)
    for _ in range(4):
        gradient = torch.randn(4096, generator=generator)
        left.grad = gradient.clone(); right.grad = gradient.clone()
        expected.step(); actual.step()
    assert torch.equal(left, right)
