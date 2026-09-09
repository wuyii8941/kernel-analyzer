import torch
import pytest

pytest.importorskip("torchao")

from torchao.optim.subclass_8bit import OptimState8bit

from kernel_analyzer.optimizer_state_repairs import (
    AdamWFirstMomentFP32,
    AdamWSecondMomentFP32,
)


def initialized_states(optimizer_type):
    parameter = torch.nn.Parameter(torch.zeros(4096))
    optimizer = optimizer_type([parameter], block_size=256)
    return optimizer._new_buffer(parameter, True), optimizer._new_buffer(parameter, False)


def test_first_moment_repair_changes_only_first_moment_storage():
    first, second = initialized_states(AdamWFirstMomentFP32)
    assert type(first) is torch.Tensor
    assert isinstance(second, OptimState8bit)


def test_second_moment_repair_changes_only_second_moment_storage():
    first, second = initialized_states(AdamWSecondMomentFP32)
    assert isinstance(first, OptimState8bit)
    assert type(second) is torch.Tensor
