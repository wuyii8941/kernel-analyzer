import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

from kernel_analyzer.compensation_control import TensorScalarCompensationControl
from scripts.replay_failure_checkpoint_controls import restore_optimizer


@pytest.mark.parametrize("enabled", [False, True])
def test_restored_quantized_optimizer_keeps_residual_dtype_and_next_writes(enabled):
    generator = torch.Generator().manual_seed(184)
    p = torch.nn.Parameter(torch.randn(4096, generator=generator))
    opt = TensorScalarCompensationControl([p], compensation_enabled=enabled)
    for _ in range(3):
        p.grad = torch.randn(4096, generator=generator)
        opt.step()
    snapshot = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    restored = TensorScalarCompensationControl([q], compensation_enabled=enabled)
    restore_optimizer(restored, snapshot)
    assert restored.state[q]["exp_avg_compensation"].dtype == torch.bfloat16
    for _ in range(3):
        gradient = torch.randn(4096, generator=generator)
        p.grad, q.grad = gradient.clone(), gradient.clone()
        opt.step()
        restored.step()
        assert torch.equal(p, q)
        assert torch.equal(opt.state[p]["exp_avg_compensation"], restored.state[q]["exp_avg_compensation"])
