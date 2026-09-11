import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

from kernel_analyzer.adamw8bit_error_compensation import (
    AdamW8bitErrorCompensated,
    optimizer_state_storage_bytes,
)


def _trajectory(optimizer_factory, gradients):
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096))
    optimizer = optimizer_factory([parameter])
    for gradient in gradients:
        parameter.grad = gradient.clone()
        optimizer.step()
    return parameter.detach().clone(), optimizer


def test_compensation_reduces_recursive_parameter_difference():
    generator = torch.Generator().manual_seed(7)
    gradients = [torch.randn(4096, generator=generator) * (index + 1)
                 for index in range(8)]
    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01)
    reference, _ = _trajectory(
        lambda values: torch.optim.AdamW(values, foreach=False, fused=False, **settings),
        gradients,
    )
    from torchao.optim import AdamW8bit
    baseline, _ = _trajectory(
        lambda values: AdamW8bit(values, block_size=256, **settings), gradients,
    )
    compensated, optimizer = _trajectory(
        lambda values: AdamW8bitErrorCompensated(values, block_size=256, **settings),
        gradients,
    )
    assert torch.linalg.vector_norm(compensated - reference) < torch.linalg.vector_norm(
        baseline - reference
    )
    assert optimizer_state_storage_bytes(optimizer) > 0


def test_small_parameter_uses_unquantized_state():
    parameter = torch.nn.Parameter(torch.ones(128))
    optimizer = AdamW8bitErrorCompensated([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.state[parameter]["exp_avg_compensation"] is None
