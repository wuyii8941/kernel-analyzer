import torch

from kernel_analyzer.antithetic import reflected_endpoint


def test_reflected_endpoint_is_exact_when_dtype_can_represent_it():
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([1.25, 1.5], dtype=torch.float32)
    realized, exact, error = reflected_endpoint(candidate, reference)
    assert torch.equal(realized, torch.tensor([0.75, 2.5]))
    assert exact is True
    assert error == 0.0


def test_reflected_endpoint_reports_bf16_representability_error():
    reference = torch.tensor([1.003], dtype=torch.float32)
    candidate = torch.tensor([1.0], dtype=torch.bfloat16)
    realized, exact, error = reflected_endpoint(candidate, reference)
    assert realized.dtype == torch.bfloat16
    assert exact is False
    assert error > 0.0
