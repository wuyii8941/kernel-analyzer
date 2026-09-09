import pytest
import torch
from scripts.same_dtype_semantic_observer import snapshot_external_inputs
from scripts.generated_nontriton_fp32_observer import fp32_external_reference


@pytest.mark.parametrize('symbol', ['mm', 'bmm', 'addmm'])
def test_reference_uses_pre_call_values_even_when_output_aliases_input(symbol):
    left = torch.arange(1., 5.).reshape(2, 2)
    right = torch.tensor([[2., 1.], [1., 3.]])
    if symbol == 'bmm':
        left, right = left.unsqueeze(0), right.unsqueeze(0)
    args = (left, right) if symbol != 'addmm' else (left, left, right)
    kwargs = {'out': left}
    if symbol == 'addmm':
        kwargs.update(alpha=2., beta=.5)
    expected = fp32_external_reference(symbol, args, kwargs).clone()
    saved_args, saved_kwargs = snapshot_external_inputs(args, kwargs)
    left.fill_(999.)
    assert 'out' not in saved_kwargs
    assert torch.equal(fp32_external_reference(symbol, saved_args, saved_kwargs), expected)
