import pytest
import torch
from kernel_analyzer.residual_rms_forward_reference import evaluate


def inputs():
    return (torch.ones(2, 4, dtype=torch.bfloat16),
            torch.full((2, 4), 1/256, dtype=torch.bfloat16),
            torch.ones(4, dtype=torch.bfloat16))


def test_denominator_uses_sum_before_bf16_write():
    x, residual, weight = inputs()
    written, inverse, output = evaluate(x, residual, weight, epsilon=1e-6)
    assert torch.equal(written, x)
    expected = torch.full((2,), ((1+1/256)**2+1e-6)**-.5)
    torch.testing.assert_close(inverse, expected)
    rounded_denominator = torch.rsqrt(written.float().square().mean(-1)+1e-6)
    assert not torch.equal(inverse, rounded_denominator)
    assert torch.equal(output, (written.float()*inverse[:, None]).bfloat16())


def test_inputs_not_mutated():
    values=inputs()
    saved=[v.clone() for v in values]
    evaluate(*values, epsilon=1e-6)
    assert all(torch.equal(a,b) for a,b in zip(values,saved))


@pytest.mark.parametrize('epsilon',[0,-1,float('nan'),True])
def test_bad_epsilon(epsilon):
    with pytest.raises(ValueError): evaluate(*inputs(),epsilon=epsilon)


def test_wrong_storage_dtype():
    x,r,w=inputs()
    with pytest.raises(ValueError): evaluate(x.float(),r,w,epsilon=1e-6)
