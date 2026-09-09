import torch
import pytest
from kernel_analyzer.depthwise_conv1d_reference import evaluate
from kernel_analyzer.depthwise_conv1d_reference import reference
from kernel_analyzer.depthwise_conv1d_reference import evaluate_with_separate_bias


def test_kernel_order_and_channels():
    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64)
    w = torch.tensor([[[2., 3.]], [[5., 7.]]], dtype=torch.float64)
    result = evaluate(x, w, padding=1, accumulation_dtype=torch.float64)
    assert result.tolist() == [[[3., 8., 13., 6.], [28., 55., 67., 30.]]]
    assert torch.equal(result, torch.nn.functional.conv1d(x, w, padding=1, groups=2))


def test_observed_length_and_no_mutation():
    x = torch.ones(1, 3, 64, dtype=torch.bfloat16)
    w = torch.ones(3, 1, 4, dtype=torch.bfloat16)
    result = evaluate(x, w, padding=3)
    assert result.shape == (1, 3, 67)
    assert result[0, 0, :5].tolist() == [1, 2, 3, 4, 4]
    assert torch.all(x == 1) and torch.all(w == 1)


def test_reject_unsupported_groups_and_nonfinite():
    x = torch.ones(1, 2, 4)
    with pytest.raises(ValueError): evaluate(x, torch.ones(2, 2, 2), padding=1)
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): evaluate(x, torch.ones(2, 1, 2), padding=1)


def test_shared_external_observer_metadata():
    x = torch.ones(1, 1536, 2, dtype=torch.bfloat16)
    w = torch.ones(1536, 1, 4, dtype=torch.bfloat16)
    candidate = torch.empty(1, 1536, 5, dtype=torch.bfloat16)
    contract = dict(groups=1536, padding=3, required_weight_shape=[1536, 1, 4], bias_included=False)
    metadata = dict(external_symbol='convolution', reference_operand_capture='PRE_INVOCATION_CLONE',
                    runtime_args=(x, w), runtime_kwargs=dict(stride=(1,), padding=(3,),
                        dilation=(1,), transposed=False, output_padding=(0,), groups=1536, bias=None))
    assert torch.equal(reference(metadata, candidate, contract), evaluate(x, w, padding=3))
    metadata['runtime_kwargs']['bias'] = torch.zeros(1536)
    with pytest.raises(ValueError): reference(metadata, candidate, contract)


def test_separate_bias_preserves_intermediate_bf16_write():
    # 1 + 1/256 ties to 1 in BF16 before bias subtracts one. A fused
    # high-precision calculation instead leaves 1/256; do not conflate them.
    x = torch.tensor([[[1., 1./256]]], dtype=torch.bfloat16)
    w = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    bias = torch.tensor([-1.], dtype=torch.bfloat16)
    result = evaluate_with_separate_bias(x, w, bias, padding=0)
    assert result.item() == 0
    assert (x.double().sum()-1).item() == 1./256
