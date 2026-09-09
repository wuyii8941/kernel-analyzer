import pytest
import torch
from kernel_analyzer.grouped_causal_softmax_reference import evaluate, select_output


def test_all_outputs_grouped_causal_and_no_mutation():
    scores = torch.zeros(8, 4, dtype=torch.bfloat16)
    saved = scores.clone()
    ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    outputs = evaluate(scores, ids, 0.5)
    expected = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                             [.5, 0, .5, 0], [0, .5, 0, .5]], dtype=torch.bfloat16)
    assert torch.equal(outputs['out_ptr2'], expected.repeat(2, 1))
    assert torch.equal(outputs['out_ptr1'], torch.tensor([1., 1., 2., 2.] * 2))
    assert torch.equal(outputs['out_ptr0'], torch.zeros(8))
    assert outputs['in_out_ptr0'][0, 1] == torch.finfo(torch.bfloat16).min
    assert torch.equal(scores, saved)


def test_statistics_precede_low_precision_score_write():
    scores = torch.tensor([[1., 2.], [1., 2.]], dtype=torch.bfloat16)
    outputs = evaluate(scores, torch.zeros(2, dtype=torch.int64), .12345)
    assert outputs['out_ptr0'][1].item() != outputs['in_out_ptr0'][1].max().float().item()
    expected = torch.softmax(scores.float()[1] * torch.tensor(.12345), -1).bfloat16()
    assert torch.equal(outputs['out_ptr2'][1], expected)


@pytest.mark.parametrize('scale', [0., float('inf'), 1e100, 1e-100])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        evaluate(torch.zeros(2, 2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.int64), scale)


def test_invalid_layout_and_nonfinite():
    ids = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError):
        evaluate(torch.zeros(3, 2, dtype=torch.bfloat16), ids, 1.)
    with pytest.raises(ValueError):
        evaluate(torch.full((2, 2), float('nan'), dtype=torch.bfloat16), ids, 1.)
    with pytest.raises(ValueError):
        evaluate(torch.zeros(2, 2, dtype=torch.bfloat16), ids.int(), 1.)


def test_production_full_pointer_snapshot_is_accepted_but_unknown_pointer_is_not():
    scores = torch.zeros(4, dtype=torch.bfloat16)
    pointers = {
        'in_out_ptr0': scores,
        'in_ptr0': torch.zeros(2, dtype=torch.int64),
        'out_ptr0': torch.empty(2),
        'out_ptr1': torch.empty(2),
        'out_ptr2': torch.empty(4, dtype=torch.bfloat16),
    }
    candidate = torch.empty(4, dtype=torch.bfloat16)
    result = select_output(
        pointers, candidate, formal_pointer='out_ptr2', rows=2, width=2, scale=1.0
    )
    assert result.shape == candidate.shape
    with pytest.raises(ValueError, match='complete pointer ABI'):
        select_output(
            dict(pointers, unexpected=torch.empty(1)), candidate,
            formal_pointer='out_ptr2', rows=2, width=2, scale=1.0,
        )
