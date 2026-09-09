import torch
import pytest
from kernel_analyzer.indexed_row_accumulation_reference import evaluate
from scripts.indexed_accumulation_observer import snapshot_indexed_inputs


def test_batched_duplicate_indices_keep_input_order():
    initial = torch.zeros(4, 2)
    indices = torch.tensor([[1, 1], [1, 2]])
    values = torch.tensor([[[2.**25, 2.], [1., 3.]], [[-2.**25, 4.], [7., 8.]]])
    captured = snapshot_indexed_inputs(initial, [indices], values, True)
    result = evaluate(**captured)
    assert torch.equal(result[1], torch.tensor([0., 9.]))
    assert torch.equal(result[2], torch.tensor([7., 8.]))
    assert torch.count_nonzero(initial) == 0


def test_noncontiguous_batch_indices_follow_logical_order():
    initial = torch.zeros(4, 2)
    indices = torch.tensor([[0, 1], [2, 0]]).t()
    values = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).transpose(0, 1)
    expected = initial.clone().index_put_((indices,), values, accumulate=True)
    assert torch.equal(evaluate(initial, indices, values), expected)


def test_broadcasting_still_rejected():
    with pytest.raises(ValueError):
        evaluate(torch.zeros(4, 2), torch.tensor([[1, 2]]), torch.zeros(2, 2))
