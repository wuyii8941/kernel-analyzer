import torch
from kernel_analyzer.softmax_saved_state_diagnostic import evaluate


def test_consistent_uniform_state_has_zero_response():
    r = evaluate(torch.zeros(2, 4, dtype=torch.bfloat16), torch.zeros(2),
                 torch.full((2,), 4.), scale=.5)
    assert torch.equal(r['row_mass'], torch.ones(2, dtype=torch.float64))
    assert torch.count_nonzero(r['constant_cotangent_response']) == 0


def test_denominator_mismatch_predicts_constant_gradient_response():
    r = evaluate(torch.zeros(1, 4, dtype=torch.bfloat16), torch.zeros(1),
                 torch.tensor([8.]), scale=.5)
    q = r['reconstructed_probability']
    direct = .5 * (q - q * q.sum(-1, keepdim=True))
    assert torch.equal(r['constant_cotangent_response'], direct)
    assert (direct > 0).all()
    assert not r['actual_backward_replayed']


def test_recomputing_stats_from_saved_scores_removes_constructed_defect():
    # Controlled inconsistency is not a claim about naturally occurring bias.
    scores = torch.tensor([[0., 0.]], dtype=torch.bfloat16)
    inconsistent = evaluate(scores, torch.tensor([.01]), torch.tensor([2.]), scale=1.)
    repaired = evaluate(scores, torch.zeros(1), torch.tensor([2.]), scale=1.)
    assert inconsistent['normalization_defect'].abs().item() > .001
    assert repaired['normalization_defect'].item() == 0
