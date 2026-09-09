from __future__ import annotations

import torch

from kernel_analyzer.update_write import parameter_write_delta, legacy_parameter_write_delta, adamw_parameter_write


def test_small_proposal_can_exist_without_a_bfloat16_parameter_write() -> None:
    base = torch.tensor([1.0], dtype=torch.bfloat16)
    proposed = torch.tensor([1e-4], dtype=torch.float32)
    assert proposed.item() != 0.0
    assert parameter_write_delta(base, proposed).item() == 0.0


def test_float32_parameter_write_retains_the_same_proposal() -> None:
    base = torch.tensor([1.0], dtype=torch.float32)
    proposed = torch.tensor([0.125], dtype=torch.float32)
    assert parameter_write_delta(base, proposed).item() == proposed.item()


def test_proposal_must_not_be_rounded_before_addition() -> None:
    base = torch.tensor([1.0], dtype=torch.bfloat16)
    proposed = torch.tensor([0.00391])
    assert legacy_parameter_write_delta(base, proposed).item() == 0
    assert parameter_write_delta(base, proposed).item() == 0.0078125
    assert adamw_parameter_write(base, -torch.ones_like(base), learning_rate=.00391,
                                representation="DIRECT_STORED").item() == 0.0078125


def test_actual_warm_adamw_readback_and_caller_state_preserved() -> None:
    base = torch.tensor([1., -2.])
    gradient = torch.tensor([.3, -.7])
    first, second = torch.tensor([.02, -.03]), torch.tensor([.01, .02])
    snapshots = [x.clone() for x in (base, gradient, first, second)]
    p = torch.nn.Parameter(base.clone())
    p.grad = gradient.clone()
    opt = torch.optim.AdamW([p], lr=.01, betas=(.9, .95), weight_decay=.1,
                            foreach=False, fused=False)
    opt.state[p] = {"step": torch.tensor(7.), "exp_avg": first.clone(), "exp_avg_sq": second.clone()}
    opt.step()
    actual = adamw_parameter_write(base, gradient, first=first, second=second,
                                   prior_step=7, learning_rate=.01, weight_decay=.1)
    torch.testing.assert_close(actual, p.detach()-base, rtol=0, atol=0)
    for original, snapshot in zip((base, gradient, first, second), snapshots):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)


def test_fp32_master_write_is_not_bf16_materialization() -> None:
    base = torch.tensor([1.], dtype=torch.bfloat16)
    delta = adamw_parameter_write(base, -torch.ones_like(base), learning_rate=1e-4)
    assert delta.item() > 0
    assert (base.float()+delta).to(torch.bfloat16).item() == base.item()
