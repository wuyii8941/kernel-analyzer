from scripts.phase14_task_reward_eval_once import arithmetic_prompt, numeric_reward


def test_numeric_reward_matches_training_definition() -> None:
    reward, predicted, exact = numeric_reward("reasoning, final 13", 13.0)
    assert (reward, predicted, exact) == (2.0, 13.0, True)
    reward, predicted, exact = numeric_reward("final 12", 13.0)
    assert predicted == 12.0
    assert not exact
    assert 0.0 < reward < 1.0


def test_arithmetic_prompt_uses_heldout_index() -> None:
    prompt, expected = arithmetic_prompt(64)
    assert "starts with 71" in prompt
    assert expected == 78.0
