from scripts.phase14_merge_task_reward import paired_bootstrap_ci, paired_comparison


def arm(name, rewards, tokens, exact):
    return {
        "arm": name,
        "rows": [
            {
                "dataset_index": index,
                "reward": reward,
                "completion_token_ids": token,
                "exact": outcome,
            }
            for index, (reward, token, outcome) in enumerate(zip(rewards, tokens, exact, strict=True))
        ],
    }


def test_paired_comparison_counts_semantic_levels() -> None:
    result = paired_comparison(
        arm("a", [1.0, 2.0], [[1], [2]], [False, True]),
        arm("b", [1.0, 1.0], [[1], [3]], [False, False]),
    )
    assert result["completion_token_forks"] == 1
    assert result["reward_differences"] == 1
    assert result["exact_outcome_forks"] == 1
    assert result["mean_reward_difference_right_minus_left"] == -0.5


def test_zero_difference_bootstrap_is_exact() -> None:
    assert paired_bootstrap_ci([0.0, 0.0], repeats=10) == [0.0, 0.0]
