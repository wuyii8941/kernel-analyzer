from __future__ import annotations

from scripts.phase12_mutation_sampling import compare_rows, self_failures


def row(**updates):
    value = {
        "case_id": "c",
        "token_index": 0,
        "token_id": 2,
        "log_normalizer": 1.0,
        "top_k_hash": "k",
        "top_p_hash": "p",
        "top_k_sampled": [1, 2],
        "top_p_sampled": [3, 4],
    }
    value.update(updates)
    return value


def test_compare_rows_distinguishes_set_and_sampling_forks() -> None:
    result = compare_rows(
        [row()],
        [row(log_normalizer=1.25, top_p_hash="other", top_k_sampled=[1, 5], top_p_sampled=[6, 4])],
    )[0]
    assert result["log_normalizer_delta"] == 0.25
    assert not result["top_k_candidate_set_fork"]
    assert result["top_p_candidate_set_fork"]
    assert result["top_k_sampling_fork_draws"] == 1
    assert result["top_p_sampling_fork_draws"] == 1
    assert result["top_p_first_draw_sampling_fork"]


def test_self_failures_requires_exact_reproduction() -> None:
    assert self_failures([row()], [row()]) == 0
    assert self_failures([row()], [row(top_p_sampled=[3, 9])]) == 1
