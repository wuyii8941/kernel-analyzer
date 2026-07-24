from __future__ import annotations

import pytest

from scripts.phase10_mutation_trajectory_arm import replay_batch_hash
from scripts.phase11_heldout_latency import branch_fork_count


def test_replay_batch_hash_changes_with_training_state() -> None:
    samples = [{"case_id": "c", "prompt_ids": [1], "response_ids": [2]}]
    state = {
        "case_id": "c",
        "token_index": 0,
        "token_id": 2,
        "old_logp": -1.0,
        "advantage": 0.5,
    }
    first = replay_batch_hash(samples, [state])
    second = replay_batch_hash(samples, [{**state, "old_logp": -1.1}])
    assert first != second
    assert first == replay_batch_hash(samples, [state])


def test_branch_fork_count_ignores_inapplicable_tokens() -> None:
    clean = {"clip_active": [False, None, True, False]}
    mutation = {"clip_active": [True, True, True, False]}
    assert branch_fork_count(clean, mutation) == 1


def test_branch_fork_count_rejects_alignment_mismatch() -> None:
    with pytest.raises(ValueError):
        branch_fork_count({"clip_active": [False]}, {"clip_active": [False, True]})
