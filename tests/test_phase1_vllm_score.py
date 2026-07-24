from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.phase1_vllm_score import chosen_prompt_logprob, extract_response_rows


def test_chosen_prompt_logprob_accepts_vllm_object() -> None:
    entry = {7: SimpleNamespace(logprob=-1.25)}
    assert chosen_prompt_logprob(entry, 7) == -1.25


def test_extract_response_rows_uses_full_prompt_positions() -> None:
    sample = {"case_id": "c", "prompt_ids": [1, 2], "response_ids": [3, 4]}
    output = SimpleNamespace(
        prompt_logprobs=[None, {2: SimpleNamespace(logprob=-0.2)}, {3: SimpleNamespace(logprob=-0.3)}, {4: SimpleNamespace(logprob=-0.4)}]
    )
    rows = extract_response_rows(sample, output)
    assert [(row["token_index"], row["token_id"], row["logp"]) for row in rows] == [
        (0, 3, -0.3),
        (1, 4, -0.4),
    ]


def test_extract_response_rows_rejects_missing_actual_token() -> None:
    sample = {"case_id": "c", "prompt_ids": [1], "response_ids": [3]}
    output = SimpleNamespace(prompt_logprobs=[None, {9: SimpleNamespace(logprob=-0.3)}])
    with pytest.raises(KeyError):
        extract_response_rows(sample, output)
