import pytest
import torch

from scripts.build_positioned_input_bank import derive
from scripts.qwen_candidate_step import text_step_values


def test_text_step_values_preserves_legacy_single_input() -> None:
    values = text_step_values({"token_ids": [1, 2, 3]}, torch.device("cpu"))
    assert len(values) == 1
    assert values[0].tolist() == [[1, 2, 3]]


def test_positioned_bank_and_step_values_keep_short_window() -> None:
    bank = derive({"states": [{"state_id": "a", "token_ids": [4, 5]}]}, start=16384)
    state = bank["states"][0]
    values = text_step_values(state, torch.device("cpu"))
    assert values[0].tolist() == [[4, 5]]
    assert values[1].tolist() == [[16384, 16385]]
    assert bank["position_ids_protocol"][
        "token_values_or_numerical_results_used_for_selection"
    ] is False


def test_positioned_bank_can_take_a_predeclared_source_slice() -> None:
    source = {"states": [
        {"state_id": str(index), "token_ids": [index, index + 1]}
        for index in range(6)
    ]}
    bank = derive(source, start=7, row_offset=2, row_limit=3)
    assert [row["state_id"] for row in bank["states"]] == ["2", "3", "4"]
    assert all(row["position_ids"] == [7, 8] for row in bank["states"])
    assert bank["position_ids_protocol"]["source_row_offset"] == 2
    assert bank["position_ids_protocol"]["source_row_limit"] == 3


@pytest.mark.parametrize(
    "kwargs", [{"row_offset": -1}, {"row_limit": 0}, {"row_offset": 99}],
)
def test_invalid_positioned_source_slice_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        derive({"states": [{"token_ids": [1]}]}, start=0, **kwargs)


@pytest.mark.parametrize("positions", [[1], [-1, 0], [1.0, 2]])
def test_invalid_position_ids_fail_closed(positions) -> None:
    with pytest.raises(ValueError, match="position_ids"):
        text_step_values(
            {"token_ids": [1, 2], "position_ids": positions},
            torch.device("cpu"),
        )
