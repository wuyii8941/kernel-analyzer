from copy import deepcopy

import pytest

from scripts.normalize_segmented_aot_capture import unique_runtime_pairs


def runtime_pair(run_index: int) -> dict:
    return {
        "run_index": run_index,
        "forward_phase": {"graph_index": 4},
        "backward_phase": {"graph_index": 2},
        "forward_inputs": [{"placeholder": "x", "runtime_token": "forward-input:x"}],
        "forward_outputs": [],
        "backward_inputs": [{
            "placeholder": "saved",
            "global_forward_matches": [{
                "identity_mode": "EXACT_PYTHON_OBJECT",
                "phase_graph_index": 4,
                "runtime_token": "forward-g4-output:saved",
                "source_node": "saved",
                "value_kind": "FORWARD_OUTPUT",
            }],
        }],
        "gates": {"runtime_identity_only": True},
    }


def test_exact_repeat_runtime_pairs_are_deduplicated() -> None:
    repeated = runtime_pair(3)
    repeated["backward_inputs"][0]["global_forward_matches"] *= 2
    rows = unique_runtime_pairs(
        [runtime_pair(0), repeated],
        {"repeat_loss_exact": True, "repeat_all_parameter_gradient_digest_exact": True},
    )
    assert len(rows) == 1
    assert rows[0][1] == [0, 3]
    assert rows[0][2] is True


def test_repeat_pair_requires_stability_and_fails_closed_on_identity_variation() -> None:
    with pytest.raises(RuntimeError, match="lacks exact observation stability"):
        unique_runtime_pairs([runtime_pair(0), runtime_pair(3)], {})

    changed = deepcopy(runtime_pair(3))
    changed["backward_inputs"][0]["global_forward_matches"].append({
        "identity_mode": "EXACT_PYTHON_OBJECT",
        "phase_graph_index": 1,
        "runtime_token": "forward-g1-output:other",
        "source_node": "other",
        "value_kind": "FORWARD_OUTPUT",
    })
    rows = unique_runtime_pairs(
        [runtime_pair(0), changed],
        {"repeat_loss_exact": True},
    )
    assert len(rows) == 1
    assert rows[0][1] == [0, 3]
    assert rows[0][2] is False
