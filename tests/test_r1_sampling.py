from __future__ import annotations

import importlib.util
from pathlib import Path


def load_merge_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "r1_merge_sampling.py"
    spec = importlib.util.spec_from_file_location("r1_merge_sampling_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cluster_bootstrap_empty_input_is_zero_interval():
    module = load_merge_module()
    assert module.cluster_bootstrap_rate([], "fork", draws=10) == [0.0, 0.0]


def test_keyed_uses_case_position_and_token():
    module = load_merge_module()
    payload = {"rows": [{"case_id": "c", "token_index": 2, "token_id": 3}]}
    assert set(module.keyed(payload)) == {("c", 2, 3)}
