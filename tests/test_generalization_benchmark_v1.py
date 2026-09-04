from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generalization_benchmark_is_frozen_and_balanced() -> None:
    protocol = json.loads(
        (ROOT / "results/property/generalization_benchmark_v1/protocol.json").read_text()
    )
    assert protocol["status"] == "FROZEN_BEFORE_ANY_BENCHMARK_MEASUREMENT"
    assert len(protocol["cases"]) == 16
    assert protocol["primary_comparison_count"] == 48
    assert len({(row["model"], row["sequence_length"], row["task_id"]) for row in protocol["cases"]}) == 16
    assert {row["model"] for row in protocol["cases"]} == {"qwen", "phi4", "deepseek8b", "mamba"}
