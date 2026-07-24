from __future__ import annotations

import importlib.util
from pathlib import Path

from datasets import Dataset


def load_phase0_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "phase0_grpo_train.py"
    spec = importlib.util.spec_from_file_location("phase0_grpo_train_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_dataset_applies_offset(monkeypatch):
    module = load_phase0_module()
    source = Dataset.from_list(
        [{"question": f"question-{index}", "answer": f"#### {index}"} for index in range(12)]
    )
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: source)
    dataset, name = module.prepare_dataset(
        {
            "dataset": {
                "name": "fixture",
                "config": "main",
                "split": "train",
                "offset": 4,
                "max_prompts": 3,
            }
        }
    )
    assert len(dataset) == 3
    assert "question-4" in dataset[0]["prompt"]
    assert "question-6" in dataset[2]["prompt"]
    assert name.endswith("[4:7]")


def test_prepare_dataset_fallback_preserves_global_indices(monkeypatch):
    module = load_phase0_module()

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(module, "load_dataset", fail)
    dataset, name = module.prepare_dataset(
        {
            "dataset": {
                "name": "missing",
                "offset": 8,
                "max_prompts": 2,
                "fallback_builtin": True,
            }
        }
    )
    assert len(dataset) == 2
    assert "starts with 15 items" in dataset[0]["prompt"]
    assert name == "forkcert_builtin_arithmetic_fallback[8:10]"
