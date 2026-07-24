from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "p2_portfolio.py"
    spec = importlib.util.spec_from_file_location("p2_portfolio_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_counts_mutation_families_and_recall():
    module = load_module()
    rows = [
        {"label": True, "family": "a"},
        {"label": True, "family": "b"},
        {"label": True, "family": "b"},
        {"label": False, "family": None},
    ]
    result = module.metrics(rows, {0, 1, 3})
    assert result["mutation_true_positives"] == 2
    assert result["mutation_family_coverage"] == 2
    assert result["token_recall"] == 2 / 3
    assert result["precision"] == 2 / 3


def test_quantile_handles_empty_and_boundaries():
    module = load_module()
    assert module.quantile([], 0.5) == 0.0
    assert module.quantile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert module.quantile([3.0, 1.0, 2.0], 0.99) == 3.0
