from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "r1_preregister.py"
    spec = importlib.util.spec_from_file_location("r1_preregister_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_predicted_rate_uses_strict_margin_less_than_delta():
    module = load_module()
    assert module.predicted_rate([0.1, 0.2, 0.3], [0.2]) == 1 / 3


def test_predicted_rate_is_zero_when_clipping_has_no_applicable_decisions():
    module = load_module()
    assert module.predicted_rate([], [0.1, 0.2]) == 0.0
    assert module.predicted_rate([0.1], []) == 0.0


def test_poisson_interval_quantiles_are_ordered():
    module = load_module()
    low = module.poisson_ppf(0.025, 3.758)
    high = module.poisson_ppf(0.975, 3.758)
    assert low == 1
    assert high >= 7
