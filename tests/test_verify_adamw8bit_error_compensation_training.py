from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interval_is_centered_on_mean():
    path = ROOT / "scripts/verify_adamw8bit_error_compensation_training.py"
    spec = importlib.util.spec_from_file_location("verify_compensation_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    values = [0.01, 0.02, 0.03, 0.04]
    low, high = module.interval(values)
    assert abs((low + high) / 2 - 0.025) < 1e-12
    assert low < high


def test_recorded_float_comparison_allows_machine_roundoff_only():
    path = ROOT / "scripts/verify_adamw8bit_error_compensation_training.py"
    spec = importlib.util.spec_from_file_location("verify_compensation_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.equivalent_recorded_value(
        [0.016856271755534905, 0.03283993901626418],
        [0.016856271755534908, 0.03283993901626418],
    )
    assert not module.equivalent_recorded_value(0.016, 0.017)
    assert not module.equivalent_recorded_value("MATERIAL_IMPROVEMENT", "NOT_CONFIRMED")
