import numpy as np
import pytest
import torch

from kernel_analyzer.optimizer_implementation_capture import (
    OptimizerImplementationCapture,
    fixed_suite_total_rms,
    original_coordinate_row,
)


def test_original_coordinate_statistics_and_fixed_suite_rms():
    row = original_coordinate_row(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
    assert row["effect_energy"] == 1.0
    assert row["repair_energy"] == 4.0
    assert row["effect_repair_inner_product"] == 2.0
    assert fixed_suite_total_rms([row] * 32) == pytest.approx(0.5)


def test_capture_uses_one_shared_profile_for_all_optimizer_stages():
    capture = OptimizerImplementationCapture([f"state-{index}" for index in range(32)])
    for index in range(32):
        repair = torch.arange(1, 9, dtype=torch.float32) + index
        candidate = repair * 1.01
        capture.append({stage: (candidate, repair) for stage in capture.rows})
    rows, stages = capture.finish()
    assert set(rows) == {"GRADIENT_INPUT", "MOMENT1_STATE", "MOMENT2_STATE", "PARAMETER_WRITE"}
    assert all(len(value) == 32 for value in rows.values())
    assert stages["PARAMETER_WRITE"]["EXACT"]["profile"]["suite"]["repair_aligned_effect"] == pytest.approx(0.01, abs=1e-7)


def test_capture_rejects_missing_stage_and_changed_coordinates():
    capture = OptimizerImplementationCapture([f"state-{index}" for index in range(32)])
    with pytest.raises(ValueError, match="all optimizer stages"):
        capture.append({})
    pair = (torch.ones(8), torch.ones(8))
    capture.append({stage: pair for stage in capture.rows})
    bad = {stage: pair for stage in capture.rows}
    bad["PARAMETER_WRITE"] = (torch.ones(7), torch.ones(7))
    with pytest.raises(RuntimeError, match="coordinate count changed"):
        capture.append(bad)
