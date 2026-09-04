from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phi_loss_direction_stress_is_complete_and_bounded() -> None:
    raw = json.loads(
        (ROOT / "results/property/loss_direction_stress_v1/phi_lmhead.json").read_text()
    )
    summary = json.loads(
        (ROOT / "results/property/loss_direction_stress_v1/phi_lmhead_summary.json").read_text()
    )
    assert raw["status"] == "COMPLETE"
    assert raw["evaluation_state_count"] == 32
    assert len(raw["random_seeds"]) == 4
    assert summary["status"] == "COMPLETE"
    assert len(summary["rows"]) == 4
    assert "no straight-line perturbation" in summary["interpretation"]
