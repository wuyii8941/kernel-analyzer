import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/direct_persistence_v4"


def test_independent_gemma_heldout_is_complete_negative_for_direct_screen():
    confirmation = json.loads((OUT / "heldout_gemma_confirmation.json").read_text())
    validation = json.loads((OUT / "heldout_gemma_validation.json").read_text())
    row = confirmation["rows"][0]
    assert confirmation["status"] == "COMPLETE_ONE_NEW_IMPL_NEGATIVE_FOR_DIRECT_SCREEN"
    assert validation["status"] == "VALID_ONE_NEW_IMPL_ROW"
    assert row["short_screen"]["verdict"] == "NO_ESCALATION_UNDER_SHORT_SCREEN"
    assert row["confirmation"]["verdict"] == "NO_DETECTED_DIRECT_PERSISTENCE_AFTER_CONFIRMATION"
    assert row["confirmation"]["A32_local"] < 1.01
    assert row["confirmation"]["A32_actual"] > 3.0
    assert row["confirmation"]["feedback_status"] == "FEEDBACK_DOMINATED_CONSEQUENCE"
