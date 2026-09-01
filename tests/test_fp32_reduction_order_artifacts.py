import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/fp32_reduction_order_v1"


def test_protocol_was_frozen_before_the_recorded_result() -> None:
    protocol = json.loads((BASE / "protocol.json").read_text())
    result = json.loads((BASE / "profile.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_EMPIRICAL_RESULTS"
    assert result["protocol_sha256"] == hashlib.sha256((BASE / "protocol.json").read_bytes()).hexdigest()
    assert result["state_ids"][:16] == result["split"]["calibration"]
    assert result["state_ids"][16:] == result["split"]["confirmation"]


def test_same_fp32_order_experiment_keeps_every_declared_invariant() -> None:
    result = json.loads((BASE / "profile.json").read_text())
    assert result["status"] == "COMPLETE"
    assert result["dtype"] == "FP32"
    assert result["tf32_allowed"] is False
    assert all(result["invariants"].values())
    assert result["profiles"]["LOCAL"]["status"] == "EXACT_ZERO_EFFECT"
    assert all(row["candidate_minus_repair_l2"] > 0.0 for row in result["diagnostics"])


def test_declared_order_pair_is_a_complete_negative_not_a_long_run_case() -> None:
    result = json.loads((BASE / "profile.json").read_text())
    decision = result["primary_update_decision"]
    assert decision["status"] == "SOURCE_PREDICTOR_NOT_CONFIRMED"
    assert decision["confirmed_update_effects"] == []
    assert all(not row["confirmed"] for row in decision["effects"].values())
    assert "not a long-run" in result["claim_boundary"]
