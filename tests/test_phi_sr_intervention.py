import json
from pathlib import Path

import pytest
import torch

from scripts.run_phi_mm_sr_intervention import norm_match_update


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/property/joint_bias_formation_v1/oracle_repair_v3/phi_sr_update_norm_matched.json"
)
PROTOCOL = (
    ROOT
    / "results/property/joint_bias_formation_v1/oracle_repair_v3/protocol.json"
)


def test_norm_match_preserves_direction_and_matches_energy():
    value = torch.tensor([3.0, 4.0])
    matched = norm_match_update(value, 10.0)
    assert torch.linalg.vector_norm(matched).item() == pytest.approx(10.0)
    assert torch.dot(value, matched).item() > 0.0
    assert matched[0].item() / matched[1].item() == pytest.approx(3.0 / 4.0)


def test_norm_match_rejects_zero_to_nonzero():
    with pytest.raises(ValueError):
        norm_match_update(torch.zeros(3), 1.0)


def test_phi_sr_result_declares_stateless_sgd_scope():
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema"] == "kernel-analyzer-phi-real-sr-intervention-v2"
    assert payload["steps"] == 16
    assert payload["update_mapping"] == {
        "name": "STATELESS_SGD_FP32_MASTER",
        "learning_rate": 0.001,
        "moment_state": "NONE",
    }
    assert "not the 32-step zero-moment AdamW" in payload["claim_boundary"]


def test_oracle_protocol_does_not_attach_phi_sr_to_adamw_result():
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["optimizer"]["name"] == "AdamW"
    intervention = payload["phi_matched_intervention"]
    assert intervention["steps"] == 16
    assert intervention["update_mapping"].startswith("STATELESS_SGD_FP32_MASTER")
    assert "cannot be used as its mechanism intervention" in intervention["scope_note"]
