import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def test_oracle_v2_has_three_headline_positives():
    payload = json.loads(
        (BASE / "oracle_baselines/frozen_evaluation_v2/comparison_v2.json").read_text()
    )
    assert payload["status"] == "COMPLETE_FROZEN_14_ROW_COMPARISON_3_POSITIVE_HEADLINES"
    assert payload["cohort"]["rows"] == 14
    assert payload["cohort"]["positive_rows"] == 3
    assert payload["comparisons"]["prefix16_effective_update_persistence_oracle"]["auroc"] == 1.0


def test_random_null_covers_all_carriers_and_seeds():
    payload = json.loads(
        (BASE / "carrier_distribution/random_null_v2/distribution.json").read_text()
    )
    assert payload["carrier_count"] == 12
    assert len(payload["random_null_summary"]) == 5
    assert all(len(row["random_nulls"]) == 5 for row in payload["rows"])


def test_adamw_mapping_is_complete():
    payload = json.loads((BASE / "phi_three_stage_adamw.json").read_text())
    assert payload["status"] == "COMPLETE_ORDERED_32_STATE_COMMON_STATE_ADAMW"
    curve = payload["stages"]["effective_update_error"]["coherence_curve"]
    assert curve[-1]["horizon"] == 32


def test_random_null_loss_is_complete():
    payload = json.loads((BASE / "four_scale_arms/random_null_loss.json").read_text())
    assert payload["status"] == "COMPLETE_UNSEEN_FP32_EVALUATION"
    assert payload["random_null_A"] > 0
