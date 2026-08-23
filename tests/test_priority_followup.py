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


def test_general_mechanism_map_is_measured_and_fail_closed():
    payload = json.loads((BASE / "general_mechanism_map_v1.json").read_text())
    assert payload["status"] == "COMPLETE_MEASURED_MECHANISM_MAP_WITH_ABSTENTIONS"
    assert len(payload["headline_three_stage"]) == 3
    assert payload["background_feedback_audit"]["sampled_rows"] == 12
    assert payload["background_feedback_audit"]["local_diffusive_feedback_persistent_rows"] == 11
    phi = next(
        row for row in payload["response_even_odd"]
        if row["case_id"] == "phi4_seq64_lmhead_dx"
    )
    assert phi["status"] == "UNRESOLVED_EXACT_NEGATIVE_ERROR_NOT_REPRESENTABLE"
    assert phi["approximate_even_odd_not_used"] is True


def test_generic_predictor_evaluation_abstains_without_complete_inputs():
    payload = json.loads((BASE / "joint_predictor_evaluation_v1.json").read_text())
    assert payload["status"] == "COMPLETE_EVALUATION_ALL_ABSTAIN_INPUT_INCOMPLETE"
    assert payload["fully_eligible_cases"] == 0
    assert payload["heldout_confirmation"] == "NOT_APPLICABLE_NO_FROZEN_SCORE_EMITTED"
    assert all(row["verdict"].startswith("ABSTAIN_") for row in payload["rows"])
    assert all("source_measurement" in row for row in payload["rows"])


def test_liger_raw_recapture_was_summarized_before_cache_cleanup():
    payload = json.loads((BASE / "liger_three_stage_raw_recapture_summary.json").read_text())
    assert payload["status"] == "COMPLETE"
    assert payload["raw_vector_capture"]["state_count"] == 32
    assert payload["capture_provenance"]["raw_vectors_retained"] is False
    assert payload["capture_provenance"]["raw_vectors_verified_before_cleanup"] is True
    assert all(len(rows) == 32 for rows in payload["raw_vector_capture"]["layers"].values())


def test_current_status_uses_the_three_case_headline_and_bounded_oracle():
    payload = json.loads((BASE / "current_status.json").read_text())
    assert payload["headline"]["operator_source_or_transport_persistence_cases"] == 3
    assert len(payload["headline"]["case_ids"]) == 3
    assert payload["oracle"]["evaluation_rows"] == 14
    assert payload["oracle"]["positive_rows"] == 3
    assert payload["oracle"]["control_rows"] == 11
    assert payload["mechanism_map"]["universal_property_claimed"] is False
