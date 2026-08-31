import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/training_bias_profile_v2"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_batch_one_retains_every_frozen_case() -> None:
    protocol = load(BASE / "prospective_batch_1_protocol.json")
    summary = load(BASE / "prospective_batch_1/summary.json")
    expected = {
        f"{row['model']}_seq{row['sequence_length']}_{row['task_id'].replace(':', '_')}"
        for row in protocol["cases"]
    }
    assert set(summary["cases"]) == expected
    assert summary["multiplicity"]["primary_tests"] == 12
    assert summary["multiplicity"]["abstentions_retained_as_p_one"] is True
    assert sum(
        row["primary_update_result"] == "ABSTAIN"
        for row in summary["cases"].values()
    ) == 2


def test_new_deepseek_normalization_effect_is_bounded_and_corrected() -> None:
    summary = load(BASE / "prospective_batch_1/summary.json")
    row = summary["cases"]["deepseek8b_seq256_backward_1714_in_out_ptr0"]
    aligned = row["stages"]["ADAMW_UPDATE"]["branches"]["repair_aligned"]
    assert row["confirmed_update_branches"] == ["repair_aligned"]
    assert aligned["estimate"] < -0.13
    assert aligned["confidence_interval_95"][1] < -0.11
    assert aligned["holm_adjusted_p"] <= 0.01


def test_phi_loss_backward_is_a_complete_negative_not_an_abstention() -> None:
    summary = load(BASE / "prospective_batch_1/summary.json")
    row = summary["cases"]["phi4_seq64_backward_495_out_ptr1"]
    assert row["primary_update_result"] == "NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL"
    assert row["confirmed_update_branches"] == []


def test_attention_projection_repeats_the_relative_update_form() -> None:
    summary = load(BASE / "prospective_batch_2/summary.json")
    row = summary["cases"]["deepseek8b_seq128_backward_1256_out_ptr0"]
    aligned = row["stages"]["ADAMW_UPDATE"]["branches"]["repair_aligned"]
    assert row["confirmed_update_branches"] == ["repair_aligned"]
    assert aligned["estimate"] < -0.10
    assert aligned["confidence_interval_95"][1] < -0.09
    assert aligned["holm_adjusted_p"] <= 0.01


def test_new_confirmed_cases_have_bounded_loss_consequences() -> None:
    paths = (
        BASE / "prospective_batch_1/consequence/deepseek_norm_4096.json",
        BASE / "prospective_batch_2/consequence/deepseek_attn_projection_4096.json",
    )
    for path in paths:
        row = load(path)
        assert row["status"] == "COMPLETE_PAIRED_LOSS_SPLIT"
        assert row["planned_horizon_steps"] == 4096
        assert row["step_count"] == 1
        assert row["loss_audit"]["any_period_split"] is True
        assert abs(row["loss_audit"]["final_gap"]) > row["loss_audit"]["tolerance"]
        assert "not long-horizon persistence" in row["claim_boundary"]
