import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/direct_persistence_v4"


def load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_v4_retrospective_keeps_candidate_out_of_confirmed_denominator():
    metrics = load("retrospective_metrics.json")
    assert metrics["nominal_discovery_view"]["positives"] == 3
    assert metrics["confirmed_only_view"]["positives"] == 2
    assert metrics["confirmed_only_view"]["rows"] == 14
    assert metrics["candidate"]["status"] == "UNRESOLVED_CANDIDATE"
    assert metrics["candidate"]["case_id"] == "multishape-backward-cell-0543"


def test_v4_holm_uses_all_twelve_discovery_rows():
    metrics = load("retrospective_metrics.json")
    families = metrics["multiple_testing"]["families"]
    assert len(families["PREDECLARED_3"]) == 3
    assert len(families["RESULT_BLIND_DISCOVERY_12"]) == 12
    assert "multishape-backward-cell-0543" in families["RESULT_BLIND_DISCOVERY_12"]

    cohort = load("cohort.json")
    candidate = next(row for row in cohort["rows"] if row["case_id"] == "multishape-backward-cell-0543")
    assert candidate["holm"]["RESULT_BLIND_DISCOVERY_12"]["family_size"] == 12
    assert candidate["holm"]["RESULT_BLIND_DISCOVERY_12"]["holm_reject_alpha_0_05"] is False


def test_v4_multiplicity_artifact_has_three_frozen_families():
    multiplicity = load("multiplicity.json")
    assert multiplicity["primary_method"] == "Holm family-wise correction"
    assert len(multiplicity["families"]["PREDECLARED_3"]["case_ids"]) == 3
    assert len(multiplicity["families"]["RESULT_BLIND_DISCOVERY_12"]["case_ids"]) == 12
    assert len(multiplicity["families"]["ALL_15_SENSITIVITY"]["case_ids"]) == 15
    assert multiplicity["candidate_policy"]["never_relabel_as_negative"] is True


def test_v4_contribution_shares_are_signed_and_sum_to_one():
    payload = load("contribution_table.json")
    assert len(payload["rows"]) == 3
    for row in payload["rows"]:
        assert row["status"] == "COMPLETE_DERIVED_RESULTANT_ATTRIBUTION"
        shares = row["signed_projection_share"]
        assert abs(sum(shares.values()) - 1.0) < 1e-8
        assert row["raw_vector_evidence"]["per_step_cross_gram_recomputable"] is False


def test_v4_protocol_is_frozen_before_heldout_and_never_safe():
    protocol = load("protocol.json")
    assert protocol["status"] == "FROZEN_BEFORE_HELDOUT_REVEAL"
    assert protocol["name"] == "Cold-start AdamW Direct Persistence Screen"
    assert protocol["short_screen"]["no_safe_verdict"] is True
    assert protocol["heldout"]["status"] == "COMPLETE_FRESH_POOL_V3_NO_DIRECT_POSITIVE"
    assert protocol["heldout"]["completed_rows"] == 3
    assert protocol["heldout"]["direct_positive_rows"] == 0
    assert protocol["optimizer_state_sensitivity"]["status"] == "COMPLETE_SAME_STATE_ABLATION_AND_NATURAL_PHASE_RESPONSE"
    assert protocol["optimizer"]["moment_reset_every_step"] is False


def test_phi_protocols_remain_separate_in_v4_summary():
    summary = load("summary.json")
    assert "stateless-SGD" in summary["phi_sr_scope"]
    assert "AdamW A=1.029" in summary["phi_sr_scope"]
