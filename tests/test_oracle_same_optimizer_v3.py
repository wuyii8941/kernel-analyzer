import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json"
)


def test_same_optimizer_oracle_keeps_full_result_blind_sample():
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE_15_ROW_ADAMW_COMPARISON"
    assert payload["cohort"]["rows"] == 15
    assert payload["cohort"]["result_blind_sample_rows"] == 12
    assert {row["optimizer"] for row in payload["rows"]} == {
        "AdamW(beta1=0.9,beta2=0.95,eps=1e-8,weight_decay=0)"
    }
    rows = {row["case_id"]: row for row in payload["rows"]}
    assert "multishape-backward-cell-0543" in rows
    assert rows["multishape-backward-cell-0543"]["adamw_local_persistent"] is True


def test_same_optimizer_oracle_does_not_reuse_historical_sgd_labels():
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = {row["case_id"]: row for row in payload["rows"]}
    assert rows["qwen_seq128_lmhead_dx"]["historical_sgd_source_persistent"] is True
    assert rows["qwen_seq128_lmhead_dx"]["adamw_local_persistent"] is False
    assert payload["cohort"]["historical_sgd_rows_remaining_positive_under_adamw"] == 2
