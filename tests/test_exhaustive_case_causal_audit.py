import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_source_record_has_four_question_review():
    data = json.loads((ROOT / "results/property/case_causal_audit_v1/exhaustive_source_records.json").read_text())
    assert data["record_count"] == 866
    assert data["counts_by_source"] == {
        "MEASURED_POSITION": 551, "HISTORICAL_MATRIX_ROW": 301,
        "MAINLINE_ROLE_RECORD": 6, "LEGACY_CASE_REAUDIT": 8,
    }
    required = {"code_location", "numerical_error", "why_nonzero_bias", "intervention", "assessment"}
    assert all(required <= row.keys() for row in data["rows"])
    assert not any("PENDING" in row["assessment"] for row in data["rows"])


def test_measurement_rows_do_not_claim_root_from_energy():
    data = json.loads((ROOT / "results/property/case_causal_audit_v1/exhaustive_source_records.json").read_text())
    measured = [row for row in data["rows"] if row["source_kind"] == "MEASURED_POSITION"]
    assert len(measured) == 551
    assert all("ROOT" in row["assessment"] or "IDENTITY" in row["assessment"] or "SUMMARY" in row["assessment"]
               for row in measured)
