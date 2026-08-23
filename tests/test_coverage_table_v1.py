import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_coverage_table_keeps_systematic_and_directed_scopes_separate():
    table = json.loads((ROOT / "results/coverage/coverage_table_v1.json").read_text())
    assert table["model_count_with_actual_artifacts"] == 10
    assert table["systematic_census_model_count"] == 4
    assert table["systematic_census_cell_count"] == 12
    assert len(table["core_rows"]) == 12
    assert len(table["directed_or_heldout_rows"]) == 6
    assert table["uniform_sample_completion"]["current_uniform_cases"] == 0


def test_coverage_totals_match_frozen_audit():
    table = json.loads((ROOT / "results/coverage/coverage_table_v1.json").read_text())
    assert table["totals"] == {
        "eager_invocations": 466419,
        "candidate_invocations": 70171,
        "primary_fb_proof_units": 186807,
        "t1_audited": 1562,
        "t1_pass": 1390,
        "t1_reject": 172,
    }
