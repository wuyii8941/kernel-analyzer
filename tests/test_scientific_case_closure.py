import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scientific_case_closure_is_explicit_and_deduplicated(tmp_path):
    import os
    subprocess.run([sys.executable, "scripts/build_scientific_case_closure.py"], cwd=ROOT, check=True,
                   env={**os.environ, "KA_OUTPUT_DIR": str(tmp_path)})
    data = json.loads((tmp_path / "scientific_case_closure.json").read_text())
    rows = data["rows"]
    assert len(rows) == len({row["problem_group"] for row in rows}) == 9
    assert data["summary"]["scientific_problem_groups"] == 7
    assert data["summary"]["coverage_collections"] == 2
    assert data["summary"]["end_to_end_causal_chains"] == 1
    assert data["summary"]["closed_local_arithmetic_subcases"] == 2
    assert data["summary"]["all_groups_have_unique_root"] is False
    for row in rows:
        assert all(row[key] for key in ("implementation_boundary", "numerical_source", "bias_formation", "decisive_interventions", "remaining_limit", "next_root_cause_test", "evidence"))
