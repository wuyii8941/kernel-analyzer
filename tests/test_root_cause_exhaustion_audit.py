from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exhaustion_audit_is_reproducible_and_fail_closed(tmp_path) -> None:
    import os
    subprocess.run([sys.executable, "scripts/build_root_cause_exhaustion_audit.py"], cwd=ROOT, check=True,
                   env={**os.environ, "KA_OUTPUT_DIR": str(tmp_path)})
    result = json.loads((tmp_path / "exhaustion_audit.json").read_text())
    assert result["audited_row_count"] == 54
    assert result["active_problem_group_count"] == 52
    assert result["negative_control_count"] == 2
    ledger = json.loads(
        (ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json").read_text()
    )
    assert {row["problem_group"] for row in result["rows"]} == {
        row["problem_group"] for row in ledger["rows"]
    }
    assert result["status"] == "NO_ADDITIONAL_ROOT_CAUSE_CONCLUSION_FROM_RETAINED_EVIDENCE"
    assert all(row["further_offline_inference"] is False for row in result["rows"])
    assert all(row["missing_observation"] for row in result["rows"])
