from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exhaustion_audit_is_reproducible_and_fail_closed() -> None:
    subprocess.run([sys.executable, "scripts/build_root_cause_exhaustion_audit.py"], cwd=ROOT, check=True)
    result = json.loads((ROOT / "results/property/root_cause_closure_v1/exhaustion_audit.json").read_text())
    assert result["scientific_problem_group_count"] == 7
    assert result["status"] == "NO_ADDITIONAL_ROOT_CAUSE_CONCLUSION_FROM_RETAINED_EVIDENCE"
    assert all(row["further_offline_inference"] is False for row in result["rows"])
    assert all(row["missing_observation"] for row in result["rows"])
