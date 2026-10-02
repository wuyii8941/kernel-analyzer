import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_signature_candidate_aligned_screen_is_search_only() -> None:
    subprocess.run(
        [sys.executable, "scripts/analyze_signature_candidate_aligned_screen.py"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "scripts"},
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        (ROOT / "results/property/case_causal_audit_v1/signature_candidate_aligned_screen.json").read_text()
    )
    assert report["status"] == "SCREEN_ONLY_NOT_A_ROOT_CAUSE_COUNT"
    assert report["record_count"] == 31
    assert report["all_records_include_possible_upstream_differences"] is True
    assert all(item["promotion_status"] == "SCREEN_ONLY_REGION_SUBSTITUTION" for item in report["records"])
    assert any(
        item["candidate_family"] == "embedding_dense_backward"
        and item["aligned_negative_count"] == item["state_count"]
        for item in report["records"]
    )
