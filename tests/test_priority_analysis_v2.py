from pathlib import Path

from scripts.verify_priority_analysis_v2 import verify


ROOT = Path(__file__).resolve().parents[1]


def test_saved_priority_analyses_verify():
    result = verify(ROOT / "results/property/result_analysis_v2")
    assert result["status"] == "VERIFIED", result["errors"]
    assert result["checks"]["all_selective_complement_checks"] is True
