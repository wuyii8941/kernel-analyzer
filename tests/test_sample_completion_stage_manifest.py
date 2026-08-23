import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage_manifest_does_not_promote_legacy_data():
    result = json.loads((ROOT / "results/property/sample_completion_v1/stage_manifest.json").read_text())
    assert result["counts"]["roster_units"] == 24
    assert result["counts"]["uniform_32_complete"] == 0
    assert all(row["scientific_label"] is None for row in result["stages"])
