from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "property" / "training_bias_profile_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_five_case_empirical_artifacts_are_complete_and_bound_to_manifest() -> None:
    manifest = json.loads((RESULT / "five_case_manifest.json").read_text())
    summary = json.loads((RESULT / "five_case_summary.json").read_text())

    assert manifest["status"] == "COMPLETE"
    assert manifest["numeric_rules_unchanged_after_result_reveal"] is True
    assert manifest["claim_scope"] == "FROZEN_INPUT_WINDOW_SUITE_AT_ONE_CHECKPOINT"
    assert summary["status"] == "COMPLETE"
    assert summary["claim_scope"] == manifest["claim_scope"]
    assert summary["multiplicity"]["primary_tests"] == 15
    assert summary["multiplicity"]["explanation_tests"] == 30

    expected_cases = {
        "liger": ["residual_direction"],
        "phi": ["repair_aligned"],
        "qwen_lmhead": [],
        "qwen_vproj": ["repair_aligned"],
        "mamba_inproj": ["repair_aligned"],
    }
    assert manifest["primary_update_results"] == expected_cases
    assert set(summary["cases"]) == set(expected_cases)
    for case, branches in expected_cases.items():
        assert summary["cases"][case]["confirmed_update_branches"] == branches

    for name, expected_digest in manifest["raw_files"].items():
        path = RESULT / "five_case_raw" / name
        assert _sha256(path) == expected_digest
        payload = json.loads(path.read_text())
        assert payload["status"] == "COMPLETE"
        assert len(payload["state_ids"]) == 32
        assert payload["determinism"]["all_exact"] is True
        assert set(payload["stages"]) == {
            "LOCAL", "PARAMETER_GRADIENT", "ADAMW_UPDATE"
        }

    assert _sha256(RESULT / manifest["summary_file"]["path"]) == manifest[
        "summary_file"
    ]["sha256"]
    for name, expected_digest in manifest["protocol_files"].items():
        assert _sha256(RESULT / name) == expected_digest
