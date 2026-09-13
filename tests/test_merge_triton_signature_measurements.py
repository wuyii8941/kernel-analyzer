import hashlib
import json

import pytest

from scripts.merge_triton_signature_measurements import merge


def test_merge_attaches_only_valid_exact_positions(tmp_path):
    release = tmp_path / "release"
    artifact = tmp_path / "analysis.json"
    artifact.write_text(json.dumps({"measurement_status": "VALID"}))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    inventory = {"records": [
        {"release": str(release), "task_id": "backward:1:x",
         "runtime_measurement_status": "NOT_ASSESSED"},
        {"release": str(release), "task_id": "backward:2:x",
         "runtime_measurement_status": "NOT_ASSESSED"},
    ]}
    summary = {
        "schema": "triton-signature-runtime-batch-summary-v1",
        "rows": [
            {"source_release": str(release), "task_id": "backward:1:x",
             "case_id": "case-1", "measurement_status": "VALID",
             "analysis_artifact": str(artifact), "analysis_sha256": digest},
            {"source_release": str(release), "task_id": "backward:2:x",
             "case_id": "case-2", "measurement_status": "NOT_ASSESSED",
             "analysis_artifact": None, "analysis_sha256": None},
        ],
    }
    result = merge(inventory, [summary], verify_artifacts=True)
    assert result["records"][0]["runtime_measurement_status"] == "VERIFIED"
    assert result["records"][1]["runtime_measurement_status"] == "NOT_ASSESSED"
    assert result["triton_signature_measurement_merge"]["updated_positions"] == 1
    assert result["records"][0]["triton_signature_measurement_evidence"][
        "coverage_only"
    ] is True


def test_merge_rejects_changed_or_absent_evidence(tmp_path):
    release = tmp_path / "release"
    artifact = tmp_path / "analysis.json"
    artifact.write_text("{}")
    inventory = {"records": [{"release": str(release), "task_id": "task"}]}
    row = {
        "source_release": str(release), "task_id": "task", "case_id": "case",
        "measurement_status": "VALID", "analysis_artifact": str(artifact),
        "analysis_sha256": "wrong",
    }
    summary = {"schema": "triton-signature-runtime-batch-summary-v1", "rows": [row]}
    with pytest.raises(ValueError, match="changed"):
        merge(inventory, [summary], verify_artifacts=True)
    row["analysis_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row["task_id"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        merge(inventory, [summary], verify_artifacts=True)


def test_merge_updates_observed_catalog_support_summary(tmp_path):
    release = tmp_path / "release"
    artifact = tmp_path / "analysis.json"
    artifact.write_text("{}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    catalog = {
        "positions": [{
            "release": str(release), "task_id": "task", "operator_family": "REDUCTION",
            "release_task_package_sha256": "package", "support_status": "READY_FOR_MEASUREMENT",
            "runtime_measurement_status": None,
        }],
        "summary": {},
    }
    summary = {"schema": "triton-signature-runtime-batch-summary-v1", "rows": [{
        "source_release": str(release), "task_id": "task", "case_id": "case",
        "measurement_status": "VALID", "analysis_artifact": str(artifact),
        "analysis_sha256": digest,
    }]}
    result = merge(catalog, [summary], verify_artifacts=True)
    assert result["positions"][0]["support_status"] == "VALID_MEASUREMENT_COMPLETED"
    assert result["summary"]["position_support_status_counts"] == {
        "VALID_MEASUREMENT_COMPLETED": 1
    }
