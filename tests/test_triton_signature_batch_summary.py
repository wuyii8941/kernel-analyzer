import hashlib
import json
from pathlib import Path

from scripts.summarize_triton_signature_batches import summarize


def manifest(tmp_path):
    return {"groups": [{
        "group_id": "g", "campaign_output": str(tmp_path / "out"),
        "source_campaigns": [{
            "case_id": "c", "task_id": "backward:1:out",
            "operator_family": "ELEMENTWISE", "signature_key": ["ELEMENTWISE"],
            "source_release": str(tmp_path / "source-release"),
        }],
    }]}


def test_summary_reads_per_position_analysis_from_shared_capture(tmp_path):
    payload = manifest(tmp_path)
    task_id = "backward:1:out"
    run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
    run = tmp_path / "out" / "runs" / run_id
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"status":"VALID"}')
    (run / "analysis.json").write_text(json.dumps({
        "measurement_status": "VALID", "equivalence_decision": "NON_EQUIVALENT",
        "bias_analysis": {"fixed_suite_total_rms": 0.2,
                          "fixed_suite_aligned_ratio_of_sums": -0.1},
    }))
    result = summarize(payload)
    assert result["valid_measurement_count"] == 1
    assert result["rows"][0]["fixed_suite_parameter_write_total_rms"] == 0.2
    assert result["rows"][0]["population_guarantee"] is False
    assert result["rows"][0]["source_release"] == str(tmp_path / "source-release")
    assert result["rows"][0]["analysis_sha256"] == hashlib.sha256(
        (run / "analysis.json").read_bytes()
    ).hexdigest()


def test_timeout_does_not_become_measurement(tmp_path):
    payload = manifest(tmp_path)
    output = Path(payload["groups"][0]["campaign_output"])
    output.mkdir(parents=True)
    (output / "execution_timeout.json").write_text("{}")
    result = summarize(payload)
    assert result["rows"][0]["execution_status"] == "EXECUTION_TIMEOUT_NOT_MEASURED"
    assert result["valid_measurement_count"] == 0
