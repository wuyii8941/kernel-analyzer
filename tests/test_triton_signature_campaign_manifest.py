from pathlib import Path

from scripts.build_triton_signature_campaign_manifest import build
from scripts.build_triton_signature_campaign_manifest import _valid_prior_tasks


def row(task, family, signature, *, status="READY_FOR_MEASUREMENT", method="AOT_REPLAY"):
    return {
        "release": "/data1/tzh/release",
        "task_id": task,
        "operator_family": family,
        "implementation_kind": "TRITON",
        "phase": "BACKWARD",
        "symbol": signature.replace("{N}", "1"),
        "symbol_signature": signature,
        "support_status": status,
        "carrier": "weight",
        "reference_candidates": [{"reference_method": method}],
    }


def queue_row(task, family, signature, method="AOT_REPLAY"):
    value = row(task, family, signature, method=method)
    value["wave"] = "NEW_SIGNATURE_SECOND"
    return value


def test_selects_one_unmeasured_signature_and_not_one_per_position():
    catalog = {"positions": [
        row("old", "SOFTMAX", "softmax_{N}", status="VALID_MEASUREMENT_COMPLETED"),
        row("a", "ELEMENTWISE", "pointwise_{N}"),
        row("b", "ELEMENTWISE", "pointwise_{N}"),
        row("c", "ELEMENTWISE", "different_{N}"),
    ]}
    queue = {"rows": [
        queue_row("old2", "SOFTMAX", "softmax_{N}"),
        queue_row("a", "ELEMENTWISE", "pointwise_{N}"),
        queue_row("b", "ELEMENTWISE", "pointwise_{N}"),
        queue_row("c", "ELEMENTWISE", "different_{N}"),
    ]}
    runtime = {"/data1/tzh/release": {
        "architecture": "test", "model": "/data1/tzh/model",
        "input_bank": "/data1/tzh/input.json", "allow_graph_breaks": False,
    }}
    result = build(catalog, queue, runtime, output_root=Path("/data1/tzh/out"))
    assert [item["task_id"] for item in result["campaigns"]] == ["a", "c"]
    assert result["selected_operator_family_counts"] == {"ELEMENTWISE": 2}
    assert result["selection_uses_numerical_outcomes"] is False


def test_prior_valid_task_excludes_its_signature():
    catalog = {"positions": [
        row("prior", "FUSED_MIXED", "fused_{N}"),
        row("new", "FUSED_MIXED", "fused_{N}"),
    ]}
    queue = {"rows": [queue_row("new", "FUSED_MIXED", "fused_{N}")]}
    runtime = {"/data1/tzh/release": {
        "architecture": "test", "model": "/data1/tzh/model",
        "input_bank": "/data1/tzh/input.json", "allow_graph_breaks": False,
    }}
    result = build(
        catalog,
        queue,
        runtime,
        output_root=Path("/data1/tzh/out"),
        prior_valid_tasks={("/data1/tzh/release", "prior")},
    )
    assert result["selected_campaign_count"] == 0


def test_non_triton_and_missing_runtime_do_not_become_campaigns():
    ordinary = queue_row("ordinary", "LINEAR", "mm")
    ordinary["implementation_kind"] = "EXTERN"
    missing = queue_row("missing", "ELEMENTWISE", "pointwise_{N}")
    missing["release"] = "/data1/tzh/missing-release"
    result = build(
        {"positions": []},
        {"rows": [ordinary, missing]},
        {},
        output_root=Path("/data1/tzh/out"),
    )
    assert result["selected_campaign_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["reason"] == "RUNTIME_CONFIG_NOT_UNIQUELY_RECOVERED"


def test_phase_is_recovered_from_task_id_when_queue_omits_it():
    forward = queue_row("forward:1:out_ptr0", "ELEMENTWISE", "same_{N}")
    backward = queue_row("backward:1:out_ptr0", "ELEMENTWISE", "same_{N}")
    forward.pop("phase")
    backward.pop("phase")
    runtime = {"/data1/tzh/release": {
        "architecture": "test", "model": "/data1/tzh/model",
        "input_bank": "/data1/tzh/input.json", "allow_graph_breaks": False,
    }}
    result = build(
        {"positions": []}, {"rows": [forward, backward]}, runtime,
        output_root=Path("/data1/tzh/out"),
    )
    assert [item["phase"] for item in result["campaigns"]] == ["FORWARD", "BACKWARD"]


def test_valid_retry_is_mapped_back_to_original_release(tmp_path):
    output = tmp_path / "result"
    task_id = "backward:7:out_ptr0"
    run_id = __import__("hashlib").sha256(task_id.encode()).hexdigest()[:20]
    status = output / "runs" / run_id / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"status":"VALID"}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps({"campaigns": [{
        "release": "/data1/tzh/repaired",
        "original_release": "/data1/tzh/original",
        "task_id": task_id,
        "adapter": "GENERIC_COVERAGE",
        "campaign_output": str(output),
    }]}))
    assert _valid_prior_tasks([manifest]) == {("/data1/tzh/original", task_id)}
