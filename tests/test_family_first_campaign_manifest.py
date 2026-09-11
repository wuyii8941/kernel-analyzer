import json

from scripts.build_family_first_campaign_manifest import build, discover_runtime_configs
from scripts.run_family_first_campaigns import command, observed_status


def test_campaign_manifest_uses_only_existing_generic_reference_methods(tmp_path):
    queue = {"rows": [
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "ELEMENTWISE",
         "release": "/r", "task_id": "backward:1:out", "carrier": "weight",
         "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "SOFTMAX",
         "release": "/r", "task_id": "backward:2:out", "carrier": "weight",
         "reference_candidates": [{"family": "SOFTMAX_BACKWARD"}]},
    ]}
    configs = {"/r": {"architecture": "a", "model": "/m", "input_bank": "/b",
                       "source_protocols": ["/p"]}}
    result = build(queue, configs, output_root=tmp_path)
    assert result["selected_operator_families"] == ["ELEMENTWISE"]
    assert result["campaigns"][0]["implementation_kind"] == "TRITON"
    assert result["execution_policy"]["coverage_position_count_is_not_a_success_metric"]
    assert result["skipped_new_family_rows"][0]["reason"] == "GENERIC_REFERENCE_METHOD_UNAVAILABLE"
    result["campaigns"][0]["case_plan"] = "/plan.json"
    freeze = command(result["campaigns"][0], "freeze", device="cuda:0")
    assert "--release" in freeze and "--plans" in freeze


def test_freeze_propagates_segmented_compile_policy(tmp_path):
    campaign = {
        "campaign_output": str(tmp_path), "release": "/r", "case_plan": "/p",
        "runtime": {"architecture": "a", "model": "/m", "input_bank": "/b",
                    "allow_graph_breaks": True},
    }
    assert "--allow-graph-breaks" in command(campaign, "freeze", device="cuda:0")


def test_specialized_family_campaign_keeps_declared_plan_and_runtime_inputs(tmp_path):
    queue = {"rows": [{
        "wave": "NEW_FAMILY_FIRST", "operator_family": "NORMALIZATION",
        "release": "/r", "task_id": "forward:1:out", "carrier": "weight",
        "implementation_kind": "TRITON",
        "reference_candidates": [{
            "reference_method": "RESIDUAL_RMS_FORWARD_COMMON_INPUT",
            "bound_plan": "/declared/normalization.json",
        }],
    }]}
    configs = {"/r": {"architecture": "deepseek8", "model": "/m",
                       "input_bank": "/b", "source_protocols": ["/p"]}}
    result = build(queue, configs, output_root=tmp_path, include_specialized=True)
    campaign = result["campaigns"][0]
    assert campaign["adapter"] == "RESIDUAL_RMS_FORWARD"
    assert campaign["specialized_plan"] == "/declared/normalization.json"
    campaign["case_plan"] = "/case.json"
    run = command(campaign, "run", device="cuda:2")
    assert "run_residual_rms_forward_capture.py" in run[1]
    assert "--family-plan" in run and "/declared/normalization.json" in run
    assert "--architecture" in run and "deepseek8" in run
    assert "--states" in run and "32" in run


def test_runtime_discovery_does_not_merge_different_compile_policies(tmp_path):
    import json
    first = tmp_path / "a" / "protocol.json"
    second = tmp_path / "b" / "protocol.json"
    first.parent.mkdir(); second.parent.mkdir()
    base = {"release": "/r", "architecture": "a", "model": "/m", "input_bank": "/b"}
    first.write_text(json.dumps(base | {"allow_graph_breaks": False}))
    second.write_text(json.dumps(base | {"allow_graph_breaks": True}))
    assert "/r" not in discover_runtime_configs([first, second])


def test_orchestration_distinguishes_command_return_from_measurement(tmp_path):
    campaign = {"campaign_output": str(tmp_path), "task_id": "backward:1:out"}
    assert observed_status(campaign, "run", 0) == "MEASUREMENT_INCOMPLETE_OR_RUNNING"
    import hashlib, json
    run = tmp_path / "runs" / hashlib.sha256(campaign["task_id"].encode()).hexdigest()[:20]
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "EXECUTION_FAILED"}))
    assert observed_status(campaign, "run", 0) == "MEASUREMENT_EXECUTION_FAILED"


def test_specialized_summary_reads_family_completion_artifact(tmp_path):
    from scripts.summarize_family_first_campaigns import summarize

    output = tmp_path / "campaign"
    output.mkdir()
    (output / "completion_verification.json").write_text(json.dumps({
        "records": [{
            "task_id": "forward:1:out", "status": "RECORDED_MEASUREMENT_CHECKED",
            "analysis": {"measurement_status": "VALID",
                         "equivalence_decision": "EQUIVALENT",
                         "bias_analysis": {"fixed_suite_total_rms": 0.0}},
        }],
    }))
    result = summarize({"campaigns": [{
        "adapter": "RESIDUAL_RMS_FORWARD", "campaign_output": str(output),
        "operator_family": "NORMALIZATION", "task_id": "forward:1:out",
        "case": {"case_id": "case", "reference_method": "RESIDUAL_RMS_FORWARD_COMMON_INPUT"},
    }]})
    assert result["valid_measurement_count"] == 1
    assert result["rows"][0]["equivalence_decision"] == "EQUIVALENT"
