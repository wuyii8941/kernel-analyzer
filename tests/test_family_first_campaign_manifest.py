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


def test_campaign_manifest_can_be_restricted_to_static_unmeasured_frontier(tmp_path):
    queue = {"rows": [
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "ELEMENTWISE",
         "release": "/r", "task_id": "backward:1:out", "carrier": "weight",
         "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "SOFTMAX",
         "release": "/r", "task_id": "backward:2:out", "carrier": "weight",
         "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
    ]}
    configs = {"/r": {"architecture": "a", "model": "/m", "input_bank": "/b",
                       "source_protocols": ["/p"]}}
    result = build(queue, configs, output_root=tmp_path,
                   frontier_families={"ELEMENTWISE"})
    assert result["selected_operator_families"] == ["ELEMENTWISE"]
    assert result["selection_scope"] == "STATIC_UNMEASURED_FRONTIER_FAMILY_ROWS_ONLY"


def test_campaign_manifest_records_separate_release_rebind(tmp_path):
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    queue = {"rows": [{
        "wave": "NEW_FAMILY_FIRST", "operator_family": "ELEMENTWISE",
        "release": "/r", "task_id": "backward:1:out", "carrier": "weight",
        "implementation_kind": "TRITON",
        "reference_candidates": [{"reference_method": "AOT_REPLAY"}],
    }]}
    configs = {"/r": {"architecture": "a", "model": "/m", "input_bank": "/b",
                       "source_protocols": ["/p"]}}
    result = build(queue, configs, output_root=tmp_path / "out",
                   release_overrides={"ELEMENTWISE": str(replacement)})
    campaign = result["campaigns"][0]
    assert campaign["release"] == str(replacement)
    assert campaign["original_release"] == "/r"
    assert campaign["release_override"] is True


def test_campaign_manifest_records_explicit_graph_break_policy(tmp_path):
    queue = {"rows": [{
        "wave": "NEW_FAMILY_FIRST", "operator_family": "ELEMENTWISE",
        "release": "/r", "task_id": "backward:1:out", "carrier": "weight",
        "implementation_kind": "TRITON",
        "reference_candidates": [{"reference_method": "AOT_REPLAY"}],
    }]}
    configs = {"/r": {"architecture": "a", "model": "/m", "input_bank": "/b",
                       "allow_graph_breaks": False, "source_protocols": ["/p"]}}
    result = build(queue, configs, output_root=tmp_path / "out",
                   allow_graph_breaks_families={"ELEMENTWISE"})
    assert result["campaigns"][0]["runtime"]["allow_graph_breaks"] is True


def test_campaign_manifest_can_select_static_fallback_task_after_family_block(tmp_path):
    queue = {"rows": [
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "FUSED_MIXED",
         "release": "/mamba", "task_id": "backward:1:out", "carrier": "a",
         "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
        {"wave": "NEW_FAMILY_FIRST", "operator_family": "FUSED_MIXED",
         "release": "/deepseek", "task_id": "backward:2:out", "carrier": "b",
         "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
    ]}
    configs = {
        "/mamba": {"architecture": "mamba", "model": "/m", "input_bank": "/b",
                    "source_protocols": ["/p"]},
        "/deepseek": {"architecture": "deepseek8", "model": "/d", "input_bank": "/b",
                       "source_protocols": ["/p"]},
    }
    result = build(
        queue, configs, output_root=tmp_path,
        task_overrides={"FUSED_MIXED": "backward:2:out"},
    )
    assert result["campaigns"][0]["task_id"] == "backward:2:out"
    assert result["task_overrides"] == {"FUSED_MIXED": "backward:2:out"}


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


def test_summary_records_queue_timeout_as_invalid_execution(tmp_path):
    from scripts.summarize_family_first_campaigns import summarize

    output = tmp_path / "campaign"
    output.mkdir()
    (output / "execution_timeout.json").write_text(json.dumps({"timeout_seconds": 60}))
    result = summarize({"campaigns": [{
        "adapter": "GENERIC_COVERAGE", "campaign_output": str(output),
        "operator_family": "ELEMENTWISE", "task_id": "backward:1:out",
        "case": {"case_id": "case", "reference_method": "AOT_REPLAY"},
    }]})
    assert result["rows"][0]["execution_status"] == "EXECUTION_TIMEOUT"
    assert result["rows"][0]["measurement_status"] == "NOT_ASSESSED"
    assert result["rows"][0]["failure_reason"] == "EXECUTION_TIMEOUT_AFTER_60_SECONDS"


def test_family_runner_subprocess_environment_includes_repo_import_paths():
    from scripts.run_family_first_campaigns import ROOT, subprocess_environment

    value = subprocess_environment()["PYTHONPATH"].split(":")
    assert str(ROOT / "src") in value
    assert str(ROOT) in value


def test_unmeasured_family_frontier_excludes_measured_and_preserves_blocked():
    from scripts.build_unmeasured_triton_family_frontier import build

    catalogue = {"positions": [
        {"operator_family": "SOFTMAX", "support_status": "VALID_MEASUREMENT_COMPLETED",
         "implementation_kind": "TRITON"},
        {"operator_family": "ELEMENTWISE", "support_status": "READY_FOR_MEASUREMENT",
         "implementation_kind": "TRITON"},
        {"operator_family": "MASK_POSITION_CONTROL", "support_status": "IDENTIFIED",
         "implementation_kind": "TRITON"},
    ]}
    queue = {"rows": [
        {"operator_family": "ELEMENTWISE", "wave": "NEW_FAMILY_FIRST", "release": "/r",
         "task_id": "backward:1:out", "implementation_kind": "TRITON",
         "reference_candidates": [{"reference_method": "AOT_REPLAY"}]},
    ]}
    report = {"families": [{"family_id": "SOFTMAX", "support_stage_counts": {
        "VALID_MEASUREMENT_COMPLETED": 1}}]}
    result = build(catalogue, queue, report)
    assert result["already_covered_families"] == ["SOFTMAX"]
    assert [x["operator_family"] for x in result["new_family_targets"]] == ["ELEMENTWISE"]
    assert result["blocked_or_unbound_families"][0]["operator_family"] == "MASK_POSITION_CONTROL"
    assert result["selection_uses_numerical_outcomes"] is False


def test_unmeasured_family_frontier_does_not_call_historical_role_negative():
    from scripts.build_unmeasured_triton_family_frontier import build

    catalogue = {"positions": [{"operator_family": "LINEAR",
                                  "support_status": "IDENTIFIED",
                                  "implementation_kind": "EXTERN"}]}
    result = build(catalogue, {"rows": []},
                   {"families": [{"family_id": "LINEAR", "historical_role_records": 2}]})
    row = result["families"][0]
    assert row["status"] == "ALREADY_COVERED_DO_NOT_REPEAT"
    assert "EXPLICIT_ROLE_FAMILY" not in {x["kind"] for x in row["historical_or_measurement_evidence"]}


def test_unmeasured_frontier_does_not_rerun_prior_failed_family(tmp_path):
    from scripts.build_unmeasured_triton_family_frontier import build

    catalogue = {"positions": [{"operator_family": "ELEMENTWISE",
                                  "support_status": "READY_FOR_MEASUREMENT",
                                  "implementation_kind": "TRITON"}]}
    queue = {"rows": [{"operator_family": "ELEMENTWISE", "wave": "NEW_FAMILY_FIRST",
                         "release": "/r", "task_id": "backward:1:out",
                         "implementation_kind": "TRITON"}]}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": [{
        "operator_family": "ELEMENTWISE", "execution_status": "EXECUTION_FAILED",
        "failure_reason": "graph mismatch",
    }]}))
    result = build(catalogue, queue, prior_campaigns=[summary])
    assert result["new_family_targets"] == []
    assert result["blocked_or_unbound_families"][0]["reason"] == "MEASUREMENT_BLOCKED_AFTER_PRIOR_ATTEMPT"


def test_unmeasured_frontier_promotes_valid_campaign_to_covered(tmp_path):
    from scripts.build_unmeasured_triton_family_frontier import build

    catalogue = {"positions": [{"operator_family": "ELEMENTWISE",
                                  "support_status": "READY_FOR_MEASUREMENT",
                                  "implementation_kind": "TRITON"}]}
    queue = {"rows": [{"operator_family": "ELEMENTWISE", "wave": "NEW_FAMILY_FIRST",
                         "release": "/r", "task_id": "backward:1:out",
                         "implementation_kind": "TRITON"}]}
    summary = tmp_path / "run.json"
    summary.write_text(json.dumps({"rows": [{
        "operator_family": "ELEMENTWISE", "status": "MEASUREMENT_VALID",
    }]}))
    result = build(catalogue, queue, prior_campaigns=[summary])
    assert result["new_family_targets"] == []
    assert result["families"][0]["status"] == "ALREADY_COVERED_DO_NOT_REPEAT"
