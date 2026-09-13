from scripts.summarize_triton_signature_groups import summarize


def test_outputs_from_one_compiled_region_are_one_exact_group(tmp_path):
    release = str(tmp_path / "release")
    positions = []
    rows = []
    for index in range(2):
        task = f"backward:1:out_ptr{index}"
        positions.append({
            "release": release, "task_id": task, "release_task_package_sha256": "pkg",
            "candidate_region_id": "region", "phase": "BACKWARD", "symbol": "kernel",
            "carrier": f"parameter.{index}", "operator_family": "REDUCTION",
        })
        rows.append({
            "source_release": release, "task_id": task, "measurement_status": "VALID",
            "analysis_sha256": "same", "signature_key": ["REDUCTION", "TRITON"],
            "fixed_suite_parameter_write_total_rms": 0.1 + index,
        })
    result = summarize({"positions": positions}, [{
        "schema": "triton-signature-runtime-batch-summary-v1", "rows": rows,
    }])
    assert result["valid_position_count"] == 2
    assert result["structural_signature_count"] == 1
    assert result["exact_candidate_computation_count"] == 1
    assert result["independent_root_cause_count"] is None
