from scripts.build_observed_kernel_catalog import build
from scripts.build_family_first_execution_queue import build as build_queue
from scripts.verify_observed_kernel_catalog import verify


def task(task_id, symbol, endpoint, *, kind="TRITON", region="forward:0"):
    return {
        "task_id": task_id, "symbol": symbol, "implementation_kind": kind,
        "candidate_region_id": region, "phase": "FORWARD",
        "exact_semantic_endpoint_id": endpoint, "exact_aot_endpoint_id": endpoint,
        "status": "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT",
    }


def test_catalog_and_queue_preserve_support_stages(tmp_path, monkeypatch):
    # The production builder reads immutable release task packages.  Use a
    # small real gzip package here to exercise the same join.
    import gzip, json
    release = tmp_path / "release"
    release.mkdir()
    rows = [
        task("forward:0:out", "mm", "forward:graph0:mm_default", kind="EXTERN"),
        task("forward:1:out", "triton_poi_fused_silu_1", "forward:graph0:silu", region="forward:1"),
        task("forward:2:out", "opaque_kernel", None, region="forward:2"),
    ]
    with gzip.open(release / "same_dtype_tasks.json.gz", "wt") as stream:
        json.dump({"rows": rows, "reference_cut_tasks": []}, stream)
    inventory = {"records": [
        {"release": str(release), "task_id": "forward:0:out", "symbol": "mm",
         "implementation_kind": "EXTERN", "phase": "FORWARD", "formal_pointer": "output_0",
         "reference_candidates": [{"family": "ROW_SUM"}], "carrier": "weight",
         "runtime_measurement_status": "VERIFIED"},
        {"release": str(release), "task_id": "forward:1:out", "symbol": "triton_poi_fused_silu_1",
         "implementation_kind": "TRITON", "phase": "FORWARD", "formal_pointer": "out_ptr0",
         "reference_candidates": [{"family": "SILU_BACKWARD"}], "carrier": "weight",
         "runtime_measurement_status": "NOT_CAPTURED"},
        {"release": str(release), "task_id": "forward:2:out", "symbol": "opaque_kernel",
         "implementation_kind": "TRITON", "phase": "FORWARD", "formal_pointer": "out_ptr0",
         "reference_candidates": [], "carrier": None,
         "runtime_measurement_status": "NOT_ASSESSED_BY_THIS_INVENTORY"},
    ]}
    catalog, summary = build(inventory, root=tmp_path)
    assert summary["position_count"] == 3
    assert summary["kernel_invocation_count"] == 3
    assert summary["all_positions_classified_including_explicit_unresolved"]
    assert summary["position_support_status_counts"] == {
        "VALID_MEASUREMENT_COMPLETED": 1, "READY_FOR_MEASUREMENT": 1, "IDENTIFIED": 1,
    }
    queue = build_queue(catalog)
    assert queue["position_count"] == 1
    assert queue["rows"][0]["operator_family"] == "SILU_GATING"


def test_existing_generic_reference_capabilities_are_connected(tmp_path):
    import gzip, json
    release = tmp_path / "release"
    release.mkdir()
    rows = [
        task("backward:0:out", "triton_poi_fused_add_1", "backward:graph0:add"),
        task("backward:1:out", "mm", "backward:graph0:mm", kind="EXTERN", region="backward:1"),
    ]
    with gzip.open(release / "same_dtype_tasks.json.gz", "wt") as stream:
        json.dump({"rows": rows, "reference_cut_tasks": [{"task_id": "backward:graph0:add"}]}, stream)
    inventory = {"records": [
        {"release": str(release), "task_id": row["task_id"], "symbol": row["symbol"],
         "implementation_kind": row["implementation_kind"], "phase": "BACKWARD",
         "formal_pointer": "out", "reference_candidates": [], "carrier": "weight",
         "eligibility": "NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT",
         "runtime_measurement_status": "NOT_ASSESSED_BY_THIS_INVENTORY"}
        for row in rows
    ]}
    catalog, summary = build(inventory, root=tmp_path)
    assert summary["position_support_status_counts"] == {"READY_FOR_MEASUREMENT": 2}
    methods = [row["available_reference_bindings"][0]["reference_method"]
               for row in catalog["positions"]]
    assert methods == ["AOT_REPLAY", "EXTERNAL_FP32_RECOMPUTE"]
    assert catalog["positions"][0]["available_reference_bindings"][0][
        "single_kernel_source_attribution"] == "NOT_ESTABLISHED"


def test_queue_deduplicates_copied_release_packages_and_keeps_completed_status():
    base = {
        "release_task_package_sha256": "same-package", "task_id": "backward:1:out",
        "canonical_release": "/canonical", "operator_family": "ELEMENTWISE",
        "implementation_kind": "TRITON", "phase": "BACKWARD", "symbol": "triton_add_1",
        "symbol_signature": "triton_add_{N}", "carrier": "weight",
        "available_reference_bindings": [{"reference_method": "AOT_REPLAY"}],
        "reference_discovery": "EXISTING_GENERIC_AOT_REPLAY_CAPABILITY",
        "classification_confidence": "EXACT_ENDPOINT",
    }
    completed = {**base, "release": "/historical-copy",
                 "support_status": "VALID_MEASUREMENT_COMPLETED"}
    pending = {**base, "release": "/canonical", "support_status": "READY_FOR_MEASUREMENT"}
    assert build_queue({"positions": [pending, completed]})["position_count"] == 0

    queue = build_queue({"positions": [pending]})
    assert queue["position_count"] == 1
    assert queue["rows"][0]["release"] == "/canonical"


def test_catalog_verifier_checks_counts_and_queue_binding():
    positions = [{
        "release": "/release", "task_id": "backward:0:out",
        "release_task_package_sha256": "package", "operator_family": "ELEMENTWISE",
        "support_status": "READY_FOR_MEASUREMENT",
    }]
    summary = {
        "schema": "observed-kernel-catalog-summary-v1",
        "position_count": 1,
        "distinct_release_qualified_position_count": 1,
        "position_support_status_counts": {"READY_FOR_MEASUREMENT": 1},
        "distinct_position_support_status_counts": {"READY_FOR_MEASUREMENT": 1},
    }
    catalogue = {
        "schema": "observed-kernel-catalog-v1", "summary": summary,
        "positions": positions,
    }
    queue = {
        "catalog_sha256": "catalog-sha", "position_count": 1,
        "selection_uses_numerical_outcomes": False,
        "rows": [{"release": "/release", "task_id": "backward:0:out"}],
    }
    result = verify(catalogue, summary, queue, catalog_sha256="catalog-sha")
    assert result["status"] == "VERIFIED"
    assert result["queued_position_count"] == 1
    assert result["all_positions_have_a_family_label"]
