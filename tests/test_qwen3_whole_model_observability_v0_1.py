from verify_qwen3_whole_model_observability_v0_1 import compare_transition


def record(token="same"):
    summary = {"tensor_hashes_sha256": token}
    return {
        "valid": True,
        "pre_state": {"parameter": token},
        "post_state": {"parameter": token},
        "anchors": {"observed_scorer_sha256": token},
        "continuous": {
            "scorer_logps": [[1.0]],
            "loss": 1.0,
            "scaled_gradient": summary,
            "unscaled_gradient": summary,
            "clipped_gradient": summary,
            "parameter_update": summary,
        },
        "semantic": {"clip_count": 0},
        "realization": {
            "graph_family_digest": token,
            "compiler_config_digest": token,
        },
        "compiler": {"graph_manifests": []},
        "observability": {
            "trace_files": [],
            "generated_code_count": 0,
            "provenance_mapping_count": 0,
        },
    }


def test_exact_endpoints_without_artifacts_are_not_inventory_eligible():
    result = compare_transition(record(), record())
    assert result["instrumented_pipeline_equivalent"]
    assert not result["artifacts_auditable"]
    assert not result["operator_kernel_inventory_eligible"]


def test_forward_identity_does_not_hide_backward_change():
    baseline = record()
    traced = record()
    traced["continuous"]["clipped_gradient"] = {"tensor_hashes_sha256": "changed"}
    result = compare_transition(baseline, traced)
    assert result["checks"]["scorer_exact"]
    assert not result["checks"]["clipped_gradient_exact"]
    assert not result["instrumented_pipeline_equivalent"]
    assert result["forward_observability_equivalent"]
    assert not result["training_transition_equivalent"]
