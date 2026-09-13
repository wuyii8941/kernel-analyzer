import json

from scripts.verify_adamw8bit_iid_training_confirmation import verify


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_verifier_recomputes_complete_paired_result(tmp_path):
    protocol = {
        "source_sha256": {}, "stream_count": 2, "conditions": ["OFF", "ON"],
        "steps": 1, "evaluation_states": 2, "material_improvement_margin": 0.01,
    }
    save(tmp_path / "protocol.json", protocol)
    paired = []
    for stream, (off, on) in enumerate(((1.1, 1.0), (1.1, 1.0))):
        for condition, value in (("OFF", off), ("ON", on)):
            save(tmp_path / "runs" / f"stream_{stream:02d}_{condition}.json", {
                "status": "COMPLETE", "stream": stream, "condition": condition,
                "evaluation_loss_by_step": {"1": [value, value]},
            })
        paired.append(off - on)
    from scripts.verify_adamw8bit_iid_training_confirmation import interval
    save(tmp_path / "summary.json", {
        "primary": {
            "paired_finite_count": 2, "paired_values": paired,
            "decision": "MATERIAL_IMPROVEMENT",
            "complete_case_description": {
                "mean": sum(paired) / 2, "interval_95": interval(paired),
            },
        }
    })
    result = verify(tmp_path)
    assert result["status"] == "VERIFIED"
    assert result["all_frozen_pairs_have_finite_endpoints"] is True


def test_verifier_does_not_impute_numerical_failure(tmp_path):
    save(tmp_path / "protocol.json", {
        "source_sha256": {}, "stream_count": 1, "conditions": ["OFF", "ON"],
        "steps": 1, "evaluation_states": 1, "material_improvement_margin": 0.01,
    })
    save(tmp_path / "runs" / "stream_00_OFF.json", {
        "status": "NUMERICAL_FAILURE", "stream": 0, "condition": "OFF",
    })
    save(tmp_path / "runs" / "stream_00_ON.json", {
        "status": "COMPLETE", "stream": 0, "condition": "ON",
        "evaluation_loss_by_step": {"1": [1.0]},
    })
    save(tmp_path / "summary.json", {"primary": {"decision": "NOT_ASSESSED_DUE_TO_NUMERICAL_FAILURE"}})
    result = verify(tmp_path)
    assert result["status"] == "VERIFIED"
    assert result["all_frozen_pairs_have_finite_endpoints"] is False
    assert result["recomputed_primary"]["paired_finite_count"] == 0
