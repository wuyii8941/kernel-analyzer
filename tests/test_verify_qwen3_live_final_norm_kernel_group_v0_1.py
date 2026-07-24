from verify_qwen3_live_final_norm_kernel_group_v0_1 import validate


def fixtures():
    contract = {"reference_scorer_sha256": "e", "candidate_scorer_sha256": "c"}
    report = {
        "valid": True,
        "status": "VALID",
        "gates": {"all": True},
        "anchors": {
            "eager": ["e", "e"],
            "candidate": ["c", "c"],
            "noop": ["c", "c"],
            "repair": ["r", "r"],
            "restored": ["c", "c"],
        },
        "kernel_group": {
            "provenance": {"kernel_id": "k", "fx_node_metadata": [{"name": "n"}]}
        },
        "same_input_production": {
            "observed": True,
            "repeat_exact": True,
            "records": [
                [{"compiled_to_reference_output": {"nonzero": 1}}],
                [{"compiled_to_reference_output": {"nonzero": 1}}],
            ],
        },
        "fixed_original_suffix_mediation": {
            "observed_continuous": True,
            "candidate_to_repair": {"nonzero": 1},
            "off_to_on": 0,
            "on_to_off": 0,
            "semantic_disagreement": 0.0,
        },
    }
    inventory = {"kernels": [{"generated_symbol": "symbol", "kernel_id": "k"}]}
    gate = {"forward_kernel_inventory_eligible": True}
    return contract, report, inventory, gate


def test_noop_control_is_mandatory(monkeypatch, tmp_path):
    contract, report, inventory, gate = fixtures()
    path = tmp_path / "contract.json"
    path.write_text(__import__("json").dumps(contract))
    manifest = {"realization_contract": str(path), "pointwise_kernel": "symbol"}
    report["anchors"]["noop"] = ["changed", "changed"]
    assert "no-op proxy changed candidate" in validate(report, manifest, inventory, gate)


def test_continuous_mediation_does_not_require_semantic_mediation(monkeypatch, tmp_path):
    contract, report, inventory, gate = fixtures()
    path = tmp_path / "contract.json"
    path.write_text(__import__("json").dumps(contract))
    manifest = {"realization_contract": str(path), "pointwise_kernel": "symbol"}
    assert validate(report, manifest, inventory, gate) == []
