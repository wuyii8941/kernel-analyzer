from verify_qwen3_eager_layer_boundary_mediation_v0_1 import (
    disagreement_coordinates,
    validate,
)
import hashlib


def fixture(tmp_path):
    contract = {
        "reference_scorer_sha256": "eager",
        "candidate_scorer_sha256": "candidate",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(__import__("json").dumps(contract))
    manifest = {
        "realization_contract": str(contract_path),
        "generated_kernel": "kernel",
        "expected_runtime_calls": 2,
        "selected_call_indices": [0],
    }
    record = {
        "call_index": 0,
        "weight_storage_identity": True,
        "weight_exact": True,
        "residual_value_transport_contract": True,
        "norm_value_transport_contract": True,
        "destination_layout_preserved": True,
    }
    report = {
        "valid": True,
        "status": "VALID",
        "gates": {"gate": True},
        "anchors": {
            "eager": ["eager", "eager"],
            "candidate": ["candidate", "candidate"],
            "restored": ["candidate", "candidate"],
        },
        "baseline_endpoint": {"fork_coordinates": [[0, 1]]},
        "kernel_family": {
            "provenance_rows": [{"kernel_id": "a"}, {"kernel_id": "b"}]
        },
        "treatments": {
            "0": {
                "noop": {
                    "hashes": ["candidate", "candidate"],
                    "call_records": [
                        [{"calls": 2, "interventions": 0}],
                        [{"calls": 2, "interventions": 0}],
                    ],
                },
                "intervention": {
                    "call_records": [
                        [{"calls": 2, "interventions": 1}],
                        [{"calls": 2, "interventions": 1}],
                    ],
                    "boundary_record_repeat_exact": True,
                    "boundary_records": [[record], [record]],
                },
                "fixed_original_suffix_mediation": {
                    "observed_continuous": True,
                    "off_to_on": 0,
                    "on_to_off": 1,
                    "semantic_disagreement": 0.5,
                },
            }
        },
    }
    inventory = {
        "kernels": [
            {"generated_symbol": "kernel", "kernel_id": "a"},
            {"generated_symbol": "kernel", "kernel_id": "b"},
        ]
    }
    gate = {"forward_kernel_inventory_eligible": True}
    eager = {"semantic": {"clip_decisions": [[False, False]]}}
    compiled = {"semantic": {"clip_decisions": [[False, True]]}}
    return report, manifest, inventory, gate, eager, compiled


def test_disagreement_coordinates():
    assert disagreement_coordinates([[False, True]], [[True, True]]) == [[0, 0]]


def test_valid_fixture(tmp_path):
    assert validate(*fixture(tmp_path)) == []


def test_rejects_noop_change(tmp_path):
    values = fixture(tmp_path)
    values[0]["treatments"]["0"]["noop"]["hashes"][0] = "changed"
    assert "no-op differs" in " ".join(validate(*values))


def test_rejects_transport_failure(tmp_path):
    values = fixture(tmp_path)
    values[0]["treatments"]["0"]["intervention"]["boundary_records"][0][0][
        "destination_layout_preserved"
    ] = False
    assert "destination_layout_preserved" in " ".join(validate(*values))


def test_valid_contextual_layer(tmp_path):
    values = fixture(tmp_path)
    report, manifest = values[0], values[1]
    manifest["contextual_layer_slices"] = [1]
    record = {
        "call_index": 1,
        "weight_storage_identity": True,
        "weight_exact": True,
        "residual_value_transport_contract": True,
        "norm_value_transport_contract": True,
        "destination_layout_preserved": True,
    }
    arms = {
        name: {
            "hashes": ["candidate", "candidate"] if name == "noop" else [name, name],
            "repeat_exact": True,
            "call_records": [[count], [count]],
            **(
                {"matches_independent_entry_boundary_treatment": True}
                if name == "compiled_block"
                else {"matches_independent_exit_boundary_treatment": True}
                if name == "eager_block"
                else {}
            ),
        }
        for name, count in {
            "noop": {"calls": 2, "entry_injections": 0, "exit_injections": 0},
            "compiled_block": {"calls": 2, "entry_injections": 1, "exit_injections": 0},
            "eager_block": {"calls": 2, "entry_injections": 1, "exit_injections": 1},
        }.items()
    }
    report["contextual_layer_slices"] = {
        "1": {
            **arms,
            "same_eager_input_composite_layer_production": {
                "observed": True,
                "compiled_exit_records_repeat_exact": True,
                "compiled_and_eager_arms_pre_repair_exit_exact": True,
                "compiled_exit_record": record,
            },
            "fixed_original_suffix_layer_mediation": {
                "observed_continuous": True,
                "off_to_on": 0,
                "on_to_off": 1,
                "semantic_disagreement": 0.5,
            },
        }
    }
    assert validate(*values) == []


def test_valid_subblock_and_kernel_slice(tmp_path):
    values = fixture(tmp_path)
    report, manifest, inventory = values[0], values[1], values[2]
    manifest.update(
        {
            "intermediate_generated_kernel": "mid",
            "subblock_layer_slices": [1],
        }
    )
    inventory["kernels"].extend(
        [
            {"generated_symbol": "mid", "kernel_id": "m0"},
            {"generated_symbol": "mid", "kernel_id": "m1"},
            {"generated_symbol": "mid", "kernel_id": "m2"},
        ]
    )
    code_path = tmp_path / "generated_kernel.py"
    code = "tl.sum(x)\nlibdevice.rsqrt(y)\nz.to(tl.float32)\ntl.store(out, z)\n"
    code_path.write_text(code)
    code_sha256 = hashlib.sha256(code.encode()).hexdigest()
    for row in inventory["kernels"][-3:]:
        row.update(
            {
                "output_code_path": str(code_path),
                "output_code_sha256": code_sha256,
            }
        )
    report["kernel_family"]["intermediate_provenance_rows"] = [
        {
            "kernel_id": f"m{i}",
            "output_code_path": str(code_path),
            "output_code_sha256": code_sha256,
        }
        for i in range(3)
    ]
    boundary = {
        "call_index": 1,
        "weight_storage_identity": True,
        "weight_exact": True,
        "destination_layout_preserved": True,
    }
    intermediate = {
        "call_index": 1,
        "weight_storage_identity": True,
        "weight_exact": True,
        "attention_transport_contract": True,
        "post_norm_transport_contract": True,
        "destination_layout_preserved": True,
        "same_input_kernel_post_norm": {"nonzero": 1},
        "same_input_kernel_post_norm_production": True,
        "same_input_kernel_reference_post_norm_sha256": "ref",
        "same_input_kernel_input_residual_sha256": "res",
        "same_input_kernel_input_attention_sha256": "attn",
    }
    arms = {}
    for name, injections in {
        "noop": 0,
        "compiled_attention": 0,
        "eager_attention": 1,
        "kernel_reference": 1,
        "eager_block": 1,
    }.items():
        arms[name] = {
            "hashes": ["candidate", "candidate"] if name == "noop" else [name, name],
            "repeat_exact": True,
            "call_records": [
                {
                    "boundary": [{"calls": 2, "entry_injections": 1 if name != "noop" else 0, "exit_injections": 1 if name == "eager_block" else 0}],
                    "intermediate": [{"calls": 3, "injections": injections}],
                }
            ]
            * 2,
            "boundary_records": [[boundary], [boundary]],
            "intermediate_records": [[intermediate], [intermediate]],
        }
    arms["compiled_attention"]["matches_independent_entry_boundary_treatment"] = True
    arms["eager_block"]["matches_independent_exit_boundary_treatment"] = True
    report["subblock_layer_slices"] = {
        "1": {
            "arms": arms,
            "same_input_intermediate_kernel_production": {
                "generated_kernel": "mid",
                "call_index": 1,
                "observed": True,
                "record": intermediate,
            },
            "same_eager_input_attention_region_production": {
                "observed": True,
                "records_repeat_exact": True,
                "compiled_and_eager_attention_arms_pre_repair_exact": True,
                "compiled_attention_record": intermediate,
            },
            "same_eager_attention_input_mlp_region_production": {
                "observed": True,
                "records_repeat_exact": True,
                "eager_attention_and_eager_block_arms_pre_repair_exit_exact": True,
                "compiled_mlp_exit_record": boundary,
            },
            **{
                name: {
                    "observed_continuous": True,
                    "off_to_on": 0,
                    "on_to_off": 0,
                    "semantic_disagreement": 0.0,
                }
                for name in (
                    "fixed_original_suffix_attention_region_mediation",
                    "fixed_original_suffix_kernel_mediation",
                    "fixed_original_suffix_mlp_region_mediation",
                )
            },
        }
    }
    assert validate(*values) == []
