from scripts.build_current_qwen_generated_inventory import (
    digest,
    extend_direct_inventory_contracts,
)


def test_masked_scatter_backward_gets_complete_fb_contract() -> None:
    artifact = {
        "status": "UNRESOLVED_DIRECT_RUNTIME_CALL_SUPPLEMENT",
        "rows": [{
            "symbol": "masked_scatter_backward",
            "runtime_function": "torch.ops.aten.masked_scatter_backward.default",
            "mathematical_derivation": None,
        }],
        "gates": {
            "generated_output_code_present": True,
            "all_direct_call_phases_resolved": True,
            "direct_call_region_ids_unique": True,
            "direct_calls_not_silently_in_base_kernel_denominator": True,
            "every_direct_call_has_source_digest": True,
            "every_direct_call_has_forward_backward_math": False,
            "candidate_values_used": False,
            "property_generalization_allowed": False,
        },
        "inventory_sha256": "stale",
    }
    extend_direct_inventory_contracts(artifact)
    assert artifact["status"] == "COMPLETE_DIRECT_RUNTIME_CALL_SUPPLEMENT"
    derivation = artifact["rows"][0]["mathematical_derivation"]
    assert "dsource" in derivation["backward_vjp"]
    unsigned = dict(artifact)
    unsigned.pop("inventory_sha256")
    assert artifact["inventory_sha256"] == digest(unsigned)
