import gzip
import json
from pathlib import Path

from scripts.check_coverage_denominator import verify


ROOT = Path(__file__).resolve().parents[1]


def load_ledger():
    with gzip.open(ROOT / "results/coverage/fb_proof_unit_ledger.json.gz", "rt") as handle:
        return json.load(handle)


def test_contract_freezes_fb_units_without_family_deduplication():
    contract = json.loads((ROOT / "results/coverage/coverage_contract.json").read_text())
    assert contract["schema"] == "kernel-analyzer-coverage-contract-v2"
    assert contract["unit_rules"]["no_operator_family_deduplication"] is True
    assert contract["unit_rules"]["shared_backward_forms_one_minimal_closed_region"] is True
    assert contract["unit_rules"]["unresolved_invalid_and_abstained_remain_in_denominator"] is True
    assert contract["unit_rules"]["primary_fb_unit_must_contain_at_least_one_forward_invocation"] is True
    assert contract["unit_rules"]["backward_only_component_is_auxiliary_not_primary_fb"] is True
    assert contract["ordered_gates"] == [
        "EXECUTED", "MATH_CLOSED", "CANDIDATE_BOUND", "NUMERIC_MEASURED",
        "T1_LOCAL", "T2_CAUSAL", "T3_COHERENT", "T4_ACCUMULATION",
    ]


def test_fb_units_losslessly_partition_every_invocation():
    ledger = load_ledger()
    members = [
        row_id
        for unit in ledger["units"]
        for row_id in unit["members"]["all_invocation_rows"]["ids"]
    ]
    assert len(members) == len(set(members)) == 111529
    assert ledger["denominators"]["execution_census_invocations"] == 111529
    assert ledger["denominators"]["closed_accounting_components"] == 42767
    assert ledger["denominators"]["primary_fb_proof_units"] == 41031
    assert ledger["denominators"]["auxiliary_backward_accounting_units"] == 1736
    assert ledger["denominators"]["active_full_step_fb_proof_units"] == 33733
    assert ledger["partition_audit"]["states_or_repeats_multiply_primary_denominator"] is False


def test_origin_accounting_does_not_impersonate_analytic_proof():
    ledger = load_ledger()
    assert all(unit["gates"]["EXECUTED"] for unit in ledger["units"])
    assert all(unit["gates"]["FB_ORIGIN_BOUND"] for unit in ledger["units"])
    primary = [
        unit for unit in ledger["units"]
        if unit["denominator_role"] == "PRIMARY_FB_PROOF"
    ]
    assert sum(unit["gates"]["FB_ANALYTICALLY_PROVED"] for unit in primary) == 0
    assert sum(unit["gates"]["MATH_CLOSED"] for unit in primary) == 0
    assert any(
        unit["mathematics"]["status"] == "ORIGIN_BOUND_FORMULA_REGISTERED_ONLY"
        for unit in primary
    )
    assert any(
        unit["unit_kind"] == "FUSED_SHARED_BACKWARD_SEMANTIC_REGION"
        for unit in ledger["units"]
    )
    assert any(
        unit["unit_kind"] == "EMPTY_OR_ELIDED_FB_UNIT"
        for unit in ledger["units"]
    )


def test_backward_only_components_do_not_enter_scientific_fb_denominator():
    ledger = load_ledger()
    primary = [
        unit for unit in ledger["units"]
        if unit["denominator_role"] == "PRIMARY_FB_PROOF"
    ]
    auxiliary = [
        unit for unit in ledger["units"]
        if unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
    ]
    assert len(primary) == 41031
    assert len(auxiliary) == 1736
    assert all(unit["members"]["forward_invocation_rows"]["count"] > 0 for unit in primary)
    assert all(unit["members"]["forward_invocation_rows"]["count"] == 0 for unit in auxiliary)
    assert ledger["partition_audit"]["every_primary_unit_contains_forward"] is True
    assert ledger["partition_audit"]["every_backward_only_component_is_auxiliary"] is True


def test_fusion_does_not_collapse_fb_units():
    ledger = load_ledger()
    multiplicity = ledger["fusion_audit"]["candidate_region_to_primary_fb_unit_multiplicity"]
    assert ledger["fusion_audit"]["many_to_one_regions_preserve_all_fb_units"] is True
    assert any(int(count) > 1 and regions > 0 for count, regions in multiplicity.items())


def test_no_forward_origin_link_is_dangling():
    ledger = load_ledger()
    assert all(
        audit["dangling_origin_links"] == []
        for audit in ledger["model_audits"].values()
    )


def test_missing_shapes_and_candidates_fail_closed():
    ledger = load_ledger()
    for model, matrix in ledger["shape_coverage_matrix"].items():
        assert matrix["batch1_seq64"] == "CAPTURED"
        assert matrix["batch1_seq128"] == "PENDING_EXECUTION_DERIVED_WITNESS"
        assert matrix["batch1_seq256"] == "PENDING_EXECUTION_DERIVED_WITNESS"
        cells = ledger["fb_denominator_cells"][model]
        assert cells["batch1_seq64"]["primary_fb_proof_units"] is not None
        assert cells["batch1_seq128"]["primary_fb_proof_units"] is None
        assert cells["batch1_seq256"]["primary_fb_proof_units"] is None
    assert ledger["denominators"]["declared_active_model_shape_cells"] == 12
    assert ledger["denominators"]["captured_active_model_shape_cells"] == 4
    assert ledger["denominators"]["pending_active_model_shape_cells"] == 8
    for unit in ledger["units"]:
        for cell in unit["candidate_cells"].values():
            if cell["measurement_status"] == "UNMEASURED":
                assert cell["correctness_status"] != "EQUIVALENT"


def test_summary_reports_confirmed_tested_total():
    summary = json.loads((ROOT / "results/coverage/fb_coverage_summary.json").read_text())
    assert summary["schema"] == "kernel-analyzer-fb-coverage-summary-v2"
    for model in summary["models"].values():
        assert model["origin_bound_fb_units"] == model["primary_fb_proof_units"]
        assert model["analytic_fb_proof_units"] <= model["primary_fb_proof_units"]
        for candidate in model["candidate_configurations"].values():
            assert candidate["directional_bias_confirmed_fb_units"] <= candidate["numerically_tested_fb_units"]
            assert candidate["numerically_tested_fb_units"] <= candidate["total_fb_units"]


def test_candidate_runtime_denominator_is_cell_specific():
    summary = json.loads((ROOT / "results/coverage/fb_coverage_summary.json").read_text())
    runtime = summary["candidate_runtime_region_denominator"]
    assert runtime["declared_active_model_candidate_shape_cells"] == 21
    assert runtime["captured_active_model_candidate_shape_cells"] == 1
    assert runtime["pending_active_model_candidate_shape_cells"] == 20
    cells = runtime["cells"]
    captured = cells["qwen3_1p7b"]["bf16_inductor_full_step"]["batch1_seq64"]
    assert captured["runtime_region_denominator"]["compute_invocations"] == 1446
    assert cells["qwen3_1p7b"]["liger_fused"]["batch1_seq64"]["runtime_region_denominator"] is None
    assert cells["phi4_mini_3p8b"]["bf16_inductor_full_step"]["batch1_seq128"]["runtime_region_denominator"] is None


def test_standalone_denominator_verifier():
    assert verify() == {
        "legacy_execution_census_invocations": 111529,
        "legacy_primary_fb_proof_units": 41031,
        "legacy_auxiliary_backward_accounting_units": 1736,
        "active_execution_census_invocations": 466419,
        "active_primary_fb_accounting_units": 186807,
        "active_analytic_fb_proof_units": 186807,
        "active_fully_closed_cells": 0,
    }
