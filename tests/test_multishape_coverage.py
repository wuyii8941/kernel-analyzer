import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    with gzip.open(ROOT / "results/coverage/fb_multishape_ledger.json.gz", "rt") as handle:
        return json.load(handle)


def test_active_denominator_is_exactly_four_models_by_three_shapes() -> None:
    ledger = load()
    assert set(ledger["denominator_cells"]) == {
        "qwen3_1p7b",
        "mamba_130m",
        "phi4_mini_3p8b",
        "deepseek_r1_0528_qwen3_8b",
    }
    assert ledger["denominators"]["declared_model_shape_cells"] == 12
    assert ledger["audits"]["shape_evidence_inherited"] is False
    assert ledger["audits"]["operator_family_deduplication_used"] is False


def test_qwen_three_shapes_have_independent_origin_censuses() -> None:
    cells = ledger_cells = load()["denominator_cells"]["qwen3_1p7b"]
    assert set(cells) == {"batch1_seq64", "batch1_seq128", "batch1_seq256"}
    assert cells["batch1_seq64"]["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED"
    assert cells["batch1_seq128"]["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED"
    assert cells["batch1_seq256"]["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED"
    # Weak observation is the scientific denominator.  The strong-origin
    # witness contains four observer-induced forward detach calls.
    assert cells["batch1_seq128"]["execution_census_invocations"] == 9269
    assert cells["batch1_seq256"]["execution_census_invocations"] == 9269
    assert len({row["source_ledger"] for row in cells.values()}) == 3
    assert ledger_cells["batch1_seq128"]["primary_fb_proof_units"] is not None


def test_no_declared_cell_is_left_uncaptured() -> None:
    ledger = load()
    assert ledger["denominators"]["captured_origin_bound_cells"] == 12
    assert ledger["denominators"]["pending_cells"] == 0
    assert sum(
        cell["analytic_fb_proof_units"] == cell["primary_fb_proof_units"]
        for model in ledger["denominator_cells"].values()
        for cell in model.values()
    ) == 12


def test_mamba_three_shapes_have_independent_origin_censuses() -> None:
    cells = load()["denominator_cells"]["mamba_130m"]
    assert all(row["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED" for row in cells.values())
    assert cells["batch1_seq64"]["execution_census_invocations"] == 56411
    assert cells["batch1_seq128"]["execution_census_invocations"] == 108635
    assert cells["batch1_seq256"]["execution_census_invocations"] == 213083
    assert len({row["source_ledger"] for row in cells.values()}) == 3


def test_phi_three_shapes_have_independent_math_closed_censuses() -> None:
    cells = load()["denominator_cells"]["phi4_mini_3p8b"]
    assert all(row["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED" for row in cells.values())
    assert all(row["execution_census_invocations"] == 8223 for row in cells.values())
    assert len({row["source_ledger"] for row in cells.values()}) == 3


def test_deepseek_three_shapes_have_independent_math_closed_censuses() -> None:
    cells = load()["denominator_cells"]["deepseek_r1_0528_qwen3_8b"]
    assert all(row["status"] == "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED" for row in cells.values())
    assert cells["batch1_seq64"]["execution_census_invocations"] == 12044
    assert cells["batch1_seq128"]["execution_census_invocations"] == 11885
    assert cells["batch1_seq256"]["execution_census_invocations"] == 11885
    assert len({row["source_ledger"] for row in cells.values()}) == 3


def test_captured_invocations_are_owned_once_and_proof_levels_are_explicit() -> None:
    ledger = load()
    assert ledger["audits"]["all_qualified_invocations_owned_once"] is True
    assert ledger["audits"]["all_captured_cells_origin_bound"] is True
    assert ledger["audits"]["all_captured_cells_analytically_proved"] is True
    assert all(unit["gates"]["EXECUTED"] for unit in ledger["units"])
    assert all(unit["gates"]["FB_ORIGIN_BOUND"] for unit in ledger["units"])
    assert sum(
        unit["denominator_role"] == "PRIMARY_FB_PROOF"
        and unit["gates"]["FB_ANALYTICALLY_PROVED"]
        for unit in ledger["units"]
    ) == 186807


def test_typed_triton_replacements_supersede_only_their_historical_invalid_cells() -> None:
    status = json.loads((ROOT / "results/coverage/four_model_full_operator_status.json").read_text())
    abi = json.loads((ROOT / "results/coverage/triton_reference_abi_audit.json").read_text())
    assert abi["status"] == "INVALID_REFERENCE_ABI"
    assert status["status"] == "COMPLETE"
    assert status["counts"]["triton_execution_censuses_closed"] == 12
    closed = [
        row for row in status["cells"]
        if row["gates"]["triton_numeric_reference_valid"]
    ]
    assert status["counts"]["triton_precision_oracles_closed"] == len(closed)
    assert len(closed) == 12
    assert all(
        row["triton_precision_oracle"]["path"].endswith("typed_triton_oracle.json.gz")
        and row["triton_precision_oracle"]["gates"]["typed_triton_pointer_abi_valid"] is True
        for row in closed
    )
    same_dtype_closed = [
        row for row in status["cells"]
        if row["gates"]["same_dtype_optimization_reference_valid"]
    ]
    assert status["counts"]["same_dtype_optimization_oracles_closed"] == len(
        same_dtype_closed
    )
    assert status["counts"]["canonical_eager_fb_math_closed"] == 12
    default_aot_closed = [
        row for row in status["cells"]
        if row["gates"]["default_aot_fb_analytically_proved"]
    ]
    assert status["counts"]["default_aot_fb_math_closed"] == len(
        default_aot_closed
    )
    assert default_aot_closed
    assert all(
        all(row["default_aot_fb_math"]["binding_gates"].values())
        and row["default_aot_fb_math"]["capture"]["gate"] is True
        for row in default_aot_closed
    )
    assert status["counts"]["fully_closed_cells"] == sum(
        row["status"] == "COMPLETE_ALL_DECLARED_GATES"
        for row in status["cells"]
    )
    compact = json.loads((ROOT / "results/coverage/invalid_triton_raw_manifest.json").read_text())
    assert compact["status"] == "COMPACTED_INVALID_RAW_REMOVED"
    assert len(compact["files"]) == 12
    assert compact["total_bytes"] == 5692051451
    for row in compact["files"]:
        assert row["state"] == "DELETED_INVALID_ABI_REGENERABLE"
        assert not (ROOT / row["path"]).exists()
        assert (ROOT / row["retained_oracle"]).is_file()
