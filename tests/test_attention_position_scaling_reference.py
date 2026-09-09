from pathlib import Path

import pytest
import torch

from kernel_analyzer.attention_position_scaling_source import check_source
from kernel_analyzer.source_reference_registry import get_reference


ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "triton_poi_fused__to_copy_add_arange_div_floor_log_mul_unsqueeze_4"


@pytest.mark.parametrize("length", [128, 512])
def test_actual_ministral_position_scaling_source(length):
    path = (ROOT / "results/property/tcmp_allop_v1/heldout"
            / f"ministral3_3b_text{length}/trace"
            / "model__0_forward_segment0_executed/output_code.py")
    contract = check_source(path.read_text(), SYMBOL)
    assert contract["elements"] == length
    assert contract["original_context_length"] == 16384
    assert contract["beta"] == 0.1
    assert contract["output_pointer"] == "out_ptr0"
    assert not contract["runtime_binding_complete"]


def test_changed_position_scaling_formula_is_rejected():
    path = (ROOT / "results/property/tcmp_allop_v1/heldout"
            / "ministral3_3b_text128/trace"
            / "model__0_forward_segment0_executed/output_code.py")
    changed = path.read_text().replace(
        "tmp8 = tl.full([1], 0.1, tl.float32)",
        "tmp8 = tl.full([1], 0.2, tl.float32)",
        1,
    )
    with pytest.raises(ValueError, match="arithmetic"):
        check_source(changed, SYMBOL)


def test_registered_reference_uses_declared_model_formula():
    specification = get_reference("ATTENTION_POSITION_SCALING")
    path = (ROOT / "results/property/tcmp_allop_v1/heldout"
            / "ministral3_3b_text128/trace"
            / "model__0_forward_segment0_executed/output_code.py")
    contract = specification.check_source(path.read_text(), SYMBOL)
    candidate = torch.empty(128, dtype=torch.bfloat16)
    metadata = {
        "symbol": SYMBOL,
        "formal_pointer": "out_ptr0",
        "input_output_storage_aliases": [],
        "runtime_pointers": {"out_ptr0": torch.empty_like(candidate)},
    }
    result = specification.evaluate(
        metadata, candidate, contract, variant="FP32_NATIVE",
    )
    assert result.dtype == torch.bfloat16
    assert result.shape == candidate.shape
    assert torch.equal(result, torch.ones_like(result))
    with pytest.raises(ValueError, match="output boundary"):
        specification.evaluate(
            dict(metadata, formal_pointer="in_ptr0"), candidate, contract,
            variant="FP32_NATIVE",
        )


def test_actual_high_position_fused_rotary_scaling_source_and_reference():
    specification = get_reference("ATTENTION_POSITION_SCALING")
    path = (ROOT / "results/property/numerical_coverage_v1"
            / "ministral_position_scaling_runtime_release_v3/trace"
            / "model__0_forward_segment0_executed/output_code.py")
    symbol = ("triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_div_"
              "expand_floor_log_mul_neg_sin_slice_transpose_unsqueeze_view_4")
    contract = specification.check_source(path.read_text(), symbol)
    assert contract["layout"] == "FUSED_ROTARY_QUERY_SCALING"
    assert contract["elements"] == 32 * 128 * 128

    torch.manual_seed(7)
    query_logical = torch.randn(128, 32, 128, dtype=torch.bfloat16)
    query = query_logical.permute(1, 0, 2)
    frequencies = torch.linspace(0.001, 0.064, 64, dtype=torch.float32)
    positions = torch.arange(16384, 16512, dtype=torch.int64)
    candidate = torch.empty(32, 128, 128, dtype=torch.bfloat16)
    metadata = {
        "symbol": symbol,
        "formal_pointer": "out_ptr0",
        "input_output_storage_aliases": [],
        "runtime_pointers": {
            "in_ptr0": query,
            "in_ptr1": frequencies,
            "in_ptr2": positions,
            "out_ptr0": torch.empty_like(candidate),
            "out_ptr1": torch.empty_like(candidate),
        },
    }
    result = specification.evaluate(metadata, candidate, contract, variant="FP32_NATIVE")
    assert result.shape == candidate.shape
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()

    changed = path.read_text().replace(
        "tmp34 = tl.full([1], 0.1, tl.float32)",
        "tmp34 = tl.full([1], 0.2, tl.float32)", 1)
    with pytest.raises(ValueError, match="arithmetic"):
        check_source(changed, symbol)
