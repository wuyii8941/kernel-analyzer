import json
from pathlib import Path

import pytest
import gzip

from scripts.audit_qwen_shape_isomorphism import validate_parametric
from scripts.build_qwen_shape_region_bridge import alpha_normalize_buffers


ROOT = Path(__file__).resolve().parents[1]


def test_shape_parameter_validator_accepts_affine_and_quadratic_extents() -> None:
    assert validate_parametric(65, 129, 257, "affine") == (0, 1)
    assert validate_parametric(4096, 16384, 65536, "quadratic") == (0, 1)
    assert validate_parametric("cuda:3", "cuda:0", "cuda:0", "device") == (0, 1)


def test_shape_parameter_validator_rejects_unrelated_change() -> None:
    with pytest.raises(RuntimeError, match="non-parametric difference"):
        validate_parametric(64, 129, 257, "bad")


def test_qwen_three_shape_program_isomorphism_is_complete() -> None:
    audit = json.loads((ROOT / "results/coverage/qwen_three_shape_isomorphism.json").read_text())
    assert audit["status"] == "COMPLETE_EXACT_SHAPE_PARAMETRIC_PROGRAM_ISOMORPHISM"
    assert audit["denominator"]["strong_eager_invocations_per_shape"] == 9273
    assert audit["denominator"]["aot_call_function_nodes_per_shape"] == 8985
    assert audit["gates"]["complete_eager_dataflow_isomorphism"] is True
    assert audit["gates"]["complete_aot_target_and_edge_isomorphism"] is True
    assert audit["gates"]["all_nonconstant_values_are_device_or_sequence_parametric"] is True
    assert audit["gates"]["operator_name_shape_similarity_pairing_used"] is False
    assert audit["gates"]["candidate_values_used"] is False


def test_direct_aten_alpha_normalization_preserves_roles_not_buffer_numbers() -> None:
    left = "aten.index_put_(buf1126, [primals_1], buf1125, True)"
    right = "aten.index_put_(buf1183, [primals_1], buf1182, True)"
    wrong = "aten.index_put_(buf1182, [primals_1], buf1182, True)"
    assert alpha_normalize_buffers(left) == alpha_normalize_buffers(right)
    assert alpha_normalize_buffers(left) != alpha_normalize_buffers(wrong)


def test_qwen_shape_region_bridges_retain_every_compute_boundary() -> None:
    expected = {128: (1447, 1446, 0, 0), 256: (1501, 1497, 2, 1)}
    for shape, counts in expected.items():
        with gzip.open(
            ROOT / f"results/coverage/qwen_seq{shape}_candidate_fb_bridge.json.gz", "rt"
        ) as handle:
            bridge = json.load(handle)
        denominator = bridge["denominator"]
        assert (
            denominator["all_compute_boundaries"],
            denominator["exact_semantic_key_regions"],
            denominator["seq256_split_refinement_regions"],
            denominator["seq256_loss_refinement_regions"],
        ) == counts
        assert bridge["gates"]["candidate_correctness_inferred"] is False
        assert all(bridge["direct_aten"]["checks"].values())


def test_shape_specific_t1_positives_bind_to_exact_fb_units_fail_closed() -> None:
    expected = {128: (45, 32, 30, 2), 256: (52, 44, 39, 5)}
    for shape, counts in expected.items():
        with gzip.open(
            ROOT / f"results/coverage/qwen_seq{shape}_executed_v1_t1_fb_queue.json.gz", "rt"
        ) as handle:
            queue = json.load(handle)
        denominator = queue["denominator"]
        assert (
            denominator["t1_positive_endpoints"],
            denominator["t1_positive_regions"],
            denominator["regions_eligible_for_t2"],
            denominator["regions_unresolved_before_t2"],
        ) == counts
        assert queue["bindings"]["binding_mode"] == (
            "EXACT_SHAPE_REGION_REFINEMENT_TO_SEQ64_FB_REGISTRY"
        )
        assert queue["gates"]["all_t1_positive_regions_retained"] is True
        assert queue["gates"]["candidate_correctness_inferred_from_identity"] is False
        assert all(row["source_seq64_region_ids"] for row in queue["rows"])
