from kernel_analyzer.semantic_family_candidates import suggest_family
from scripts.group_uncovered_semantic_candidates import group


def test_specific_operations_precede_generic_reduction():
    assert suggest_family([
        "aten.sum", "aten.nll_loss_forward", "aten._log_softmax",
    ])["suggested_family"] == "CROSS_ENTROPY"
    assert suggest_family([
        "aten.mean", "aten.pow", "aten.rsqrt",
    ])["suggested_family"] == "NORMALIZATION"


def test_new_control_and_unresolved_groups_remain_review_only():
    control = suggest_family(["aten.cumsum", "aten.ne", "aten.slice"])
    assert control["suggested_family"] == "POSITION_OR_MASK_CONTROL"
    assert control["family_already_in_reporting_catalogue"] is False
    assert control["semantic_review_status"] == "REQUIRES_HUMAN_REVIEW"
    assert suggest_family(["aten.exp", "aten.mul"])["suggested_family"] == (
        "UNRESOLVED_MIXED_NUMERICAL")


def test_rotary_variants_without_paired_sin_cos_are_recognized_conservatively():
    cosine_fused = suggest_family([
        "aten.cos", "aten.bmm", "aten.cat", "aten.neg", "aten.mul",
    ])
    assert cosine_fused["suggested_family"] == "ROTARY"
    assert cosine_fused["suggestion_basis"] == (
        "COSINE_WITH_ROTATE_HALF_EXPRESSION")

    precomputed = suggest_family([
        "aten.slice", "aten.neg", "aten.cat", "aten.mul", "aten.transpose",
    ])
    assert precomputed["suggested_family"] == "ROTARY"
    assert precomputed["suggestion_basis"] == (
        "ROTATE_HALF_WITH_PRECOMPUTED_FREQUENCIES")

    # Neither weak signature is sufficient on its own.
    assert suggest_family(["aten.cos", "aten.bmm"])["suggested_family"] == (
        "UNRESOLVED_MIXED_NUMERICAL")
    assert suggest_family(["aten.cat", "aten.neg"])["suggested_family"] == (
        "UNRESOLVED_MIXED_NUMERICAL")


def test_attention_position_scaling_stays_a_new_review_candidate():
    result = suggest_family([
        "aten.arange", "aten.div", "aten.floor", "aten.log", "aten.mul",
        "aten.add", "aten.unsqueeze", "aten._to_copy",
    ])
    assert result["suggested_family"] == "ATTENTION_POSITION_SCALING"
    assert result["suggestion_basis"] == (
        "LOG_BLOCK_POSITION_SCALING_EXPRESSION")
    assert result["family_already_in_reporting_catalogue"] is False
    assert result["semantic_review_status"] == "REQUIRES_HUMAN_REVIEW"


def test_position_construction_and_recurrent_selection_are_separated():
    position = suggest_family([
        "aten.arange", "aten.add", "aten.slice", "aten.constant_pad_nd",
        "aten.sub", "aten.unsqueeze",
    ])
    assert position["suggested_family"] == "POSITION_OR_MASK_CONTROL"

    recurrence = suggest_family([
        "aten.bmm", "aten.select_backward", "aten.add", "aten.squeeze",
    ])
    assert recurrence["suggested_family"] == "RECURRENCE"
    assert recurrence["suggestion_basis"] == (
        "BMM_WITH_TIME_SELECTION_BACKWARD")


def test_group_counts_definitions_not_model_locations():
    result = group({"clusters": [{
        "representative_symbol": "a", "definition_count": 4,
        "source_file_count": 2,
        "compiler_original_aten": ["aten.gelu", "aten.mul"],
    }, {
        "representative_symbol": "b", "definition_count": 3,
        "source_file_count": 1,
        "compiler_original_aten": ["aten.cumsum", "aten.ne"],
    }]})
    assert result["definition_count"] == 7
    assert result["structural_cluster_count"] == 2
    assert {row["suggested_family"] for row in result["families"]} == {
        "GELU", "POSITION_OR_MASK_CONTROL"}
