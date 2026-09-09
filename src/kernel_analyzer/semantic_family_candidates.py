"""Conservative family suggestions from compiler-carried operator semantics.

These rules organize human review.  They never establish a reference,
runtime support, numerical equivalence, bias, or a distinct root cause.
"""
from __future__ import annotations


KNOWN_FAMILIES = {
    "NORMALIZATION", "ROTARY", "EMBEDDING", "CROSS_ENTROPY", "SOFTMAX",
    "GELU", "SILU_GATING", "SOFTPLUS", "CONVOLUTION", "RECURRENCE",
    "REDUCTION",
}


def suggest_family(original_aten: list[str]) -> dict:
    operations = set(original_aten)

    def has(name: str) -> bool:
        return "aten." + name in operations

    if has("nll_loss_forward") or has("nll_loss_backward"):
        family, basis = "CROSS_ENTROPY", "NLL_LOSS_OPERATION"
    elif (has("_softmax") or has("_softmax_backward_data")
          or has("_log_softmax") or has("_log_softmax_backward_data")
          or "prims.prepare_softmax_online" in operations):
        family, basis = "SOFTMAX", "SOFTMAX_OR_LOG_SOFTMAX_OPERATION"
    elif has("convolution") or has("convolution_backward"):
        family, basis = "CONVOLUTION", "CONVOLUTION_OPERATION"
    elif has("gelu") or has("gelu_backward"):
        family, basis = "GELU", "GELU_OPERATION"
    elif has("silu") or has("silu_backward"):
        family, basis = "SILU_GATING", "SILU_OPERATION"
    elif has("embedding") or has("embedding_dense_backward"):
        family, basis = "EMBEDDING", "EMBEDDING_OPERATION"
    elif has("sin") and has("cos"):
        family, basis = "ROTARY", "PAIRED_SIN_COS_OPERATIONS"
    elif has("cos") and has("bmm") and has("cat") and has("neg"):
        # Inductor can materialize sine in a neighbouring kernel and carry only
        # the cosine expression into this fused rotate-half computation.  The
        # cat/neg pair is essential here: cos+bmm by itself is not enough to
        # identify rotary position embedding.
        family, basis = "ROTARY", "COSINE_WITH_ROTATE_HALF_EXPRESSION"
    elif has("cat") and has("neg") and has("slice") and has("mul"):
        # Applying already-computed cos/sin values can contain no transcendental
        # operation in this kernel.  The rotate-half expression remains visible
        # as slice, negation, concatenation, and multiplication.
        family, basis = "ROTARY", "ROTATE_HALF_WITH_PRECOMPUTED_FREQUENCIES"
    elif (has("arange") and has("div") and has("floor") and has("log")
          and has("mul")):
        # Seen in Ministral as
        # 1 + c * log(floor(position / block_size) + 1), followed by use in
        # attention.  This is deliberately separate from sin/cos rotary math.
        # The suggestion still requires human confirmation of the graph use.
        family, basis = (
            "ATTENTION_POSITION_SCALING",
            "LOG_BLOCK_POSITION_SCALING_EXPRESSION",
        )
    elif has("rsqrt") and (has("mean") or has("sum")):
        family, basis = "NORMALIZATION", "REDUCTION_WITH_RECIPROCAL_SQRT"
    elif has("softplus") and has("exp") and (has("bmm") or has("select")):
        family, basis = "RECURRENCE", "EXPONENTIAL_SOFTPLUS_STATE_COMPUTATION"
    elif has("softplus") or has("softplus_backward"):
        family, basis = "SOFTPLUS", "SOFTPLUS_OPERATION"
    elif has("select_backward") and has("bmm"):
        family, basis = "RECURRENCE", "BMM_WITH_TIME_SELECTION_BACKWARD"
    elif (has("cumsum") or has("bitwise_and") or has("where")
          or (has("arange") and (has("slice") or has("constant_pad_nd")))):
        family, basis = "POSITION_OR_MASK_CONTROL", "INDEX_MASK_OR_PREFIX_OPERATION"
    elif has("sum") or has("mean") or has("max"):
        family, basis = "REDUCTION", "UNRESOLVED_REDUCTION_COMPUTATION"
    elif operations and operations <= {
        "aten._to_copy", "aten._unsafe_view", "aten.arange", "aten.cat",
        "aten.clone", "aten.constant_pad_nd", "aten.expand", "aten.fill",
        "aten.slice", "aten.squeeze", "aten.stack", "aten.sub", "aten.t",
        "aten.transpose", "aten.unsqueeze", "aten.view",
    }:
        family, basis = "DATA_MOVEMENT_OR_LAYOUT", "LAYOUT_AND_INDEXING_OPERATIONS_ONLY"
    else:
        family, basis = "UNRESOLVED_MIXED_NUMERICAL", "NO_CONSERVATIVE_RULE_MATCH"
    return {
        "suggested_family": family,
        "suggestion_basis": basis,
        "family_already_in_reporting_catalogue": family in KNOWN_FAMILIES,
        "semantic_review_status": "REQUIRES_HUMAN_REVIEW",
    }
