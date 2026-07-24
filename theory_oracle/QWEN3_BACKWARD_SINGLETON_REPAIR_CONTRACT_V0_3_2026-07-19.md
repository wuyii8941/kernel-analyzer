# Qwen3 backward singleton safe-softmax repair contract v0.3

This treatment replaces the single generated
`triton_red_fused__safe_softmax_add_prepare_softmax_online_view_12` invocation
with the source-graph boundary `attention_scores + attention_mask` followed by
eager `aten._safe_softmax` over the last dimension in FP32.

The generated family mutates its destination buffer; the replacement must
write the eager result to that same buffer and preserve all other compiled
calls.  A valid run must satisfy the same frozen scorer, candidate-identity,
single-hook, single-module and single-repair gates used by the earlier
singleton campaign, and must retain complete clipped-gradient and update
vectors.

The estimand is selected-state intervention impact.  This is a generated fused
reduction-family repair, not an attribution to each constituent operation, a
correctness judgment, or a statement about other attention layers or states.
