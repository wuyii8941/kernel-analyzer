# Qwen3 post-attention residual-RMSNorm repair findings v0.1

The audit in
`results/operator_oracle/qwen3_candidate_post_attention_norm_repair_v0_1/audit.json`
passes all frozen-artifact, anchor, graph, treatment-integrity, repeat,
no-recompile, restoration and arithmetic gates.

Calls 0, 14 and 27 all have deterministic nonzero scorer effects.  Call 0
increases whole-eager L2 distance by about 5.27%; call 14 decreases it by about
5.65%; call 27 has a much smaller effect and decreases distance by about 0.026%.
The family is causally active at the frozen state, but both magnitude and
discrepancy direction are position-conditioned.

The evidence belongs to the fused attention-residual plus RMSNorm plus cast
invocation.  It cannot separate addition, reduction order, `rsqrt`, scaling and
cast placement, and it is not a population, injection, sufficiency or
correctness result.
