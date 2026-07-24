# Qwen3 key RMSNorm-rotary repair findings v0.1

The independent audit in
`results/operator_oracle/qwen3_candidate_key_norm_rotary_repair_v0_1/audit.json`
passes every frozen-artifact, anchor, graph, treatment-integrity, repeat,
no-recompile, restoration and direction-arithmetic gate.

Calls 0, 14 and 27 all have deterministic nonzero scorer effects. Repairs at
calls 0 and 27 reduce whole-eager L2 distance by about 9.28% and 8.35%, while
the call-14 repair increases it by about 1.68%. The fused family is active at
the frozen state, but its discrepancy direction is layer conditioned.

The evidence belongs to the whole fused key RMSNorm, rotary application and
strided-output invocation. It cannot identify a constituent primitive and is
repair-only, one-state implementation-relative impact evidence rather than an
injection, population, sufficiency or correctness result.
