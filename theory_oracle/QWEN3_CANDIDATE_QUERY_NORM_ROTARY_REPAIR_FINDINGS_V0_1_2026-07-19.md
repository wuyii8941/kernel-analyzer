# Qwen3 query RMSNorm-rotary repair findings v0.1

The independent audit in
`results/operator_oracle/qwen3_candidate_query_norm_rotary_repair_v0_1/audit.json`
passes the frozen-artifact, eager/candidate anchor, graph-family,
treatment-integrity, repeat, no-recompile, restoration and direction-arithmetic
gates.

Calls 0, 14 and 27 all have deterministic nonzero scorer effects. Repairs at
calls 0 and 14 increase whole-eager L2 distance by about 4.20% and 1.22%; the
repair at call 27 decreases it by about 0.57%. Thus the fused family is active
at the frozen state, but replacing one invocation by eager PyTorch arithmetic
is neither uniformly corrective nor position invariant.

The evidence belongs to the whole fused query RMSNorm, rotary application,
query scaling and layout invocation. It cannot assign the effect to a
constituent primitive. It is repair-only, one-state implementation-relative
impact evidence, not injection, population, sufficiency or correctness
evidence.
