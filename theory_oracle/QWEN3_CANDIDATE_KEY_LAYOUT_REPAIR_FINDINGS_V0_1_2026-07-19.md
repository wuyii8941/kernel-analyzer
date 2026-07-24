# Qwen3 key layout/materialization repair findings v0.1

The independent audit in
`results/operator_oracle/qwen3_candidate_key_layout_repair_v0_1/audit.json`
passes the frozen-artifact, eager/candidate anchor, graph-family,
treatment-integrity, repeat, no-recompile, restoration and direction-arithmetic
gates.

Calls 0, 14 and 27 are all exact null effects at the scorer endpoint. Under the
frozen state and shape, replacing the generated key head-repeat,
transpose/materialization and attention scaling invocation with the declared
eager reconstruction does not change the candidate scorer tensor.

This is valid one-state null evidence for the entire fused invocation, not a
proof that layout operations are harmless in general. It neither separates the
constituent operations nor transports to other shapes/states and carries no
injection, population or correctness claim.
