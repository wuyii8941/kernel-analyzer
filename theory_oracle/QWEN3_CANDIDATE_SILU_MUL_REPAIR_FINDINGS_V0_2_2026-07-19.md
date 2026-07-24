# Qwen3 candidate fused SiLU-multiply repair findings v0.2

The generic audit at
`results/operator_oracle/qwen3_candidate_silu_mul_repair_v0_2/audit.json`
passes all frozen-artifact, anchor, graph, call-count, repeat, no-recompile,
restoration and arithmetic gates.

Calls 0, 14 and 27 all have deterministic nonzero effects on selected-token
log-probs.  Their relation to eager differs by position: call 0 increases L2
distance by about 9.84%, while calls 14 and 27 reduce it by about 1.12% and
2.48%.  Thus the fused family is causally active at this state, but its
discrepancy-explanation effect is layer-conditioned.

Credit belongs only to the fused generated invocation.  The treatment
materializes eager FP16 SiLU before FP16 multiplication, so it tests the fused
versus separated boundary; it cannot distinguish SiLU approximation from
rounding placement or multiplication propagation.

Revision v0.1 failed on mismatched logical view shapes before any repaired
output and receives zero credit.  V0.2 restored the generated kernel's flat
equal-numel element correspondence without changing the selected calls or
observing target outcomes first.
