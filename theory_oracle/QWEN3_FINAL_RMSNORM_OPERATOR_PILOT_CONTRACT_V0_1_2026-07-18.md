# Qwen3 final RMSNorm operator pilot contract v0.1

## Question

At the frozen Qwen3-0.6B GRPO step-29 matched state, does changing only the
final decoder `Qwen3RMSNorm` invocation change the eager/compiled selected-token
log-probability discrepancy?

The target is the named DL operator invocation `model.norm`. Qwen3 implements it
with cast, square, reduction-mean, add, reciprocal-square-root, multiply and cast
operations. This pilot attributes only at the RMSNorm invocation level; it does
not yet distinguish those constituent ATen operations.

## Arms

- `whole_eager`: frozen original eager endpoint.
- `split_EEE`: eager pre-norm decoder, eager final RMSNorm, eager `lm_head`.
- `split_CCC`: compiled pre-norm decoder, compiled final RMSNorm, compiled
  `lm_head`.
- `repair_CEC`: compiled pre-norm decoder, eager final RMSNorm, compiled head.
- `injection_ECE`: eager pre-norm decoder, compiled final RMSNorm, eager head.

The state, input, precision/runtime protocol, selected-token observable, eager
anchor and candidate anchor are identical to the frozen `lm_head` pilot.

## Fail-closed gates

1. `whole_eager` equals the frozen eager anchor bit-for-bit.
2. `split_EEE` equals `whole_eager` bit-for-bit.
3. `split_CCC` equals the frozen whole-compiled anchor bit-for-bit.
4. Every arm is bit-exact across two repetitions.

Gate 3 is essential: separating the final RMSNorm can inhibit fusion with the
last decoder layer. If it fails, all contrasts are only diagnostics of a new
partitioned realization and cannot attribute the original candidate discrepancy.

## Interpretation

If valid, repair and injection are selected-state, intervention-dependent effects
of the final RMSNorm invocation on selected-token log-probabilities. A null effect
excludes this invocation for this state/protocol only. A non-null effect motivates
separate interventions on its constituent atomic operations. Neither result is a
correctness verdict or a population root-cause claim.
