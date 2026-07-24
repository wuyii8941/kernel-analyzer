# Qwen3 final RMSNorm operator pilot findings v0.1

## Verdict

`VALID_SELECTED_STATE_OPERATOR_ATTRIBUTION`, with a null
implementation-specific effect for the final `model.norm` Qwen3RMSNorm
invocation.

All fail-closed gates passed. In particular, splitting pre-norm decoder,
RMSNorm and `lm_head` reproduced both frozen whole-model endpoint hashes
bit-for-bit. All arms were repeat-exact.

## Result

The total eager-to-compiled selected-token log-probability discrepancy was again
L2 `0.09163622558116913`. Both contrasts were exactly zero:

- repair, compiled prefix + eager RMSNorm versus compiled prefix + compiled
  RMSNorm: L2 `0`;
- injection, eager prefix + compiled RMSNorm versus eager prefix + eager
  RMSNorm: L2 `0`.

The repair residual equaled the total discrepancy. Therefore the eager versus
compiled realization of final RMSNorm is not a discrepancy source or
implementation-specific modifier at this selected state. The discrepancy is
already present before final RMSNorm.

## Limits

- RMSNorm is a named DL operator invocation but decomposes into multiple ATen
  operations; no constituent-operation attribution is claimed.
- The null implementation contrast does not show that RMSNorm's mathematical
  map fails to propagate or rescale an upstream discrepancy.
- This is one matched state, one shape and one runtime/compiler protocol.
- Correctness, update propagation and population generality are not established.

The next localization boundary is the final decoder layer. That boundary is a
composite module and must be described as localization rather than as a final
operator attribution result. Only after locating a contributing layer should its
Linear, RMSNorm, attention and residual-add invocations be intervened on.
