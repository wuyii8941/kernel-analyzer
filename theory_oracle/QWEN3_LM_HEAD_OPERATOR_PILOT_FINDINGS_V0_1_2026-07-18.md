# Qwen3 `lm_head` operator pilot findings v0.1

## Verdict

`VALID_SELECTED_STATE_OPERATOR_ATTRIBUTION`, with a null effect for the final
`lm_head` linear invocation.

All four fail-closed gates passed:

- the whole eager scorer reproduced its frozen SHA256 exactly;
- splitting eager body and eager head preserved that scorer bit-for-bit;
- separately compiling body and head reproduced the frozen whole-compiled scorer
  bit-for-bit;
- every arm was bit-exact over two target-state repetitions.

Thus, for this state, the body/head split did not introduce the usual fusion or
compilation-boundary confound.

## Result

The eager-to-compiled selected-token log-probability discrepancy had:

- L2: `0.09163622558116913`;
- mean absolute difference: `0.001379203051328659`;
- maximum absolute difference: `0.04759788513183594`;
- signed mean candidate-minus-reference: `0.00038522854447364807`.

Changing only `lm_head` produced exactly zero change in both directions:

- repair, compiled body + eager head versus compiled body + compiled head: L2 `0`;
- injection, eager body + compiled head versus eager body + eager head: L2 `0`.

The repair residual equaled the total discrepancy exactly. Therefore the eager
versus compiled realization of the final linear invocation is not a numerical
discrepancy source or implementation-specific modifier for this selected scorer
observable under this protocol. The discrepancy is already present at the
hidden-state input to `lm_head`.

This intervention does **not** test whether the mathematical linear map propagates,
amplifies or suppresses an upstream hidden-state discrepancy. That requires a
separate upstream-discrepancy injection estimand.

## What this does not establish

- It does not show which upstream operator generated the discrepancy.
- It does not establish a population effect across training states.
- It does not establish correctness or long-run training harm.
- It does not exclude `lm_head` as a propagation map for discrepancy injected
  upstream.
- It does not imply that `lm_head` has zero effect under other shapes, dtypes,
  compiler configurations or states.

The next upstream boundary is the final Qwen3 RMSNorm. That intervention must
again reproduce both frozen endpoint anchors before any causal interpretation.
