# Qwen3 SDPA-decomposition barrier contract v0.1

## Goal

Attempt to isolate qk-bmm, safe-softmax and probability-value-bmm at layers 0,
14 and 27 under the frozen Qwen3 step-29 state.

The reconstruction follows the observed AOT decomposition: convert q/k/value
to FP32, multiply q and transposed k separately by `sqrt(scale)`, perform qk
matmul, add the mask, apply `aten._safe_softmax`, perform probability-value
matmul, cast to the query dtype, then transpose/contiguous.

## Fail-closed reconstruction gate

Before any coverage credit, the all-eager decomposed model must reproduce the
pre-instrumentation eager selected-log-probability tensor exactly.  If it does
not, the result is a valid invalidation of this treatment design and no qk,
softmax or pv invocation is covered.

If reconstruction passes, use the same four-arm fixed-boundary design and
integrity gates as prior batches for nine selected invocations.  Passing rows
remain `BARRIER_CONDITIONED`; they do not transport to the original candidate
unless its anchor is reproduced.
