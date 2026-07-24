# Phase 2 Log-Softmax Conditional Coverage

> Hypothetical demand-side calculation only. The canonical Trainer canary later showed FP32 logits on both sides and exact-zero cross delta, so this half-input switch is not instantiated by the canonical recipe.

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- real iteration-2 old/new logprobs and advantages: PASS
- clipping sign branches applied separately: PASS
- same-state L4 alternate logprobs: FAIL / not recorded
- analytic legal bound: FAIL
- fragile or bug labels emitted: PASS / prohibited

## Delta Self Control
The L4 isolation paths are bitwise self-consistent. This coverage calculation consumes margins only and does not invent an alternate-path delta.

## Summary
| bound_kind | conditional_bound | applicable_decisions | conditional_stable_count | conditional_unknown_count | conditional_stable_rate | analytic_legal |
| --- | --- | --- | --- | --- | --- | --- |
| vendor_documented_conditional | 0.018293388146037714 | 39936 | 39620 | 316 | 0.9920873397435898 | False |

## Interpretation
`margin > B_conditional` is a hypothetical conditional coverage calculation under the documented CUDA assumptions. Remaining rows are unknown, not fragile. Because the canonical Trainer path makes this switch a no-op, these counts are not a canonical stability result and cannot support bug classification.

## External Validity
The envelope and margins are T4 FP16 results. Native BF16 requires new kernels, deltas, and bounds.
