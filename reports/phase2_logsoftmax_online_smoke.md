# Phase 2 Canonical Log-Softmax Canary

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. Exact-zero switches are treated as no-ops until a canary proves otherwise.

## Confound Checklist
- real Trainer/Accelerate execution path: PASS
- same in-memory optimizer step and logits: PASS
- half-input and explicit-float switch attempted: PASS
- nonzero canary delta: FAIL
- natural-scan claim enabled: FAIL / prohibited

## Delta Self Control
Across 512 token decisions, ref self max and alt self max are both exactly zero. Cross max is also exactly zero.

## Summary

| token_decisions | logits_dtype | ref_output_dtype | alt_output_dtype | cross_nonzero | cross_max | training_match_max | actual_forks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | torch.float32 | torch.float32 | torch.float32 | 0 | 0 | 0 | 0 |

## Interpretation

In the actual Trainer path, the Accelerate-wrapped model exposes FP32 logits before selective log-softmax. Calling `.float()` is therefore a no-op. Under P1, canonical L4 fails its positive-switch requirement and is removed from natural-scan attribution. The standalone half-input isolation remains a controlled kernel study only.

The `99.2087%` margin coverage in `reports/phase2_logsoftmax_coverage.md` is a hypothetical calculation under a half-input path that is not instantiated by this canonical Trainer recipe. It is not a canonical stability result.

## External Validity

This no-op finding is specific to the installed Accelerate/Transformers/TRL FP16 stack on T4. Other engines may expose FP16 or BF16 logits to log-softmax and must rerun the canary.
