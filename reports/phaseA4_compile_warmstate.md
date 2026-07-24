# Phase A4 Compile Warm-State Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- same model object and CUDA context across all three passes: PASS
- same ordered tokenized samples: PASS
- SDPA-loaded path with MATH backend locked: PASS
- warm-state attribution allowed only if pass 2 equals pass 3 bitwise

## Delta Self Control
| comparison | tokens | affected_cases | nonzero_tokens | mean | p50 | p99 | max | bitwise_equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold_to_warm | 3072 | 20 | 1835 | 0.0013191850359955705 | 9.522773325443268e-07 | 0.015124995708465577 | 0.02783489227294922 | False |
| warm_to_warm | 3072 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | True |

## External Validity
This is a T4 FP16 compile-state audit. It does not establish BF16 compile determinism.

## Conclusion
Warm-state gate: PASS
Warm-up may be excluded from measurement only if warm_to_warm is bitwise equal. cold_to_warm deltas remain a compile-state effect and must be reported separately.
