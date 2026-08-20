# Fixed-state conditional debiasing

Candidate: `qwen_seq64_forward_8_output`.
Fixed conditions: 16.

Cross-condition direction is not a gate. Each row below is certified from independent repair draws at one unchanged training state.

## JOINT

Aggregate: `LOCAL_SOURCE_CONDITIONALLY_DEBIASED_WITH_SYSTEMATIC_CANDIDATE_F_B_EFFECT`.

| Layer estimand | Centered conditions | Biased conditions | Total |
|---|---:|---:|---:|
| `candidate_adamw_zero_update_effect_removed` | 0 | 16 | 16 |
| `candidate_gradient_effect_removed` | 0 | 16 | 16 |
| `candidate_local_effect_removed` | 0 | 16 | 16 |
| `candidate_sgd_update_effect_removed` | 0 | 16 | 16 |
| `repair_local_residual` | 16 | 0 | 16 |

The local repair residual is referenced to the exact declared source component. Gradient/update rows are candidate-minus-repair-ensemble effects; they are not an absolute certificate for the repaired arm.
