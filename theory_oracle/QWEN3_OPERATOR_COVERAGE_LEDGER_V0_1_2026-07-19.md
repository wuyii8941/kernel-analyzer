# Qwen3 operator coverage ledger v0.1

## Current verdict

| Claim | Verdict |
|---|---|
| `frozen_forward_descriptive` | `COMPLETE` |
| `frozen_forward_operator_causal` | `INCOMPLETE` |
| `frozen_training_step_causal` | `INCOMPLETE` |
| `population_operator_causal` | `UNINSTANTIATED` |
| `discrepancy_oracle_definition` | `NOT_INVALIDATED_BY_COVERAGE_GAP` |

The frozen forward realization is descriptively inventoried, but no high-level operator family is fully causally covered. Only final `lm_head` and final RMSNorm have valid intervention evidence.

## High-level forward denominator

| Family | Invocations | Valid interventions | State | Missing |
|---|---:|---:|---|---|
| Linear | 197 | 1 | PARTIAL | 196 q/k/v/o/gate/up/down projections |
| RMSNorm | 113 | 1 | PARTIAL | 112 input/post-attention/q/k norm invocations |
| attention_bmm | 56 | 0 | MAPPED_NOT_INTERVENED |  |
| attention_softmax | 28 | 0 | MAPPED_NOT_INTERVENED |  |
| rotary_embedding | 28 | 0 | MAPPED_NOT_INTERVENED |  |
| MLP_SiLU | 28 | 0 | MAPPED_NOT_INTERVENED |  |
| MLP_gate_multiply | 28 | 0 | MAPPED_NOT_INTERVENED |  |
| residual_add | 56 | 0 | MAPPED_NOT_INTERVENED |  |
| token_embedding | 1 | 0 | MAPPED_NOT_INTERVENED |  |
| causal_mask_construction | 1 | 0 | MAPPED_NOT_INTERVENED |  |

Valid invocation coverage: 2/536 (0.373%).

## Training domains not causally covered

| Domain | Descriptive evidence | Causal state |
|---|---|---|
| scorer_postprocess | SOURCE_INSPECTED | UNINSTANTIATED_PROPAGATION |
| grpo_loss_and_event | SOURCE_INSPECTED | PARTIAL_BRANCH_REPAIR_ONLY |
| model_backward | GRAPH_INVENTORIED | UNINSTANTIATED |
| gradient_control | PATH_EXECUTED | UNINSTANTIATED |
| optimizer_amp_scheduler | PATH_EXECUTED | UNINSTANTIATED |

## Interpretation

Oracle measurement validity and operator-analysis completeness are separate. The current coverage gap does not undo the B/H/N/U or selected-state Oracle definition, but it prohibits a complete operator-root-cause claim.

Raw ATen/prims type and invocation details are in the paired JSON ledger.
