# Current bias + operator + Oracle mainline

This is the authoritative short status. Older round notes are experiment
history and must not be used for the current case count.

## Question

When does a numerical difference introduced by one LLM training operator turn
into a parameter-update error that keeps accumulating instead of canceling?

## Current result

There are **three headline cases**:

- Liger fused cross entropy;
- Phi-4 `lm_head dX`;
- Qwen `lm_head dX`.

The same 32-state measurement was taken at three points:

| case | operator output | parameter gradient | effective parameter update |
|---|---:|---:|---:|
| Liger fused CE | 2.984 | 2.931 | 2.931 |
| Phi-4 `lm_head dX` | 2.074 | 4.701 | 4.701 |
| Qwen `lm_head dX` | 1.008 | 1.698 | 1.698 |

Liger is already directional at the operator boundary. For Phi and Qwen,
the directional effect becomes stronger after the real backward pass reaches
the parameter gradient. Stateless SGD preserves it. AdamW can suppress it:
on Phi, gradient `A=4.665` becomes update `A=1.031`.

## Oracle result

The frozen evaluation has **14 rows: 3 positives and 11 nonzero controls**.
Using only the first 16 steps of effective parameter-update errors:

- AUROC: `1.00`;
- recall: `3/3`;
- false positives: `2/11`;
- flagged rows: `5/14`;
- precision: `3/5`.

The same set gives AUROC `0.242` for local error RMS and `0.50` for BF16 dtype.
This shows that error size and dtype are poor substitutes for measuring whether
the update error keeps pointing consistently along the training trajectory.

## Controls and boundaries

- In the 12-row screen-negative audit, 11 rows have locally canceling operator
  updates but persistent later training feedback. They are controls, not new
  operator-local bias cases.
- The project has results on 10 models. Only Qwen3-1.7B, Phi-4-mini,
  DeepSeek-R1-Qwen3-8B, and Mamba-130M received systematic all-operator
  coverage. The other six models received targeted or held-out tests.
- Historical six-case and eight-row registries remain useful audit records,
  but they are not the current headline count.
- The general source/backward/optimizer/propagation predictor has no eligible
  complete row yet. It abstains rather than filling in missing measurements.
- Complete runtime accounting still needs the separate Mamba timing rerun and
  one uncontended timing pass for the headline cases. All scientific 32-step
  evidence is already complete; these remaining runs measure cost only.

Machine-readable sources:

- `results/property/joint_bias_formation_v1/general_mechanism_map_v1.json`
- `results/property/joint_bias_formation_v1/source_persistence_reclassification.json`
- `results/property/joint_bias_formation_v1/oracle_baselines/frozen_evaluation_v2/comparison_v2.json`
- `results/property/joint_bias_formation_v1/current_status.json`
