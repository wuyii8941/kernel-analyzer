# Training-Step Exact-Control Mapping v0.1 — 2026-07-16

## Purpose

Map independently labeled existing cases to exact components of the Training-Step Oracle. These controls validate local transition obligations; they are not all full BERT-program controls.

## Controls

| Control | Independent obligation | Broken result | Fixed/control result | Training-step field |
|---|---|---|---|---|
| foreach `alpha` ignored | declared option must scale the update operand | compiled behaves as `alpha=1` when `alpha=2` | current nightly preserves `alpha` | optimizer/update option relation |
| `create_graph` metadata dropped | result must preserve required differentiation metadata | `requires_grad=False`, no `grad_fn` | current nightly preserves metadata | gradient/autograd state |
| `slice_scatter(...).sum` backward | backward placement relation | compiled input gradient becomes zero | current nightly gradient matches | gradient transition |
| expanded-index update | exact indexing relation | candidate updates wrong rows | clone/no-op-expand controls or later fix | state mutation/index relation |
| materialized BERT/SGD banks | structure/options/buffers/RNG exact relation | no injected positive in this bank | 256/256 covered exact accepts | full-step exact negative control |

## Interpretation

- The confirmed bugs provide independently labeled exact positives and matched fixed/non-trigger negatives.
- Their output magnitude is irrelevant to the exact relation.
- They validate component contracts, not the numerical BERT next-state envelope.
- A constituent control is not evidence that the same operator realization caused the BERT step discrepancy.

## Remaining exact-control gap

The materialized BERT executor still lacks a fresh, full-step positive control that violates one declared exact next-state field while preserving matched initial state and candidate identity. Such a mutation must be independently labeled and frozen before scoring; changing the candidate configuration itself would create an invalid unmatched-state control rather than a semantic violation.
