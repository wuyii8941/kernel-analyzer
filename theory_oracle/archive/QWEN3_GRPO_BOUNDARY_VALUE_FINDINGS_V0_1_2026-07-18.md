# Qwen3 GRPO Boundary-Conditioning Incremental-Value Findings v0.1 — 2026-07-18

## Verdict

Boundary conditioning is **nonredundant with raw absolute log-probability delta on
the frozen v0.4 bank**.

This is retrospective finite-bank construct evidence. It is not a held-out
predictive result, population claim, correctness verdict or new event discovery.

## Complete-bank results

The analysis covers all 4,608 nonzero-advantage tokens from both trajectories and
retains the twenty parent rollout-state clusters. Nine clusters contain applicable
nonzero-advantage tokens; the remaining clusters contain only zero-advantage rows
for this endpoint.

The two frozen stable clipping events rank:

| event | raw `abs(delta)` rank | direction/distance boundary rank |
|---|---:|---:|
| trajectory A, step 14, flat 277 | 7 | 1 |
| trajectory A, step 17, flat 77 | 49 | 2 |

The largest raw-delta non-event has absolute delta about `0.03114`, larger than
both event deltas. It lies about `2.836` from the relevant reference boundary and
its signed movement does not cross back toward that boundary.

Descriptive average precision is about `0.0918` for raw delta and `1.0` for the
boundary score. This difference must not be described as out-of-sample predictive
performance: the boundary score algebraically uses the same sign-specific event
geometry that defines a crossing. Its purpose is to validate that the semantic
Oracle retains information that raw magnitude discards.

## Interpretation

The result directly realizes two counterexamples required by the theory:

- a larger numerical delta with no semantic event because it is far from the
  boundary and directionally irrelevant;
- a smaller delta with an event because it is aligned with a nearby boundary.

It therefore defeats the narrow kill criterion “the semantic Oracle merely ranks
cases exactly as raw numerical delta does” on this bank. This is a post-execution
semantic detector, not a predictor: because it consumes the observed candidate
delta, repeating its AP calculation on held-out states would still be largely
algebraic. A stronger predictive claim would require a score frozen on one bank
that does **not** consume held-out candidate delta or event labels.

## Schema erratum discovered

The v0.4 evaluator's denominator of 4,608 correctly includes all nonzero-advantage
tokens and applies sign-specific upper/lower clipping boundaries. One old unified
endpoint identifier contains the phrase `negative_advantage`, which is narrower
than the actual estimand. Numerical results are unchanged, but future records must
name the endpoint `grad_context_grpo_clipping` or explicitly state “all nonzero
advantage signs.” Frozen v0.4 artifacts are retained rather than silently renamed.

## Evidence

- frozen diagnostic contract:
  `QWEN3_GRPO_BOUNDARY_VALUE_CONTRACT_V0_1_2026-07-18.md`;
- evaluator:
  `evaluate_qwen3_grpo_boundary_value_v0_1.py`;
- result:
  `results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/boundary_value_v0_1.json`.
