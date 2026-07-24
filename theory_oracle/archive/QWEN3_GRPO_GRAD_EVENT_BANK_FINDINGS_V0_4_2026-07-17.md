# Qwen3 GRPO Grad-Enabled Event-Bank Findings v0.4 — 2026-07-17

## Verdict

The frozen finite-bank discrepancy and grad-context clipping-impact subject is
**VALID** and its declared required ledgers are **COMPLETE**.

The strict finite-bank clipping compatibility endpoint is `REJECT`: two stable
eager/compiled clipping disagreements were observed.  This is not a numerical
correctness verdict, a population prevalence estimate or an update-effect result.

## Validity evidence

- Frozen manifest preflight: 61/61 checks passed.
- Twenty pre-minibatch state records and 10,240 token records are complete.
- Every scorer output required grad and used the actual Accelerate FP16 output
  wrapper plus SDPA MATH.
- Each trajectory had ten tracked graph specializations and thirty compiled runtime
  invocations; no claimed candidate call fell back to eager.
- Gradients, tensor versions and Trainer steps were unchanged by every probe; RNG
  was exactly restored.
- For each trajectory, margin rows, samples, final rollout and the 2.3GB final model
  matched the older uninstrumented-control artifact byte-for-byte.
- The independent result audit recomputed event identity and returned `VALID`.

## Discrepancy structure

The descriptive candidate-minus-reference average current-token log-probability
shift over the complete bank is approximately `-5.05e-5`.  It is not a compiler
bias claim: the finite-bank state means range from approximately `-2.65e-4` to
`+2.17e-4`, with population SD across the twenty fixed state means approximately
`1.13e-4`.

Thus the global mean hides stable direction changes across states.  The appropriate
description is a small negative finite-bank average shift plus substantial
state-conditioned heterogeneity.

Both implementations were exactly self-stable in both measured repeats at every
token.  Observed within-state runtime variability is therefore zero under this
protocol.  This does not imply universal deterministic GPU execution.

## Semantic impact

Among 4,608 nonzero-advantage token decisions:

- stable `0->1` disagreements: 2;
- stable `1->0` disagreements: 0;
- repeat-unstable decisions: 0;
- finite-bank disagreement: `2/4608`;
- finite-bank directional shift: `2/4608`.

Both events occur in trajectory A.  Under the predeclared ordering, the selected
follow-up witness is optimizer step 14, rollout batch 4, flat token index 277,
case `grpo_000004_956b7dace630`, response token index 21.  It has negative
advantage; eager is unclipped and compiled is clipped.

The event count is descriptive for this frozen bank.  Two checkpoint trajectories
do not support a natural-training probability or a stable population direction.

## Why execution context is part of the Oracle

The earlier `no_grad` bank contained four events with balanced directions.  Under
the corrected grad-enabled transition context, two of those events disappear and
the two retained events both have direction `0->1`.

Therefore semantic-event identity is conditional not only on model state and input,
but also on autograd/compiler specialization and AMP output conversion.  A
measurement-path event cannot be transported into an update claim without exact
transition-context reproduction.

## Completeness boundary

The unified result is complete for its declared discrepancy, variability and
finite-bank clipping-impact ledgers.  Correctness remains `UNINSTANTIATED` because
there is no independent legal numerical relation.  Update effect, operator
attribution, long-run behavior and population prevalence remain out of scope.

The next admissible step is to reconstruct the selected grad-context event and
freeze a one-step branch intervention.  The intervention is killed if the complete
eager/compiled scorer tensors or branch decisions fail exact replay.

## Evidence

- [design](QWEN3_GRPO_GRAD_EVENT_BANK_DESIGN_V0_4_2026-07-17.md)
- `QWEN3_GRPO_GRAD_EVENT_BANK_MANIFEST_V0_4.json`
- `results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/evaluation.json`
- `results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/audit.json`
- `results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/unified_oracle_result_v0_4.json`
