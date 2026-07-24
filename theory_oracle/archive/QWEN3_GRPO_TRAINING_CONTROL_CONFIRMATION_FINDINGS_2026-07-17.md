# Qwen3 GRPO Training-Control Confirmation Findings — 2026-07-17

## Main result

The corrected v0.2 confirmation is valid and the frozen strict finite-bank training
compatibility endpoint is `REJECT`.

```text
matched rollout states: 20 / 20
token rows: 10,240
clipping-applicable tokens: 4,608
stable natural clipping disagreements: 4
reference-unclipped -> compiled-clipped: 2
reference-clipped -> compiled-unclipped: 2
directional clipping shift count: 0
compiler correctness: no claim
numerical transition: UNINSTANTIATED
population prevalence: not claimed
```

The two trajectories began from different Qwen3-0.6B weight hashes, disjoint prompt
slices and different seeds. Each used a new optimizer state and ten
reference-anchored rollout states. They are new GRPO trajectories, not historical
optimizer replay.

## Execution identity mattered

The initial v0.1 attempt completed the same nominal state count and exposed five
descriptive disagreements, but it was correctly labelled `INVALID`. Dynamo's default
eight-specialization limit caused the final two states in each trajectory to run
without a tracked Inductor invocation.

v0.2 used new prompt-disjoint banks and a predeclared recompile limit of 64. Both
trajectories compiled ten tracked graph specializations and recorded exactly 30
runtime invocations: warmup plus two measured calls for every state. Every row had
valid candidate identity, finite values and exact eager/compiled self repeats.

This is evidence that candidate identity is part of the Oracle, not logging
decoration. Without it, v0.1 would have been overclaimed.

## Bias, heterogeneity and runtime variability

The finite-bank signed log-probability shift was small and negative. It is not a
universal compiler bias. Both trajectories contained positive, negative and exact
zero token deltas, and their rollout-cluster mean shifts ranged across both signs.

Thus the observed structure is:

- a bank-relative average shift;
- stable state/checkpoint-conditioned heterogeneity;
- zero observed within-state variability under the two-repeat deterministic
  protocol;
- unknown sampling uncertainty for any broader deployment population.

No one of these may be relabelled as “floating-point variance.”

## Directional shift versus disagreement

The two crossing directions balanced exactly, so directional shift was zero while
semantic disagreement was nonzero. This is the intended counterexample to treating
one global signed mean or one directional event count as the whole Oracle.

The valid conclusion is:

> eager and compiled are not clipping-decision identical on the frozen Qwen GRPO
> bank, but this bank does not establish a persistent one-direction clipping bias.

## Boundary conditioning

Descriptive analysis over all 4,608 applicable tokens found:

- 227 non-events had absolute numerical deltas at least as large as the smallest
  event delta;
- the largest non-event delta exceeded the largest event delta;
- 28 non-events were at least as close to the reference boundary as the most distant
  event;
- delta-to-margin relational ranking separated events better than raw absolute delta
  in this finite bank.

Neither magnitude nor boundary proximity alone is the event Oracle. The discrepancy
must be sufficiently large and aligned toward the sign-specific boundary. This
supports “boundary-conditioned implementation effect” over a global numerical mean.

The AUC values are descriptive on the same scored bank, not held-out predictive
performance.

## One-step follow-up status

The predeclared earliest event was reconstructed exactly at trajectory A2, optimizer
step 11. Its token, old log-probability, eager/compiled log-probabilities, advantage
sign and candidate identity all matched the original confirmation.

The first A/B/C executor attempt is `INVALID`: its independent compiled scorer did
not preserve the Trainer's `logits_to_keep` realization, so B no longer reproduced
the confirmed compiled endpoint and B/C selected the same branch. The apparent
`B-C=0` and residual ratio 1 have no causal interpretation.

The unexecuted v0.2 correction was withdrawn after static audit. Although it fixed
Trainer scoring, its proposed clipped intervention evaluated `clamp` at the
compiled ratio. Because that ratio is inside the clipping interval, the derivative
remains nonzero and the reference path's flat branch is not reproduced.

The superseding v0.3 follow-up freezes exact Trainer scoring, mandatory A/B/C anchor
parity, the reference branch's functional form, complete B/C log-probability hashes,
compiled graph identity, and an independent result verifier. Its frozen-input
preflight passes. It remains to be executed; no new branch-level one-step
contribution is claimed yet.

## Claim boundary

- The result is application/training compatibility `REJECT`, not compiler wrong-code.
- Four events are a finite-bank count, not deployment prevalence.
- Balanced directions do not make disagreement unimportant.
- No long-run convergence, reward or quality effect is established.
- No operator, kernel or source region is identified.
- The initial checkpoints differ, but only two checkpoint strata and twenty rollout
  clusters are covered.

## Evidence

- v0.1 contract and invalid findings:
  `QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md`,
  `QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_V0_1_INVALID_FINDINGS_2026-07-17.md`;
- v0.2 contract and frozen manifest:
  `QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_2_2026-07-17.md`,
  `QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_MANIFEST_V0_2.json`;
- v0.2 evaluation:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/evaluation.json`;
- boundary analysis:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/boundary_conditioning.json`;
- invalid one-step attempt and correction:
  `QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_V0_1_INVALID_FINDINGS_2026-07-17.md`,
  `QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_V0_2_WITHDRAWAL_2026-07-17.md`,
  `QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_CONTRACT_V0_3_2026-07-17.md`,
  `QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_MANIFEST_V0_3.json`.
