# ForkCert Reviewer Handoff (2026-07-13)

## Purpose

This is a review map, not a new experiment or a claim report. It separates current evidence from stale, diagnostic, reconstructed, and incomplete artifacts so that the project can be audited without relying on narrative summaries.

## Research Logic

The intended evidence chain is:

```text
substitutable execution paths
  -> signed token-logprob difference
  -> crossing of a real training/rollout decision boundary
  -> discrete branch or sampled-token fork
  -> gradient, update, or rollout-state consequence
  -> controlled implementation intervention that removes the fork
```

The current project tests this chain primarily with Qwen3-0.6B, GRPO, eager versus `torch.compile`/Inductor, Tesla T4, and FP16. It also tests top-k/top-p sampling with common random numbers.

## Current Evidence By RQ

### RQ1: Existence

- Canonical clipping population: 51,200 token rows.
- Applicable nonzero-advantage decisions: 39,936.
- `fork_possible`: 15.
- Natural clipping branch forks: 5 (`0.01252%` of applicable decisions).
- The five cases occur at optimizer steps 5, 11, and 14.
- Common-random-number sampling: 17 top-k and 12 top-p first-draw sampled-token forks, each out of 1,024 fixed contexts per mechanism.

This supports existence under the measured configuration. It does not establish a universal rate, BF16 behavior, or that the cases are implementation bugs.

### RQ2: Propagation And Attribution

- Three original replay cases (step 5 and two step 11 tokens) have complete 24-setting Inductor configuration matrices.
- Each has at least one singleton setting that removes the branch fork.
- Effective settings span fusion-size limits, loop-order selection, and, for one case, persistent-reduction selection.
- This establishes a sufficient **compile scheduling configuration class**, not a unique source-level operator.
- The two step-14 states were reconstructed deterministically. Their rollout fields match the original 512/512 exactly, and both fork tokens pass exact-gradient replay gates.
- Both step-14 cases now have complete 24-setting matrices. Step14 t34 is eliminated by `max_fusion_size=1` or disabling persistent reductions; t116 is eliminated by disabling persistent reductions.
- Across all five forks, at least one singleton scheduling intervention eliminates the branch difference; no tested case requires an interaction-only repair.
- A complete reference-to-alternative intermediate tensor splice locating the earliest causal operator has not been delivered for all cases.

### RQ3: Prediction

- The earlier predicted count `3.758` is in-sample and must not be called held-out performance.
- Leave-one-rollout-out results predict 1.919 forks for the global-independence model and 3.154 for the signed calibrated model, versus 5 observed.
- Margin-only is the strongest tested ranker (`AP=0.09`; all positives in the top 1%).
- There are only five positives in three rollout clusters. Calibration and comparative-model claims remain weak.

### RQ4: Consequence

- Exact step-5 replay gives a nonzero token-gradient norm near 413.711 on the eager unclipped branch and zero on the compiled clipped branch for that concrete implementation/case.
- In one A/B/C matched SGD step, a compile scheduling repair removes the branch fork and makes the repaired update 54.9% closer to eager than the default compiled update.
- The recovery ratio is 61.0% at step 5 but is gone by step 20 (`A-C/A-B=1.056`).
- The old `6.49x` fork-step parameter-jump claim is not validated because parameter checkpoints exist only at steps 1, 5, and 20. Per-step gradient-scale analysis does not rescue that claim.
- The repair changes compiler scheduling globally, so it supports a causal contribution of the scheduling configuration, not proof that the single fork event is the only cause of the parameter difference.

### RQ5: Testing Utility

- Fifteen model-level artificial mutations were run over 512 aligned tokens each (7,680 mutation rows).
- All mutation canaries are nonzero; the unmutated baseline has 0/512 clipping-branch mismatches against the canonical certificates.
- There are 420 mutation-induced branch forks.
- At a matched 425-alarm budget, delta ranking has precision 1.0 and recall 0.05534; fork signal has precision 0.9882 and recall 0.05469. The paired difference is not significant (`McNemar p=0.5597`), and clustered intervals include zero.
- Fork alarms cover 11/15 mutation families; delta top-425 alarms cover only the catastrophic `rmsnorm_no_upcast` family.
- Four initial-zero-clipping-fork mutations all have nonzero one-step update distance and develop delayed clipping forks at step 2 or 3. They are not evidence of training-equivalent mutants.
- Therefore the preregistered replacement-oracle claim is rejected, while complementary semantic triage and mutation-family diversity are supported.
- These are artificial altered operations, not historical or certified bugs.

## Certification Status

The full `certified-stable / fragile / bug` classifier is not available. The Phase 2 diagnostic bound is about `2.63e5` against empirical cross-path p99 `0.0201`, and its local-contract/mapping assumptions are incomplete. Natural cases must remain `unknown`; empirical thresholds cannot be used to claim bugs.

## Authoritative Artifacts

Read these first:

| Topic | Structured data | Human-readable report |
|---|---|---|
| Frozen five-fork index | `results/baseline_manifest.json` | `reports/fork_cases.md` |
| Canonical clipping scan | `results/phase4_certificates.jsonl` | `reports/phase4.md` |
| Online signed bias | `results/phaseA0_signed_bias_online.json` | `reports/phaseA0_signed_bias_online.md` |
| Self-run independence | `results/phaseA1_self_audit.json` | `reports/phaseA1_self_audit.md` |
| Same-state audit | `results/phaseA3_checkpoint_audit.json` | `reports/phaseA3_checkpoint_audit.md` |
| Legal-bound downgrade | `results/phase2_v2.json` | `reports/phase2_v2.md` |
| Three completed attribution matrices | `results/phase9_attribution_interaction.json` | `reports/phase9_attribution_interaction.md` |
| Step-14 reconstruction gate | `results/step14_reconstruction_validation.json` | `reports/step14_reconstruction_validation.md` |
| Exact step-14 replay | `results/replay/step14_forks_validated.jsonl` | generated replay records |
| One-step A/B/C | `results/matched_step_fusion_r4/clip-step5-grpo_000001_2817771126c0-t80.json` | `reports/milestone1_step5_fusion.md` |
| 20-step trajectory | `results/trajectory_step5_fusion/merged.json` | `reports/trajectory_analysis.md` |
| Gradient confound audit | `results/phase8_matched_step_counterfactual.json` | `reports/phase8_matched_step_counterfactual.md` |
| Actual sampling forks | `results/phase8_sampling_actual_summary.json` | `reports/phase8_sampling_actual.md` |
| Risk predictor | `results/phase8_risk_summary.json` | `reports/phase8_risk_predictor.md` |
| Gated mutation catalog | `results/phase9_mutations_gated/summary.json` | `reports/phase9_mutation_evaluation_gated.md` |
| Testing utility | `results/phase9_testing_utility_gated.json` | `reports/phase9_testing_utility_gated.md` |

## Newly Completed Artifacts

- `results/attribution/task4_step14_t34_full.json`
- `results/attribution/task4_step14_t116_full.json`
- regenerated `results/phase9_attribution_interaction.json`
- regenerated `reports/phase9_attribution_interaction.md`
- corrected `reports/final_results_summary.md`

## Diagnostic Or Superseded Artifacts

- `reports/phaseA0_signed_bias.md` is a historical pooled-state analysis that reports significant signed bias. The same-state online result in `phaseA0_signed_bias_online.md` is authoritative and finds no significant bias.
- `reports/status.md`, `reports/completion_audit.md`, and `reports/plan_traceability.md` were refreshed on July 13 to remove the confounded `6.49x` causal wording and include all five attribution matrices.
- `reports/final_results_summary.md` is the current RQ-level summary after all five attribution matrices.
- `results/phase9_mutations/` is the earlier mutation output. Use `results/phase9_mutations_gated/` because it includes the baseline branch-consistency gate.
- `results/phase9_mutations_smoke*` and `results/phase9_mutations_rms_smoke/` are smoke/debug runs only.
- `*.example.*`, `phase4_online_partial*`, `online_smoke*`, and repeated `matched_step_fusion_r2/r3` outputs are development evidence, not final claim artifacts.
- FP32-versus-FP16 outputs are pipeline debugging only.

## Main Review Risks

1. **Attribution granularity:** global compiler settings are interventions, but they do not identify the earliest unique operator. Wording such as “fusion root cause proven” is too strong.
2. **Step-14 provenance:** the relevant state was deterministically reconstructed rather than preserved originally. Exact field equality and fork replay are strong gates, but the distinction must remain visible.
3. **Consequence isolation:** A/B/C changes a compiler setting for the whole batch/model. It does not isolate only one token branch while holding every other floating-point result fixed.
4. **Long-horizon claim:** the repair benefit disappears by step 20. The earlier 6.49x temporal association is not a matched per-step causal estimate.
5. **Prediction sample size:** five positives in three clusters are too few for a strong predictive-model claim.
6. **Testing labels:** the mutation benchmark asks “artificial mutation or legal path,” while ForkCert asks “semantic boundary crossed.” The benchmark is useful but not a direct validation of all intended testing utility.
7. **Report drift:** historical reports contain contradictory conclusions. Claims should be regenerated from an explicit authoritative-artifact manifest.
8. **External validity:** core evidence is one small model, T4, FP16, one training recipe, and one main backend pair. Native BF16, newer GPUs, FlashAttention, and a clean serving-engine replication remain absent.

## Directory Guide

- `src/forkcert/`: reusable detector, schema, statistics, logprob runner, hooks, bounds, and training utilities.
- `scripts/`: phase drivers and audit scripts. These are executable experiment definitions.
- `configs/`: model/path/checkpoint configurations. Step-specific configs are authoritative for replay.
- `data/`: rollout dumps, fixed samples, and frozen/reconstructed checkpoints. Large `*_policy_*` directories contain model weights.
- `results/`: machine-readable outputs. Prefer JSON/JSONL over report prose when auditing counts.
- `reports/`: generated interpretation. Dates and authoritative status matter because several reports are stale.
- `logs/`: early resume/install logs, useful mainly for environment-failure history.

## Recommended Review Order

1. `results/baseline_manifest.json`
2. `reports/fork_cases.md`
3. five rows with `actual_fork=true` in `results/phase4_certificates.jsonl`
4. `reports/phaseA1_self_audit.md` and `reports/phaseA3_checkpoint_audit.md`
5. `reports/phase8_case_step5.md` and the matched-step R4 JSON
6. `reports/phase9_attribution_interaction.md`, while remembering it currently covers only the three completed matrices
7. `reports/phase8_matched_step_counterfactual.md`
8. `reports/phase8_sampling_actual.md`
9. `reports/phase8_risk_predictor.md`
10. `reports/phase9_testing_utility_gated.md`
11. `reports/phase2_v2.md`

## Bottom Line

The strongest current result is narrow but real: under one frozen Qwen3/T4/FP16 GRPO configuration, eager and compiled execution produce five audited clipping-branch forks, and one case has direct gradient and one-step update consequences that move toward eager under a compile-scheduling intervention. A second mechanism has 29 actual coupled sampling-token forks.

The project has not yet established a legal numerical certificate, a unique operator root cause, long-horizon fork causality, predictor superiority, testing-oracle superiority, or BF16/newer-hardware generalization.
