# ForkCert Research Status After Priority Execution

## Evidence Now Established

| RQ | Current evidence | Status |
|---|---|---|
| RQ1 existence | 5 audited natural clipping forks; 29 first-draw actual sampling forks | supported on T4 FP16 |
| RQ2 propagation/attribution | all 5 forks replayable; 24/24 Inductor settings each; every case has a singleton scheduling repair | configuration-class attribution supported; unique operator unresolved |
| RQ3 prediction | leave-one-rollout-out predictor; all 5 clipping forks in top 1% margin-risk budget | useful for budget ranking but statistically sparse |
| RQ4 consequence | exact token-gradient fork; one-step A/B/C repair is 54.9% closer to eager; repair advantage is absent by step 20 | one-step causal contribution supported; persistent effect unsupported |
| RQ5 testing | 15 gated artificial mutations; matched-budget delta ranking slightly outperforms fork signal | preregistered superiority claim rejected |

## Reproducibility State

- `results/baseline_manifest.json` freezes all five fork IDs, hashes, states, and replay commands.
- Step 5 and step 11 use originally frozen checkpoints.
- Step 14 uses a deterministic reconstruction accepted only after exact equality on all 512 canonical rollout rows and replay of both eager/compile fork logprobs within `1e-6`.
- Every attribution matrix has 24/24 settings, a reproduced compile baseline fork, valid generated-code canaries, and no missing intervention.
- Current workspace is not recognized as a Git worktree, so manifests record a null commit rather than inventing provenance.

## Claim Boundaries

- No complete analytic legal bound exists; regions remain `unknown` and empirical anomaly only.
- Natural forks are not called implementation bugs.
- Global compile settings establish sufficient scheduling causes, not a unique source-level operator.
- The old 6.49x temporal result is not a gradient-normalized per-step parameter-jump estimate because intermediate parameter checkpoints are absent.
- HF-vLLM is a composite external stack comparison, not an xFormers-only attribution.
- T4 FP16 evidence does not establish native BF16, FlashAttention, or vLLM V1 behavior.

## Remaining Research Gaps

1. Earliest non-perturbing intermediate splice or narrower source-level intervention for unique-operator attribution.
2. Native BF16/FlashAttention/vLLM V1 replication on Ampere-or-newer hardware.
3. More positive rollout clusters and checkpoints for a credible held-out predictor comparison.
4. Reward/advantage and training-data consequences after actual sampling-token forks.
5. Local arithmetic contracts and eager/compiled operation mapping before any legal-bound or certified-bug claim.

## Current Entry Points

- `reports/final_results_summary.md`
- `reports/reviewer_handoff_20260713.md`
- `results/baseline_manifest.json`
- `results/phase9_attribution_interaction.json`
- `results/phase8_matched_step_counterfactual.json`
