# ForkCert Final Results Summary

## Scope And Evidence Contract

ForkCert tests whether numerical differences between substitutable execution paths cross training decision boundaries and change optimization or rollout semantics. The canonical pair is Qwen3-0.6B eager versus `torch.compile`/Inductor on Tesla T4 with FP16 autocast and MATH-locked SDPA.

All clipping claims use fixed token IDs, checkpoint, old logprobs, advantages, masks and position semantics. Self-runs use independent processes and CUDA contexts. The five natural clipping cases passed the structured confound audit. No usable analytic legal error bound exists, so no natural case is labelled `fragile`, `bug` or `certified-stable`; applicable regions remain `unknown`.

## Execution Summary

| Plan task | Result | Decision |
|---|---|---|
| Task 1: matched-step confound | Gradient scale does not explain fork-step grouping, but per-step parameter jumps were not recorded | REVISE |
| Task 2: step-11 t72 extension | 24-setting matrix completed; singleton repairs found | GO |
| Task 3: refined mutations | 15 valid mutations; paired significance audit; held-out latency; cross-decision sampling; task reward | COMPLETE, replacement-oracle claim rejected; staged complementary evidence supported |
| Task 4: attribution interaction | 24/24 settings for all 5 forks; singleton scheduling repair for all 5 | GO at configuration-class scope |
| Task 5: risk calibration | Conditional held-out framing frozen; margin-only remains best tested ranking feature | REVISE |

## RQ1: Existence

The canonical clipping scan contains 51,200 state-aligned rows. Of 39,936 nonzero-advantage decisions, 5 are natural eager/compile branch forks, a conditional rate of `0.0001252` (`0.01252%`). They occur in three rollout groups. Self deltas are exactly zero on all fork tokens, and all five cases pass tokenizer, weight/state, token, mask, position, old-logprob, advantage-sign and branch-formula checks.

Common-random-number sampling establishes a second decision mechanism. On 1,024 fixed contexts, the first coupled draw changes 17 top-k sampled tokens and 12 top-p sampled tokens. All 29 events immediately change the rollout-prefix hash. Sixteen top-k and eight top-p events occur without a candidate-set change, showing that candidate-set equality does not imply CDF sampling stability.

Supported: natural clipping and sampled-token semantic forks exist under the measured T4 FP16 configuration. Not supported: a hardware-independent fork rate, BF16 stability, or reward/advantage consequences for the sampling cases.

## RQ2: Attribution

All five clipping forks are replayable from frozen or exactly validated reconstructed pre-minibatch states. The step-14 reconstruction matches all 512 canonical rollout rows field-for-field and replays both fork logprobs within the registered `1e-6` gate. Each fork was tested under all 24 registered Inductor settings with independent Dynamo reset and cache. Every intervention changed generated code; repeated measurements where available agree within `2e-6` and exactly in branch outcome.

| Fork | Effective singleton settings | Interaction required |
|---|---|---|
| step5 t80 | fusion size 1/2, persistent reductions off, loop-order picking off | no |
| step11 t72 | fusion size 1/2, loop-order picking off | no |
| step11 t88 | fusion size 2, loop-order picking off | no |
| step14 t34 | fusion size 1, persistent reductions off | no |
| step14 t116 | persistent reductions off | no |

Every effective combination contains an already effective singleton; no case requires a two-setting interaction. The step-14 evidence is labelled reconstructed rather than originally frozen.

The supported causal source is an Inductor fusion/loop/reduction scheduling **configuration class**. Different singleton settings can repair the same branch and sometimes converge to the same logprob with different kernel inventories. This does not identify a unique source-level operator or prove a compiler bug. Quantitatively, configuration-class attribution is `5/5`; unique-operator attribution is `0/5`.

## RQ3: Prediction

The estimand is conditional on Qwen3-0.6B canonical online checkpoints, GRPO `eps=0.2`, T4 FP16, and eager versus default Inductor compile. Leave-one-rollout-out fitting prevents test-rollout tokens from entering the signed-delta distribution or calibration set.

| Predictor | Average precision | Top-1% recall | Predicted count |
|---|---:|---:|---:|
| margin only | 0.0900 | 1.00 | n/a |
| global independence | 0.0585 | 1.00 | 1.919 |
| signed conditional calibrated | 0.0270 | 1.00 | 3.154 |

Observed count is 5. The signed model improves count error over the global independence estimate but ranks positives worse than margin-only. Its predicted-count cluster-bootstrap interval is `[2.304, 4.147]`; the observed-count interval `[0, 11]` reflects only three positive rollout clusters.

The earlier `3.758` estimate was in-sample and is not held-out performance. The current evidence supports margin-based test-budget allocation, not superiority of the signed conditional predictor or prediction of individual forks.

## RQ4: Consequence

Exact replay of the step-5 target gives a nonzero per-token gradient norm of approximately `413.711` on the eager unclipped branch and zero on the compiled clipped branch. In the valid single-step A/B/C experiment, changing only Inductor `max_fusion_size` to 2 removes all batch branch forks and restores the target loss gradient:

| Quantity | A eager | B compile | C repaired compile |
|---|---:|---:|---:|
| target logp | -1.079521 | -1.075494 | -1.079734 |
| target dLoss/dlogp | -0.0033917 | 0 | -0.0033910 |
| full gradient norm | 7.82968 | 7.81473 | 7.82327 |

After one matched SGD step, `distance(A,B)=1.10493e-5` and `distance(A,C)=4.98102e-6`; C is 54.9% closer to A. At step 5 the recovery is 61.0%. At step 20, `A-C/A-B=1.056`, so the repair advantage no longer persists as the three paths encounter new numerical and branch differences.

The 20-step confound audit finds average full-gradient norms only 2.7% higher on target-fork steps (`p=0.389`, exact descriptive permutation test). The normalized A/B gradient-norm gap is not higher on fork steps (ratio `0.925`, `p=0.852`). This does not validate the earlier 6.49x parameter-divergence jump ratio because the 20-step run saved parameter checkpoints only at steps 1, 5 and 20. The 6.49x result remains temporal coupling evidence, not a gradient-normalized causal jump estimate.

Supported: a repaired compile scheduling decision causally restores the target branch/gradient and moves the matched one-step update toward eager. Not supported: persistent 20-step alignment or proof that forks are the only source of trajectory divergence.

## RQ5: Testing Utility

The gated mutation catalog executes 15 independent model-level mutations over 512 aligned step-5 tokens each. It spans RMSNorm, RoPE, attention, MLP, cross-format materialization, lm_head and vocabulary reduction. All mutations have nonzero output canaries; the unmutated batch has zero clipping-branch mismatches against canonical certificates.

Across 7,680 mutation rows, 420 change the clipping branch. For the artificial-mutation label task:

| Method | Alarms | Precision | Recall |
|---|---:|---:|---:|
| fork signal | 425 | 0.9882 | 0.0547 |
| delta ranking at 425 alarms | 425 | 1.0000 | 0.0553 |
| `abs(delta)>0.1` | 576 | 1.0000 | 0.0750 |

The point estimates slightly favor delta ranking, so the preregistered expectation that fork would show higher precision is rejected. The paired table has 114 fork-only-correct and 124 delta-only-correct rows (`McNemar p=0.5597`), and cluster-bootstrap intervals for delta-minus-fork precision and recall both include zero. Token-level mutation-identification performance is therefore statistically indistinguishable on this dataset.

The same alarm budget exposes a different testing property: fork true positives cover 11/15 mutation families, while all 425 delta-ranking true positives come from `rmsnorm_no_upcast`. Delta ranking concentrates on the largest-amplitude failure; fork distributes a fixed triage budget across more operator families. This is artificial-family coverage, not evidence of finding more historical bugs.

The four initial-zero-clipping-fork mutations were then trained for 20 matched steps. All have zero branch forks at step 1 but nonzero one-step parameter distance, between 0.257 and 0.489 times the legal eager/compile distance. Every mutation produces its first delayed clipping fork by step 2 or 3, with 74–79 branch-fork events over 20 repeated-batch updates. The independent clean rerun is bitwise identical at steps 1, 5 and 20. Thus zero instantaneous fork does not imply training equivalence: numerical infection can first propagate through the continuous within-branch gradient and only later cross a decision boundary.

The latency result survives a held-out checkpoint/batch test, but not as a fixed step-2/3 constant. At step 14 / rollout batch 4, three mutations already cause 2, 2 and 1 clipping forks at the frozen state (latency zero). The remaining reverse-chunk logsumexp mutation still has zero initial fork, a nonzero one-step parameter distance, and its first clipping fork at step 2. Fork latency is therefore state-conditioned.

On the exact 512-token step-5 batch where all four mutations survived clipping, common-random-number sampling finds first-draw sampled-token forks for three mutation types: 10 states for FP16 RoPE phase, 14 for decoder-layer BF16 round-trip, and 1 for FP16 log-softmax. The reverse-chunk logsumexp mutation has none. All sampling self-runs are exact. This supplies a concrete cross-decision propagation result: surviving one decision boundary does not imply survival of another.

Two upstream PyTorch wrong-result issues were replayed twice. `pytorch/pytorch#186577` reproduces exactly on the current T4 nightly: eager equals `aot_eager`, while Inductor differs by up to `20.703125`. It does not alter the preregistered argmax, top-16 set or fixed threshold counts. `#183986` now fails closed with an explicit expanded-alias write error instead of silently returning a wrong result. The first result is particularly important for RQ5: a real large-amplitude bug can be caught by delta while missing the observed semantic decisions.

Held-out task evaluation compares the step-14 initial checkpoint and clean/mutation 20-step checkpoints on arithmetic prompts 64-127, using the original Phase-0 numeric reward. Clean versus mutation has 5/64 generated-sequence forks, 3/64 reward differences and 3/64 exact-answer outcome forks. The mean reward difference is `+0.01568` (mutation minus clean), with paired bootstrap 95% CI `[-0.05418, 0.08718]`; task-outcome existence is supported, but no average direction is significant.

The label contract matters: the five audited natural eager/compile forks are negatives for “artificial mutation identification,” so they count as false alarms there even though they are true semantic-risk events. These artificial mutations are not historical or certified bugs, and parameter divergence is not task-level harm.

## Failed Amplitude Certification

The Phase 2 differential probability assembly produces a diagnostic logprob bound near `2.63e5` versus empirical cross-path p99 `0.0201`, a ratio near `1.30e7`. More importantly, source-order, local independence and complete eager-to-compiled arithmetic mapping assumptions are unverified. The bound is therefore not `analytic_legal` and cannot classify bugs.

The compiled graph audit finds 706 kernel calls for the exact shape, including fused reduction/transcendental/materialization templates plus external MM/BMM calls. This explains why a global amplitude proof is currently infeasible and motivates the decision-level oracle, but it does not turn empirical envelopes into certificates.

## Threats To Validity

**Operator findability:** 5/5 natural clipping forks have sufficient singleton configuration-class interventions; 0/5 have unique operator attribution. Two cases rely on an exactly validated deterministic state reconstruction rather than an originally frozen checkpoint.

**Fork-rate definability:** rates are well-defined only conditional on model, checkpoint distribution, algorithm boundary, dtype, hardware and path pair. Five positives in three rollout groups are insufficient for a universal rate.

**Attribution correctness:** all replayable cases have 24/24 setting coverage, fresh caches, generated-code canaries and consistent repeats. However, settings change scheduling globally, so they establish sufficient configuration causes rather than a minimal source instruction.

**External validity:** all core evidence is T4 FP16. T4 lacks native BF16 and FlashAttention support. The new task reward is a synthetic held-out arithmetic task, not a general capability benchmark. The HF-vLLM experiment is a composite stack comparison with large deltas and 139 aligned clipping branch changes, not an operator attribution. Ampere-or-newer BF16/FlashAttention/vLLM V1 replication remains outstanding.

A fail-closed native-BF16 replay bundle is now complete. It checks both SM80+ capability and PyTorch BF16 support, records the actual training dtype in every online row, regenerates the canonical 300-step margin/online-fork artifacts, and audits that no FP16 path is mislabeled. On the visible T4 it exits before model loading with `native BF16 requires SM80 or newer, got SM75`; therefore this is execution readiness, not BF16 evidence.

## Claim Matrix

Current evidence supports:

- audited natural clipping forks and common-random-number sampled-token forks;
- direct token-gradient semantics and one-step parameter-update consequence;
- singleton Inductor scheduling interventions that eliminate all five replayable clipping forks;
- conditional risk-budget prediction, with margin as the strongest tested ranking feature;
- a bounded testing result: fork is not a universal mutation classifier, but provides broader mutation-family coverage and exposes delayed decision forks after continuous drift;
- cross-decision evidence: three clipping-surviving mutation types produce actual common-random-number sampling-token forks;
- finite task-level consequence: clean/mutation trajectories change three held-out exact-answer/reward outcomes, without a significant average reward direction;
- one reproducible upstream Inductor wrong-result bug and one historical case that now fails closed.

Current evidence does not support:

- natural forks as implementation bugs;
- complete `stable/fragile/bug` certification;
- unique operator root causes;
- persistent long-horizon repair;
- signed global bias;
- BF16/newer-GPU generalization;
- reward or advantage consequences from the 29 natural-path sampling forks;
- native BF16 behavior or a general task-quality degradation claim.

## Authoritative Artifacts

- Natural forks: `results/baseline_manifest.json`, `results/phase4_certificates.jsonl`, `reports/fork_cases.md`
- Attribution: `results/phase9_attribution_interaction.json`, `reports/phase9_attribution_interaction.md`
- Single-step consequence: `results/matched_step_fusion_r4/clip-step5-grpo_000001_2817771126c0-t80.json`
- Trajectory: `results/trajectory_step5_fusion/merged.json`, `reports/trajectory_analysis.md`
- Confound audit: `results/phase8_matched_step_counterfactual.json`, `reports/phase8_matched_step_counterfactual.md`
- Sampling: `results/phase8_sampling_actual_certificates.jsonl`, `reports/phase8_sampling_actual.md`
- Predictor: `results/phase8_risk_summary.json`, `reports/phase8_risk_predictor.md`
- Mutations: `results/phase9_mutations_gated/summary.json`, `reports/phase9_mutation_evaluation_gated.md`
- Testing utility: `results/phase9_testing_utility_gated.json`, `reports/phase9_testing_utility_gated.md`
- RQ5 significance: `results/phase9_testing_utility_significance.json`, `reports/phase9_testing_utility_significance.md`
- Zero-initial-fork trajectories: `results/phase10_zero_fork_mutation_trajectories.json`, `reports/phase10_zero_fork_mutation_trajectories.md`
- Held-out latency: `results/phase11_heldout_latency.json`, `reports/phase11_heldout_latency.md`
- Mutation sampling propagation: `results/phase12_mutation_sampling_gated/summary.json`, `reports/phase12_mutation_sampling_gated.md`
- Historical bug replays: `results/phase13_historical_bug_replays.json`, `reports/phase13_historical_bug_replays.md`
- Held-out task reward: `results/phase14_task_reward.json`, `reports/phase14_task_reward.md`
- Extension synthesis: `reports/phase11_14_extension_summary.md`
- Native BF16 replay protocol: `reports/phase15_bf16_external_protocol.md`
- Native BF16 current-hardware gate: `results/bf16_external/preflight_t4.json`
- Bound downgrade: `results/phase2_v2.json`, `reports/phase2_v2.md`
