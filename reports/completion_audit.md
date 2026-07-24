# ForkCert v2 Completion Audit

## Verdict

The original clipping-line mechanism experiment is complete on T4 FP16 and supports natural optimization-semantic forks. The broader ForkCert v2 program is not globally complete because a usable legal bound, certified bug classification, native BF16 replication, and Phase 7 extensions remain outstanding.

## Evidence Audit

| requirement | evidence | verdict |
| --- | --- | --- |
| Real multi-minibatch margins | 300-step TRL GRPO, 153,600 rows | Proven |
| Near-boundary demand | 0.413% of applicable iteration-2 decisions below `1e-2` | Proven |
| Self-run independence | independent process/CUDA context, cross-T4, warm/cold Inductor cache | Proven for Phase 1/A4 paths |
| Same-state margin/delta/fork | online measurement inside each iteration-2 pre-minibatch state | Proven for canonical Phase 4 |
| Token/state coverage | 51,200 rows, no token mismatch or missing rollout state | Proven |
| Natural actual forks | 5 branch flips among 39,936 applicable decisions | Proven for T4 FP16 warmed compile pair |
| Full confound checklist | all five cases pass `reports/fork_cases.md` | Proven |
| Detector calibration | controlled actual forks and predicted natural count 3.758 vs observed 5 | Proven |
| Signed bias | online cluster CI contains zero; advantage interaction not significant | No directional-bias claim |
| Fork gradient semantics | exact replay, norm 413.711 on unclipped branch and 0 on clipped branch | Proven for earliest fork |
| Matched-step consequence | exact token gradient plus A/B/C SGD update; repaired C is 54.9% closer to eager after one step | One-step causal contribution supported; old 6.49x per-step jump is unverified because intermediate parameter checkpoints are absent |
| Legal B | P4 code correct; exact L6 artifact inventory has 706 kernel calls, 13 numerical Triton templates, and no complete eager arithmetic map/local contracts. Isolated L4 is conditional and canonical-no-op | Strict analytic B not achieved; decision-tree downgrade is now source-audited |
| Conditional L4 exercise | standalone L4 envelope over 39,936 iteration-2 margins, followed by canonical Trainer canary | Hypothetical 99.2087% coverage retained for portability only; canonical switch is exact-zero no-op |
| Executed bug injections | off-by-one reduction, missing mask column, forced FP16 exp/sum/log | Empirical sensitivity proven; 313/104/686 clip forks |
| Certified bug confusion matrix | requires legal B | Not triggered; all injected rows remain unknown |
| BF16 external validity | T4 has no native BF16 | Missing replication |
| Phase 7 sampling decisions | common-random-number top-k/top-p sampling on 1,024 contexts per mechanism | Proven 17 top-k and 12 top-p actual first-draw token forks; reward/advantage consequences remain untested |
| Configuration attribution | all five natural forks, 24/24 Inductor settings each, valid generated-code canaries | Proven sufficient singleton scheduling interventions for 5/5; unique source operator remains 0/5 |
| Phase 7 gradient clipping | exact step-5 four-response replay; paired norms `7.82968`/`7.81473`, self deltas zero; 300-step eager minimum margin `1.0` | No natural trigger fork in paired batch; controlled midpoint fork validates detector; full paired trajectory not executed |
| Phase 7 remaining decisions | processed-engine sampling absent; `beta=0` means no KL stop; dense Qwen3 has no MoE routing; canonical AdamW has no strict threshold target | Applicability audited; external/model/recipe extensions remain |

## Claim Discipline

- Supported: execution-path mismatch can cross PPO clipping boundaries and switch a token's policy-gradient contribution on/off.
- Supported: five naturally occurring FP16 forks in the canonical online-aligned experiment.
- Supported: one repaired scheduling configuration restores the step-5 branch/gradient and moves a matched one-step update toward eager.
- Not supported: the old 6.49x result as a gradient-normalized per-step parameter-jump estimate or persistent 20-step repair.
- Not supported: classification of these cases as `fragile` or `bug`.
- Not supported: a systematic signed logprob bias in the online training population.
- Not supported: a zero-fork or rate claim for production BF16 hardware.

## Verification

- Unit suite: 61 tests pass.
- No result uses FP32-vs-FP16 as a claim pair.
- Controlled construction is labelled calibration only.
- Empirical envelopes are explicitly prohibited from bug classification.
- Failed/no-op canary attempts and cold compile state effects remain reported rather than removed.
