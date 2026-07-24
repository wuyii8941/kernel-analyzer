# ForkCert v2 Plan Traceability

This file preserves the original Phase 0-7 scope and the later A/P corrections. A phase is complete only when its stated evidence and gates are satisfied; the presence of a script or example output is not completion.

## Original Phases

| phase | original requirement | current evidence | status / remaining gate |
| --- | --- | --- | --- |
| Phase 0 | Real PPO/GRPO multi-minibatch margin distribution; quantiles and threshold rates; save training state | Qwen3-0.6B GRPO, 300 steps, 153,600 rows; `reports/phase0.md` | Complete for T4 FP16; late iteration-2 `P(margin<1e-2)=0.00413161` |
| Phase 1 | Token-aligned dual-path logprobs, two self runs, environment fingerprint, >=50k tokens | 51,200-token compile and attention pairs; `reports/phase1.md` and `phase1_sdpa.md` | Complete for original pairs; A1 independently validated process/device/cache reproducibility |
| Phase 1.5 | L1-L6 one-variable ladder, activation injection/propagation, source attribution | `reports/phase15.md`; P1 canaries; redesigned L2; corrected L3; warmed A4 | Complete first attribution cycle: warmed MATH-locked compile self gates and independent-process checks pass; controlled probes remain labelled as such |
| Phase 2 | Computable legal bound, worst-case and probability forms, tightness decision | P4 assembly; Linear norms; L4 no-op canary; exact L6 graph and 706-call source inventory | Final downgrade on T4 pair: 23 Triton templates/453 calls plus 253 GEMM/BMM calls need eager mapping and local legal contracts; no certified regions |
| Phase 3 | Controlled fork detector test and `P(margin < delta)` calibration | `reports/phase3_online.md`, `reports/phase8_risk_predictor.md` | Detector complete; 3.758 is explicitly in-sample. Leave-one-rollout-out signed calibration predicts 3.154 versus 5 observed and is weaker than margin-only for ranking |
| Phase 4 | Natural scan with real old_logp/advantage/checkpoint, v2 certificates and confound audit | `reports/phase4.md`, `fork_cases.md`, 51,200 certificates | Complete for warmed compile FP16 pair: 5 actual forks, all confound checklists pass, regions unknown without B |
| Phase 5 | 3-5 silent bug injections and theoretical-bound confusion matrix | three altered operations executed; 313/104/686 clip forks | Empirical anomaly path complete; certified confusion matrix remains conditional on legal B |
| Phase 6 | Per-token gradient contribution and twin training | exact step-5 autograd, A/B/C matched update, and 20-step trajectory | Unclipped norm 413.71 vs clipped 0; repaired update is 54.9% closer after one step. Repair advantage disappears by step 20, and missing intermediate parameter checkpoints invalidate the old 6.49x per-step causal interpretation |
| Phase 7 | Sampling, MoE routing, KL stop, gradient clip, optimizer thresholds | common-random-number sampled-token scan; paired step-5 grad-clip audit; applicability audit | 17 top-k and 12 top-p actual first-draw forks; grad-clip natural pair has no fork; KL stop absent; dense Qwen3 cannot test MoE; no strict AdamW decision boundary identified |

## Batch A Audits

| item | required correction | evidence | status |
| --- | --- | --- | --- |
| A0 | Signed mean, uncertainty, advantage-sign split | `reports/phaseA0_signed_bias_online.md` | Complete on the authoritative same-state online pair: signed mean and advantage-sign interaction are not significant. The older pooled-state report is diagnostic only |
| A1 / P5 | Independent processes/CUDA contexts, cross-GPU, compile warm/cold caches | `reports/phaseA1_self_audit.md` | Complete for original paths; all compared outputs bitwise equal |
| A2 | Per-prompt near-boundary concentration | `reports/phaseA2_prompt_concentration.md` | Complete: not concentrated in 2-3 prompts |
| A3 | Margin and delta from the same checkpoint state | `reports/phaseA3_checkpoint_audit.md` plus canonical online scanner | Complete: the historical pooled convolution remains invalid, but canonical Phase 3/4 measures margin, both path deltas, and forks in each rollout's same in-memory iteration-2 state |
| A4 / P3 | Both paths SDPA-loaded and MATH-locked; eager vs compile only | cold diagnostic plus `phaseA4_compile_math_locked_warmed.md` | Complete: warmed full run passes self gates and two independent warmed processes are bitwise equal |
| A5 | Correct FP16->BF16->FP16 L3, controlled-probe label | `reports/phase15.md` | Complete; not treated as backend attribution |
| P1 | Canary before every attribution switch | `phase15_canaries*.md/jsonl` | Complete after L4/L6 canary corrections; failed attempts retained |
| P2 | RMSNorm no-upcast and submodule-compile probes | `reports/phase15.md` | Executed; no-upcast is catastrophic FP16 sensitivity, submodule compile remains provisional |
| P6 | Fixed FP16/BF16 external-validity statement | Phase A and updated phase reports | Required in every new report |
| P7 | Phase 4 hard gates are A1 and A3 | A1 passes; canonical online scanner satisfies A3 for all 51,200 rows | Complete before canonical Phase 4 interpretation |
| P8 | Store expected natural-fork scale and investigate 0 or >100 | Online-aligned prediction 3.758; observed 5 | Complete; result lies in the predeclared expected order rather than a 0 or >100 anomaly branch |

## Mathematical Discipline

- Theoretical and empirical envelopes remain separate. Only a valid theoretical legal bound may support a `bug` classification.
- Phase 2 v2 must use `sqrt(sum((local_probability_bound_l * propagation_gain_l_to_output)^2))`. Propagation gain remains intact inside each term.
- The resulting bound is `semi-certified`: certified local probability bounds plus empirically calibrated propagation gains.
- FP32-vs-FP16 is a debugging pair and cannot enter a claim.
- Controlled boundary construction validates the detector but cannot support a natural-fork claim.
- T4 FP16 uses unit roundoff approximately `4.9e-4`; zero forks on FP16 cannot exclude BF16 behavior.
