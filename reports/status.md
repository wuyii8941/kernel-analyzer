# ForkCert Current Status

## Environment

- Model: Qwen/Qwen3-0.6B; Qwen2.5 is not used.
- Canonical Python: `/data1/tzh/conda-envs/forkcert/bin/python`.
- Hardware: 14 Tesla T4 GPUs. T4 lacks native BF16, so canonical measurements use FP16 compute with FP32 master weights.
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, deterministic algorithms warn-only, cuDNN benchmark disabled.

## Current Result

The core decision-fork hypothesis is supported for the warmed eager-vs-compile FP16 pair. A canonical 300-step online GRPO run measured both paths at each rollout's own `policy_iteration=2` pre-minibatch state.

- 51,200 total token records over 100 rollout batches and 400 responses.
- 39,936 applicable nonzero-advantage clipping decisions; 11,264 zero-advantage rows marked not applicable.
- 165 tokens with clipping margin below `1e-2`.
- 15 fork-possible tokens and 5 actual clipping branch forks.
- Empirical independent-convolution prediction: 3.758 forks; observed: 5.
- Both online self runs have exact-zero max and p99 deltas.
- All five actual forks pass the structured tokenizer/weights/tokens/mask/position/dtype/backend/old-logp/advantage/determinism checklist.
- All five forks are replayable from frozen or exactly validated reconstructed states and have complete 24-setting Inductor matrices. Each has at least one singleton scheduling intervention that eliminates the branch fork; this is configuration-class, not unique-operator, attribution.
- RQ5 paired testing differences are not significant (`McNemar p=0.5597`). At 425 alarms, fork covers 11/15 mutation families versus 1/15 for delta ranking. Four initial-zero-clipping-fork mutations all diverge continuously after one update and create delayed clipping forks at step 2 or 3.
- Held-out step14/batch4 shows fork latency is state-conditioned: three mutations fork immediately, while reverse-chunk logsumexp remains initially stable and forks at matched update step 2.
- On the exact 512-token clipping-survival batch, three of four mutations produce common-random-number actual sampling-token forks (10, 14 and 1 first-draw states); all self-runs are exact.
- PyTorch `#186577` is independently reproduced twice (`max delta=20.703125`) but does not cross the preregistered output decisions; `#183986` now fails closed.
- Held-out arithmetic evaluation has 5/64 clean-vs-mutation completion forks and 3/64 reward/exact-answer forks; mean reward direction is not significant.
- Native-BF16 external replay is packaged but not executed: the fail-closed entry point rejects the visible Tesla T4 at SM75 before model loading. The portable protocol, runner, hardware gate and result audit are complete; an Ampere-or-newer GPU is still required for evidence.
- Regions remain `unknown`: Phase 2 does not provide a usable theoretical legal bound, so no fork is labelled fragile or bug.

## Phase Status

| phase | status | authoritative evidence |
| --- | --- | --- |
| Phase 0 | Complete on Qwen3 T4 FP16 | `phase0.md`, 153,600 margin rows; iteration-2 near-boundary rate 0.413% |
| Phase 1 | Complete | 51,200-token claim pairs, independent process/CUDA/GPU/cache self audit |
| Phase 1.5 | Complete first attribution cycle | canaries, redesigned L2, corrected controlled L3, warmed A4 independence audit |
| Phase 2 v2 | Downgraded after source audit | actual L6 artifact has 23 Triton templates/453 invocations + 253 GEMM/BMM calls; local legal contracts and eager mapping are absent; L4 is a no-op |
| Phase 3 | Complete with weak held-out calibration | detector construction passes; 3.758 is in-sample, while leave-one-rollout-out signed calibration predicts 3.154 versus 5 observed |
| Phase 4 | Complete for warmed compile FP16 pair | 51,200 certificates, 5 actual forks, all case checklists pass |
| Phase 5 | Empirical complete; certified conditional | three altered operations trigger 313/104/686 clip forks; all regions unknown without B |
| Phase 6 gradient | Complete | exact step-5 replay; unclipped gradient norm 413.711 versus clipped norm 0 |
| Phase 6 twin | Mechanism evidence, causal rate revised | one-step A/B/C repair is 54.9% closer to eager; checkpoints at only steps 1/5/20 cannot validate the old 6.49x per-step jump claim; repair advantage is gone by step 20 |
| Phase 7 | Partial | common-random-number sampling has 17 top-k and 12 top-p actual first-draw token forks; step-5 gradient-clip pair has no natural trigger fork (`7.82968` vs `7.81473` at threshold `1.0`); KL/MoE/optimizer applicability audited |

## Main Artifacts

- `reports/plan_traceability.md`: original Phase 0-7 plus A0-A5/P1-P8 requirements.
- `reports/phase4.md`: canonical natural scan.
- `reports/fork_cases.md`: five fork certificates and complete confound checklists.
- `results/phase4_online_full_enriched.jsonl`: online state-aligned path measurements.
- `results/phase4_certificates.jsonl`: 51,200 v2 certificates.
- `reports/phase2_v2.md`: legal-bound downgrade and empirical-envelope separation.
- `reports/phase2_logsoftmax_bound.md`: vendor-documented conditional near-output bound and its legal-contract limitation.
- `reports/phase2_logsoftmax_coverage.md`: hypothetical half-input portability calculation, explicitly not canonical evidence.
- `reports/phase2_logsoftmax_online_smoke.md`: canonical Trainer L4 no-op canary.
- `reports/phase2_compile_graph_audit.md`: exact FX/Inductor graph, profile, hashes, and logits delta.
- `reports/phase2_compile_source_inventory.md`: 706-call compiled numerical source inventory.
- `reports/phase6_grad_step5.md`: exact fork-token autograd result.
- `reports/phase6_twin_stepwise.md`: full-model fork/no-fork divergence comparison.
- `reports/phase7_gradclip.md`: paired full-model gradient-clipping trigger audit and 300-step eager margin coverage.
- `reports/phase7_remaining_decisions.md`: strict applicability/dependency audit for the remaining Phase 7 decisions.
- `reports/final_results_summary.md`: current RQ1-RQ5 result and claim matrix.
- `reports/phase9_attribution_interaction.md`: 24/24 setting matrices and singleton repairs for all five natural clipping forks.
- `reports/phase9_testing_utility_significance.md`: paired significance and mutation-family coverage.
- `reports/phase10_zero_fork_mutation_trajectories.md`: continuous drift followed by delayed decision forks.
- `reports/rq5_reframing_and_related_work.md`: revised oracle hierarchy and related-work positioning.

## Remaining Work

1. Execute `./run_phase15_bf16_external.sh` once on a native-BF16 Ampere-or-newer GPU. The runner is fail-closed and audited; current T4 evidence is only `results/bf16_external/preflight.json` with `passed=false` at SM75.
2. Obtain operator-input norms, exact differential reduction structures, and justified local independence before calling Phase 2 `semi_certified` or enabling certified bug classification.
3. Continue Phase 7 generation-engine processed logits after installing a supported engine; KL early-stop needs a new thresholded recipe and MoE routing needs a MoE model. No discrete optimizer threshold exists in the canonical AdamW recipe.
4. Phase 5 may run only as empirical anomaly detection until a legal theoretical B exists.
5. Unique-operator attribution still requires an earliest successful intermediate splice or a narrower source-level intervention; global Inductor settings establish only sufficient scheduling causes.

## Native BF16 Replay Bundle

- Protocol: `reports/phase15_bf16_external_protocol.md`
- Entry point: `run_phase15_bf16_external.sh`
- Fixed config: `configs/phase15_bf16_grpo.yaml`
- Current-hardware blocker: `results/bf16_external/preflight_t4.json`
- Current full-run gate: `results/bf16_external/preflight.json`
- Verification: shell syntax and Python compilation pass; full repository suite is 78/78 passing.
