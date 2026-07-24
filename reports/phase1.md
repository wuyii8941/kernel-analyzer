# Phase 1 Logprob Pipeline

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- fixed_response_tokens: PASS
- token_alignment_checked: PASS
- same_weights_config_expected: PASS
- model_weight_fingerprint_match: PASS
- deterministic_env_recorded: PASS
- warn_only_messages_recorded: PASS
- delta_self_ref_gate: PASS
- delta_self_alt_gate: PASS
- sample_and_token_scale_gate: PASS

## Delta Self Control
ref p99=0, alt p99=0, cross p50=4.69787e-05.

Phase A1 replaced the original same-process check with independent-process runs. Eager FP16, compile FP16, eager-attention FP16, and SDPA-MATH FP16 were bitwise identical across separate CUDA contexts on the same T4 and across physical T4 devices. Compile was also bitwise identical with independent cold Inductor caches. See `reports/phaseA1_self_audit.md`.

## External Validity
These measurements use T4 FP16 (`u approximately 4.9e-4`) because T4 has no native BF16 support. A fork observed in FP16 is strong evidence that the mechanism remains relevant at BF16's coarser unit roundoff; a zero-fork FP16 result is scoped to FP16 and cannot exclude BF16 forks.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 51200 | 0.002075606528277677 | 4.697870463132858e-05 | 0.012609142065048217 | 0.02150791049003602 | 0.08493077754974365 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 12800 | 0.0022889074280619485 | 0.0001890920102596283 | 0.012968468666076653 | 0.019548903107643144 | 0.06743764877319336 |
| 32-63 | 12800 | 0.0014507247655820483 | 3.956258296966553e-06 | 0.010460686683654779 | 0.01727814316749573 | 0.07720017433166504 |
| 64-95 | 12800 | 0.002294452871580146 | 5.326222162693739e-05 | 0.013245892524719236 | 0.02302632093429566 | 0.061646461486816406 |
| 96-127 | 12800 | 0.0022683410478865663 | 6.177835166454315e-05 | 0.013040366768836966 | 0.025568722486496 | 0.08493077754974365 |

## Warn Only Messages
- `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.

<!-- phaseA3:start -->
## Phase A3 Checkpoint-State Audit

- aligned step-272 batch checkpoint-state alignment: PASS
- original 51,200-token historical convolution alignment: FAIL
- Phase 1 recorded/current model hash: PASS
- Phase 4 authorization: aligned 512-token batch only; full scan requires per-rollout online state alignment

| item | value |
| --- | --- |
| Phase 1 model | data/phase0_policy_final |
| Phase 1 model SHA256 | 55fd2089d10c96a7bee1c38143ecd155d14bccf96694f4957b61c068402ebe04 |
| Phase 1 inferred step | 300 |
| Iteration-2 step range | 2..299 (stride 3) |
| Saved checkpoints | 240,270,300 |
| Exact intersection | none |
| Aligned repair snapshot | step 272, policy iteration 2 |
| Aligned repair model SHA256 | cec8eeb32f34205c2953dfc0cdb2d36d4e8fdfc43ca12369ef0e4a98eff8281f |
| Aligned repair scale | 4 cases / 512 tokens |

Phase 1 used the final step-300 root model, while iteration-2 margins were measured at pre-minibatch steps 2,5,...,299; no exact iteration-2 checkpoint was saved.

Required repair: The aligned step-272 repair is complete. For a full 51,200-token scan, measure each rollout's delta online at its own policy_iteration=2 pre-minibatch state; do not reuse the final checkpoint.
<!-- phaseA3:end -->
