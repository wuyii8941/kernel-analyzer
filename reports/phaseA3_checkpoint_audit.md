# Phase A3 Checkpoint-State Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- exact checkpoint state shared by aligned repair margin and delta: PASS
- original pooled 51,200-token state alignment: FAIL
- Phase 1 model hash verified: PASS

## Delta Self Control
Orthogonal; process independence is handled by Phase A1.

## External Validity
State alignment is precision-independent. Numerical results remain scoped to T4 FP16; a zero-fork result cannot exclude BF16 forks.

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
