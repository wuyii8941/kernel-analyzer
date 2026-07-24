# Phase 6 Gradient Contribution

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- actual_forks_present: PASS
- full_autograd: PASS
- proxy_matches_clipping_semantics: PASS

## Delta Self Control
Uses certificates generated after Phase 1 self-consistency checks.

## Summary
Gradient contribution fields were added to certificates.

The earliest canonical online fork was deterministically replayed from the original 300-step schedule. At pre-minibatch step 5, all 512 replay keys, old logprobs, eager new logprobs, and advantages matched the online run exactly. The saved snapshot was then loaded as FP32 master weights with FP16 autocast, training mode, gradient checkpointing, the original four-sample batch shape, and MATH-locked SDPA.

Both recomputed branch logprobs matched the certificate within the enforced `1e-6` tolerance. The positive-advantage token is unclipped on ref and has global per-token gradient norm `413.711243`; it is clipped on alt and has norm `0`. The contribution difference is `413.711243`.

The case remains region `unknown` because no usable theoretical B exists. This gradient result proves decision-level optimization-semantic divergence, not a fragile-versus-bug classification.

## External Validity
This result is from T4 FP16. It demonstrates the fork-to-gradient mechanism in FP16; a production BF16 replication remains required.

## Gradient Evidence
| n_certificates | n_actual_forks | forks_with_nonzero_grad_diff | mode |
| --- | --- | --- | --- |
| 1 | 1 | 1 | hf_autograd |
