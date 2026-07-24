# Phase 4 Natural Scan

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- fixed_response_tokens: PASS
- real_old_logp_present: PASS
- rollout_state_matches: PASS
- rollout_token_id_matches: PASS
- advantage_sign_present_or_zero_marked_not_applicable: PASS
- same_token_comparison: PASS
- actual_forks_need_manual_confounds: required for every actual_fork before claim

## Delta Self Control
Uses delta_self fields from Phase 1; inspect Phase 1 report for the hard gate.

## Summary
Natural scan generated v2 certificates with clipping branch decisions.

## Rates
| n_certificates | n_applicable_decisions | n_zero_advantage_not_applicable | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_not_applicable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0 | 512 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 1.0 |

## Minimum Actual Fork Case
```json
_No actual fork found._
```

## Fragile Set Entropy
| fragile_tokens | entropy_ref_p50 | entropy_ref_p95 | entropy_ref_mean |
| --- | --- | --- | --- |
| 0 | None | None | None |

## Fragile Set Breakdown
_No rows._

## Global Attribution Context
_No rows._
