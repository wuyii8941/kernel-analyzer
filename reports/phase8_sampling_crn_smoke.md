# Phase 7 Sampling Truncation Forks

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- same fixed response tokens: PASS
- same checkpoint and MATH attention backend: PASS
- warmed compile path: PASS
- two candidate-set self runs per path: PASS
- theoretical legal bound available: FAIL; regions remain unknown

## Delta Self Control
Candidate-set self mismatches: 0.

## External Validity
This scan uses the exact step-5 T4 FP16 snapshot. Temperature scaling is applied before truncation; generation-engine-specific processed-logit paths require separate replication.

## Summary
| samples | tokens | top_k | top_p | temperature | draws_per_state | state_draw_trials | self_candidate_set_failures | top_k_actual_forks | top_p_actual_forks | top_k_sampling_fork_states | top_p_sampling_fork_states | top_k_sampling_fork_draws | top_p_sampling_fork_draws | top_k_min_margin | top_p_min_margin | top_p_count_mean_ref | top_p_count_p99_ref | all_regions_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 50 | 0.9 | 1.0 | 64 | 8192 | 0 | 8 | 0 | 11 | 7 | 14 | 8 | 0.0 | 0.0006923437118530051 | 2.15625 | 13.380000000000024 | True |
