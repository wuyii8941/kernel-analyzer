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
| samples | tokens | top_k | top_p | temperature | self_candidate_set_failures | top_k_actual_forks | top_p_actual_forks | top_k_min_margin | top_p_min_margin | top_p_count_mean_ref | top_p_count_p99_ref | all_regions_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1024 | 50 | 0.9 | 0.7 | 0 | 75 | 14 | 0.0 | 7.486343383766858e-06 | 2.6748046875 | 15.0 | True |
