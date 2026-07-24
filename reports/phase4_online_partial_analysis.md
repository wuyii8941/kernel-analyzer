# Phase 4 Online State-Aligned Analysis

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- every row marked online_state_aligned: PASS
- every row is policy_iteration=2 pre-minibatch: PASS
- attention backend locked to MATH: PASS
- online self gate: PASS
- expected token coverage: INCOMPLETE

## Delta Self Control
ref p99=0, alt p99=0, cross p50=0.000183105.

## External Validity
This online scan runs on T4 FP16. Zero forks cannot exclude BF16 behavior; an FP16 fork demonstrates the mechanism but BF16 still requires replication.

## Summary
| tokens | expected_tokens | coverage_complete | rollouts | self_ref_p99 | self_alt_p99 | cross_p50 | self_gate | near_boundary_tokens_margin_lt_1e_2 | fork_possible_count | actual_fork_count | actual_fork_rate | independent_convolution_predicted_rate | independent_convolution_predicted_count | observed_minus_predicted_count | pearson_log_margin_vs_log_delta | signed_delta_mean_all | signed_delta_mean_positive_advantage | signed_delta_mean_negative_advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 51200 | False | 8 | 0.0 | 0.0 | 0.00018310546875 | True | 36 | 15 | 5 | 0.001220703125 | 0.001866757869720459 | 7.646240234375 | -2.646240234375 | -0.15521374035513857 | 3.0277296900749207e-06 | 5.762403210004171e-05 | -2.9730051755905153e-05 |

## Conditional Delta
| group | tokens | delta_mean | delta_p50 | delta_p99 | delta_max |
| --- | --- | --- | --- | --- | --- |
| margin_lt_1e-2 | 36 | 0.00406238767835829 | 0.0023593902587890625 | 0.013951587677001949 | 0.014806747436523438 |
| margin_ge_1e-2 | 4060 | 0.0021131839658239207 | 0.00017833709716796875 | 0.015958290100097625 | 0.030731201171875 |

## By Rollout
| rollout_batch | optimizer_step | tokens | near_boundary | fork_possible | actual_forks |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 512 | 6 | 2 | 0 |
| 1 | 5 | 512 | 9 | 3 | 1 |
| 2 | 8 | 512 | 4 | 1 | 0 |
| 3 | 11 | 512 | 2 | 3 | 2 |
| 4 | 14 | 512 | 4 | 3 | 2 |
| 5 | 17 | 512 | 2 | 1 | 0 |
| 6 | 20 | 512 | 4 | 1 | 0 |
| 7 | 23 | 512 | 5 | 1 | 0 |
