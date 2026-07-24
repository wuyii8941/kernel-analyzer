# Phase 4 Online State-Aligned Analysis

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- every row marked online_state_aligned: PASS
- every row is policy_iteration=2 pre-minibatch: PASS
- attention backend locked to MATH: PASS
- online self separation: PASS
- expected token coverage: PASS

## Delta Self Control
ref p99=0, alt p99=0, cross p50=0. When cross p50 is zero, the original strict-ratio scalar gate is degenerate; pass requires both self maxima to be exactly zero and at least one nonzero cross delta.

## External Validity
This online scan runs on T4 FP16. Zero forks cannot exclude BF16 behavior; an FP16 fork demonstrates the mechanism but BF16 still requires replication.

## Summary
| tokens | total_rows | expected_tokens | coverage_complete | rollouts | rollouts_with_applicable_decisions | zero_advantage_rows | self_ref_p99 | self_alt_p99 | cross_p50 | self_gate | self_gate_rule | original_strict_ratio_gate_degenerate | near_boundary_tokens_margin_lt_1e_2 | fork_possible_count | actual_fork_count | actual_fork_rate | independent_convolution_predicted_rate | independent_convolution_predicted_count | observed_minus_predicted_count | pearson_log_margin_vs_log_delta | signed_delta_mean_all | signed_delta_mean_positive_advantage | signed_delta_mean_negative_advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 39936 | 51200 | 51200 | True | 100 | 78 | 11264 | 0.0 | 0.0 | 0.0 | True | exact_zero_self_with_nonzero_cross | True | 165 | 15 | 5 | 0.0001252003205128205 | 9.409586588541594e-05 | 3.757812499999971 | 1.2421875000000289 | -0.0745880007541203 | 3.1053637846922265e-07 | 3.77862505574044e-06 | -4.609310349752736e-06 |

## Conditional Delta
| group | tokens | delta_mean | delta_p50 | delta_p99 | delta_max |
| --- | --- | --- | --- | --- | --- |
| margin_lt_1e-2 | 165 | 0.0008863391298236269 | 0.0 | 0.012310943603515624 | 0.014806747436523438 |
| margin_ge_1e-2 | 39771 | 0.00021572318778117515 | 0.0 | 0.00781364440917981 | 0.030731201171875 |

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
| 8 | 26 | 512 | 0 | 0 | 0 |
| 9 | 29 | 512 | 4 | 0 | 0 |
| 10 | 32 | 512 | 3 | 0 | 0 |
| 11 | 35 | 512 | 3 | 0 | 0 |
| 12 | 38 | 512 | 4 | 0 | 0 |
| 13 | 41 | 512 | 3 | 0 | 0 |
| 14 | 44 | 512 | 3 | 0 | 0 |
| 15 | 47 | 512 | 4 | 0 | 0 |
| 16 | 50 | 512 | 3 | 0 | 0 |
| 18 | 56 | 512 | 3 | 0 | 0 |
| 19 | 59 | 512 | 3 | 0 | 0 |
| 20 | 62 | 512 | 2 | 0 | 0 |
| 21 | 65 | 512 | 3 | 0 | 0 |
| 22 | 68 | 512 | 2 | 0 | 0 |
| 23 | 71 | 512 | 1 | 0 | 0 |
| 24 | 74 | 512 | 2 | 0 | 0 |
| 25 | 77 | 512 | 4 | 0 | 0 |
| 26 | 80 | 512 | 2 | 0 | 0 |
| 27 | 83 | 512 | 4 | 0 | 0 |
| 28 | 86 | 512 | 2 | 0 | 0 |
| 29 | 89 | 512 | 1 | 0 | 0 |
| 30 | 92 | 512 | 2 | 0 | 0 |
| 31 | 95 | 512 | 2 | 0 | 0 |
| 32 | 98 | 512 | 2 | 0 | 0 |
| 33 | 101 | 512 | 1 | 0 | 0 |
| 34 | 104 | 512 | 3 | 0 | 0 |
| 35 | 107 | 512 | 4 | 0 | 0 |
| 36 | 110 | 512 | 0 | 0 | 0 |
| 38 | 116 | 512 | 3 | 0 | 0 |
| 39 | 119 | 512 | 1 | 0 | 0 |
| 40 | 122 | 512 | 7 | 0 | 0 |
| 41 | 125 | 512 | 5 | 0 | 0 |
| 42 | 128 | 512 | 0 | 0 | 0 |
| 43 | 131 | 512 | 3 | 0 | 0 |
| 44 | 134 | 512 | 1 | 0 | 0 |
| 45 | 137 | 512 | 0 | 0 | 0 |
| 46 | 140 | 512 | 2 | 0 | 0 |
| 47 | 143 | 512 | 1 | 0 | 0 |
| 48 | 146 | 512 | 2 | 0 | 0 |
| 49 | 149 | 512 | 3 | 0 | 0 |
| 50 | 152 | 512 | 2 | 0 | 0 |
| 51 | 155 | 512 | 4 | 0 | 0 |
| 52 | 158 | 512 | 3 | 0 | 0 |
| 54 | 164 | 512 | 0 | 0 | 0 |
| 55 | 167 | 512 | 1 | 0 | 0 |
| 56 | 170 | 512 | 2 | 0 | 0 |
| 57 | 173 | 512 | 4 | 0 | 0 |
| 58 | 176 | 512 | 1 | 0 | 0 |
| 59 | 179 | 512 | 0 | 0 | 0 |
| 60 | 182 | 512 | 1 | 0 | 0 |
| 62 | 188 | 512 | 1 | 0 | 0 |
| 65 | 197 | 512 | 0 | 0 | 0 |
| 66 | 200 | 512 | 1 | 0 | 0 |
| 68 | 206 | 512 | 1 | 0 | 0 |
| 69 | 209 | 512 | 1 | 0 | 0 |
| 70 | 212 | 512 | 0 | 0 | 0 |
| 72 | 218 | 512 | 1 | 0 | 0 |
| 73 | 221 | 512 | 0 | 0 | 0 |
| 74 | 224 | 512 | 2 | 0 | 0 |
| 75 | 227 | 512 | 1 | 0 | 0 |
| 76 | 230 | 512 | 1 | 0 | 0 |
| 80 | 242 | 512 | 1 | 0 | 0 |
| 82 | 248 | 512 | 2 | 0 | 0 |
| 85 | 257 | 512 | 0 | 0 | 0 |
| 86 | 260 | 512 | 0 | 0 | 0 |
| 87 | 263 | 512 | 0 | 0 | 0 |
| 88 | 266 | 512 | 1 | 0 | 0 |
| 95 | 287 | 512 | 0 | 0 | 0 |
| 98 | 296 | 512 | 0 | 0 | 0 |
| 99 | 299 | 512 | 0 | 0 | 0 |
