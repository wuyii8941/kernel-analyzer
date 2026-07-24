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
This scan uses the exact step-5 T4 FP16 snapshot. BF16 and generation-engine processed-logit paths require separate replication.

## Summary
| samples | tokens | top_k | top_p | self_candidate_set_failures | top_k_actual_forks | top_p_actual_forks | top_k_min_margin | top_p_min_margin | top_p_count_mean_ref | top_p_count_p99_ref | all_regions_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1024 | 50 | 0.9 | 0 | 75 | 35 | 0.0 | 1.9907951355202513e-06 | 11.240234375 | 148.69999999999982 | True |

Top-k candidate sets differ at 75/1,024 decision points (7.32%); every difference is exactly one candidate entering and one leaving. Top-p sets differ at 35/1,024 points (3.42%), with mean symmetric difference 1.49 and maximum 4. Only one token forks under both truncation rules.

The zero minimum top-k margin shows that exact boundary ties occur; backend-dependent tie resolution is itself a discrete candidate-set ambiguity but cannot be classified without a legal bound and an explicit tie policy. Top-p has a strictly positive minimum cumulative-probability margin of `1.99e-6`.

The baseline is a teacher-forced raw-logit candidate-set result. The sweep below adds explicit temperature processing; generation-engine-specific penalties and sampling implementations remain outside this run.

## Temperature Sweep

| temperature | top-p mean candidate count | top-p p99 candidate count | top-p forks | fork rate | minimum probability margin |
| --- | --- | --- | --- | --- | --- |
| 0.7 | 2.675 | 15.0 | 14 | 1.37% | 7.49e-6 |
| 1.0 | 11.240 | 148.7 | 35 | 3.42% | 1.99e-6 |
| 1.3 | 217.585 | 4049.45 | 270 | 26.37% | 8.34e-8 |

Flattening the distribution increases candidate-set size, shrinks the cumulative-probability boundary margin, and sharply raises the observed top-p fork rate. Positive temperature scaling preserves logit rank, so top-k remains 75/1,024 at all three temperatures. This sweep covers explicit temperature processing but not engine-specific penalties or raw-versus-processed logprob APIs.
