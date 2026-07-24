# Phase 2 Bound Candidate Audit

## Confound Checklist
- top_sources_from_attribution: required by input sources file
- analytic_legal_sources: FAIL
- activation_norms_measured: required by input sources file
- probability_delta_recorded: 1e-06
- logsumexp_lipschitz_applied: PASS

## Delta Self Control
Refer to Phase 1 report; Phase 2 consumes only cross-path empirical deltas.

## Summary
DOWNGRADE: source file is not an analytic legal-error certificate; observed-delta heuristics cannot classify fragile versus bug.

## Tightness
| source_count | certificate_kind | legal_source_validation | logprob_bound_prob | logprob_bound_worst | empirical_delta_p99 | tightness_prob_over_p99 | tightness_worst_over_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | unverified | False | 0.518956834128431 | 0.518956834128431 | 0.0002 | 2594.7841706421545 | 2594.7841706421545 |

## Per Source
| name | mechanism | dtype | reduction_length | sum_abs | reduction | materialization_count_delta | local_scale | propagation | reduction_path_count | assumptions_verified | algorithm_order_known | input_norm_measured | propagation_certified | notes | activation_bound_worst | activation_bound_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_log_softmax_vocab_reduction | reduction_order | fp32 | 128000 | 128000.0 | tree | 0 | 0.0 | 1.0 | 2 | False | False | False | False | None | 0.25939967690673515 | 0.25939967690673515 |
| elementwise_materialization_chain | materialization_points | bf16 | 1 | 0.0 | tree | 2 | 0.01 | 1.0 | 2 | False | False | False | False | None | 7.874015748031496e-05 | 7.874015748031496e-05 |
