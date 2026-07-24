# Phase 2 Legal Error Bounds

## Confound Checklist
- top_sources_from_attribution: required by input sources file
- activation_norms_measured: required by input sources file
- probability_delta_recorded: 1e-06
- logsumexp_lipschitz_applied: PASS

## Delta Self Control
Refer to Phase 1 report; Phase 2 consumes only cross-path empirical deltas.

## Summary
VIOLATION: empirical p99(delta) exceeds probability bound; refine B, check confounds, or inspect bug candidates.

## Tightness
| source_count | logprob_bound_prob | logprob_bound_worst | empirical_delta_p99 | tightness_prob_over_p99 | tightness_worst_over_p99 |
| --- | --- | --- | --- | --- | --- |
| 2 | 3.366733860158783e-13 | 1.1718869209289551e-14 | 0.0002 | 1.6833669300793915e-09 | 5.859434604644776e-11 |

## Per Source
| name | mechanism | dtype | reduction_length | sum_abs | reduction | materialization_count_delta | local_scale | propagation | notes | activation_bound_worst | activation_bound_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase15_L1_attention_backend | algorithm_structure | fp32 | 4096 | 4.096e-09 | tree | 0 | 1e-12 | 1.0 | initial coarse source generated from Phase 1.5 measurements; refine before final certified claim | 2.9296875e-15 | 8.416831670164718e-14 |
| phase15_L2_RMSNorm_fused/unfused | materialization_points | fp32 | 4096 | 4.096e-09 | tree | 1 | 1e-12 | 1.0 | initial coarse source generated from Phase 1.5 measurements; refine before final certified claim | 2.9297471046447756e-15 | 8.416837630629196e-14 |
