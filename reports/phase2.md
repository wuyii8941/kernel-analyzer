# Phase 2 Bound Candidate Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- top_sources_from_attribution: required by input sources file
- analytic_legal_sources: FAIL
- all_nonzero_measurements_covered: PASS
- activation_norms_measured: required by input sources file
- probability_delta_recorded: 1e-06
- logsumexp_lipschitz_applied: PASS
- primary_region_bound_is_deterministic_worst: PASS

## Delta Self Control
Refer to Phase 1 report; Phase 2 consumes only cross-path empirical deltas.

## Summary
DOWNGRADE: source file is not an analytic legal-error certificate; observed-delta heuristics cannot classify fragile versus bug.

## Tightness
| source_count | certificate_kind | legal_source_validation | source_coverage_validation | logprob_bound_prob | logprob_bound_worst | empirical_delta_p99 | empirical_delta_max | tightness_prob_over_p99 | tightness_worst_over_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | empirical_heuristic | False | True | 1444845.1429441718 | 1444845.1429441718 | 0.02150791049003602 | 0.08493077754974365 | 67177383.11276336 | 67177383.11276336 |

## Per Source
| name | mechanism | dtype | reduction_length | sum_abs | reduction | materialization_count_delta | local_scale | propagation | logprob_lipschitz | reduction_path_count | assumptions_verified | algorithm_order_known | input_norm_measured | propagation_certified | notes | activation_bound_worst | activation_bound_prob | probability_failure_budget | logprob_bound_worst | logprob_bound_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase15_L1_attention_backend | algorithm_structure | fp16 | 4096 | 69230.58984375 | tree | 0 | 16.901999473571777 | 144.76535111236754 | 2.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 118139.77186381267 | 118139.77186381267 | 1.6666666666666665e-07 | 236279.54372762534 | 236279.54372762534 |
| phase15_L6_torch.compile | mixed | fp16 | 4096 | 60375.2412109375 | tree | 1 | 14.740049123764038 | 214.3268768976695 | 2.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 152536.35932638258 | 152536.35932638258 | 1.6666666666666665e-07 | 305072.71865276515 | 305072.71865276515 |
| phase15_L5_low-precision_matmul_reduction_precision | reduction_precision | fp16 | 4096 | 59951.8662109375 | tree | 0 | 14.636686086654663 | 639.2316969645769 | 2.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 451746.1670830357 | 451746.1670830357 | 1.6666666666666665e-07 | 903492.3341660714 | 903492.3341660714 |
| phase15_L4_log_softmax_precision | rounding_precision | fp16 | 128000 | 32.639220464911745 | tree | 1 | 0.000254993909882123 | 1.0 | 1.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 0.5463977099004411 | 0.5463977099004411 | 1.6666666666666665e-07 | 0.5463977099004411 | 0.5463977099004411 |
| phase15_L2_RMSNorm_fused/unfused | materialization_points | fp16 | 4096 | 4.096e-09 | tree | 1 | 1e-12 | 1.0 | 2.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 4.8283396181867564e-11 | 4.8283396181867564e-11 | 1.6666666666666665e-07 | 9.656679236373513e-11 | 9.656679236373513e-11 |
| phase15_L3_intermediate_materialization | materialization_points | fp16 | 4096 | 4.096e-09 | tree | 1 | 1e-12 | 1.0 | 2.0 | 2 | False | False | False | False | empirical heuristic inferred from observed deltas; not a legal rounding-error certificate | 4.8283396181867564e-11 | 4.8283396181867564e-11 | 1.6666666666666665e-07 | 9.656679236373513e-11 | 9.656679236373513e-11 |
