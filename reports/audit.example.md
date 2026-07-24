# ForkCert Output Audit

## Confound Checklist
- phase0_summary_present: PASS
- phase1_logprobs_present: PASS
- phase15_measurements_present: PASS
- phase2_bounds_present: PASS
- phase3_controlled_certificates_present: PASS
- phase4_certificates_present: PASS
- phase5_bug_certificates_present: PASS
- phase6_grad_certificates_present: PASS

## Delta Self Control
ref gate=True, alt gate=True, delta p50=0.0002

## Summary
REVIEW: Phase 1.5 attribution measurements are incomplete; require all six one-variable ladder levels before Phase 2 work.

## Audit
| phase0_present | phase0_tokens | phase0_late_tokens | phase0_p_margin_lt_1e_2_overall | phase0_p_margin_lt_1e_2_late | phase0_real_training | phase0_go | phase1_present | phase1_nonempty | phase1_tokens | delta_p50 | delta_p99 | delta_self_ref_p99 | delta_self_alt_p99 | delta_self_ref_gate | delta_self_alt_gate | phase15_present | phase15_nonempty | phase15_rows | phase15_levels | phase15_required_levels_present | phase15_missing_levels | phase2_present | phase2_decision | phase2_certificate_kind | phase2_bound_prob | phase2_empirical_delta_p99 | phase2_classifier_usable | phase3_present | phase3_nonempty | phase3_certificates | phase3_fork_possible | phase3_actual_forks | phase3_actual_fork_rate | phase3_detector_gate | phase3_calibration_present | phase3_calibration_gate | phase3_predicted_fork_rate_late | phase4_present | phase4_nonempty | phase4_certificates | phase4_missing_rollout_rows | phase4_actual_fork_rate | phase4_fork_possible_rate | phase4_region_unknown | phase5_present | phase5_nonempty | phase5_certificates | phase5_classified_bug | phase5_bug_recall | phase5_false_non_bug | phase5_non_kernel_injections | phase5_kernel_injection_gate | phase5_token_alignment_gate | phase5_token_bad | phase5_token_missing | phase6_present | phase6_nonempty | phase6_certificates | phase6_actual_forks | phase6_missing_grad_diff | phase6_zero_grad_diff | phase6_grad_gate | phase6_autograd_actual_forks | phase6_proxy_actual_forks | phase6_unmarked_actual_forks | phase6_autograd_gate | phase4_coverage_gate | phase6_coverage_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 2 | 2 | 1.0 | 1.0 | False | False | True | True | 2 | 0.0002 | 0.0002 | 0.0 | 0.0 | True | True | True | True | 3 | L1,L4,L6 | False | L2,L3,L5 | True | DOWNGRADE: source file is not an analytic legal-error certificate; observed-delta heuristics cannot classify fragile versus bug. | unverified | 0.518956834128431 | 0.0002 | False | True | True | 8 | 8 | 4 | 0.5 | True | True | True | 1.0 | True | True | 2 | see Phase 4 report | 1.0 | 1.0 | 2 | True | True | 6 | 6 | 1.0 | 0 | 6 | False | False | 0 | 6 | True | True | 2 | 2 | 0 | 0 | True | 0 | 2 | 0 | False | True | True |
