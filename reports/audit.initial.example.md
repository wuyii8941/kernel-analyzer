# ForkCert Output Audit

## Confound Checklist
- phase0_summary_present: PASS
- phase1_logprobs_present: PASS
- phase2_bounds_present: PASS
- phase4_certificates_present: PASS

## Delta Self Control
ref gate=True, alt gate=True, delta p50=0.0002

## Summary
REVIEW: Phase 2 bound is not usable as a classifier; do not make fragile/bug claim without refining B.

## Audit
| phase0_present | phase0_tokens | phase0_p_margin_lt_1e_2 | phase0_go | phase1_present | phase1_nonempty | phase1_tokens | delta_p50 | delta_p99 | delta_self_ref_p99 | delta_self_alt_p99 | delta_self_ref_gate | delta_self_alt_gate | phase2_present | phase2_decision | phase2_bound_prob | phase2_empirical_delta_p99 | phase2_classifier_usable | phase4_present | phase4_nonempty | phase4_certificates | phase4_actual_fork_rate | phase4_fork_possible_rate | phase4_region_fragile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 2 | 1.0 | True | True | True | 2 | 0.0002 | 0.0002 | 0.0 | 0.0 | True | True | True | VIOLATION: empirical p99(delta) exceeds probability bound; refine B, check confounds, or inspect bug candidates. | 3.366733860158783e-13 | 0.0002 | False | True | True | 2 | 1.0 | 1.0 | 2 |
