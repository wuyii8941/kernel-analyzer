# Phase 5 Bug Injection

## Confound Checklist
- synthetic_injection_only: PASS
- legal_delta_bound_supplied: PASS
- rollout_token_id_aligned: PASS
- kernel_level_bug_injection: FAIL
- posthoc_shift_smoke_only: PASS

## Delta Self Control
Uses Phase 1 delta_self fields if present; bug injection intentionally exceeds legal bound.

## Summary
Post-hoc logprob shifts validate classifier wiring only; they are not kernel-level bug evidence.

## Confusion Matrix
| n_certificates | expected_bug | classified_bug | bug_recall | false_non_bug | missing_rollout_rows | missing_rollout_token_id_rows | token_id_mismatch_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 6 | 1.0 | 0 | 0 | 6 | 0 |
