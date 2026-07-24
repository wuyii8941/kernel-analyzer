# Phase 6 Gradient Contribution

## Confound Checklist
- actual_forks_present: PASS
- full_autograd: FAIL
- proxy_matches_clipping_semantics: PASS

## Delta Self Control
Uses certificates generated after Phase 1 self-consistency checks.

## Summary
Gradient contribution fields were added to certificates.

## Gradient Evidence
| n_certificates | n_actual_forks | forks_with_nonzero_grad_diff | mode |
| --- | --- | --- | --- |
| 2 | 2 | 2 | branch_proxy |
