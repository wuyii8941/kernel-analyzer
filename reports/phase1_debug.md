# Phase 1 Logprob Pipeline

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- fixed_response_tokens: PASS
- token_alignment_checked: PASS
- same_weights_config_expected: PASS
- model_weight_fingerprint_match: PASS
- deterministic_env_recorded: PASS
- warn_only_messages_recorded: PASS
- delta_self_ref_gate: PASS
- delta_self_alt_gate: PASS
- sample_and_token_scale_gate: PASS

## Delta Self Control
ref p99=0, alt p99=0, cross p50=0.000248935.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 51200 | 0.005363577003829145 | 0.0002489350736141205 | 0.029018276929855317 | 0.07054763793945314 | 0.29408979415893555 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 12800 | 0.0057837121980395725 | 0.0006963647902011871 | 0.031051468849182105 | 0.060130293369293264 | 0.29408979415893555 |
| 32-63 | 12800 | 0.003978678507840058 | 2.2105872631072998e-05 | 0.022594399750232697 | 0.06080919027328502 | 0.1746053695678711 |
| 64-95 | 12800 | 0.005695918788799528 | 0.0003085979260504246 | 0.029558122158050523 | 0.07622653245925903 | 0.21627235412597656 |
| 96-127 | 12800 | 0.00599599852063742 | 0.0003386605530977249 | 0.031514632701873775 | 0.08513642311096198 | 0.1965193748474121 |

## Warn Only Messages
_None captured._
