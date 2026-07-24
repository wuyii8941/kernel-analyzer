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
- sample_and_token_scale_gate: FAIL

## Delta Self Control
ref p99=0, alt p99=0, cross p50=2.20168e-05.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 512 | 0.0017297256638071357 | 2.2016814909875393e-05 | 0.010343945026397704 | 0.022435224056243895 | 0.03740358352661133 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 128 | 0.0019842211496374773 | 5.0202012062072754e-05 | 0.009905421733856199 | 0.026894602775573782 | 0.03740358352661133 |
| 32-63 | 128 | 0.0007163713585338627 | 1.8347054719924927e-06 | 0.004511009156703948 | 0.010663356781005861 | 0.015326261520385742 |
| 64-95 | 128 | 0.002318014363950738 | 1.7939135432243347e-05 | 0.014007657766342146 | 0.025146505832672122 | 0.0260164737701416 |
| 96-127 | 128 | 0.0019002957831064649 | 0.0001177201047539711 | 0.010929477214813228 | 0.01628332972526551 | 0.022364139556884766 |

## Warn Only Messages
_None captured._
