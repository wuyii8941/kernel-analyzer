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
ref p99=0, alt p99=0, cross p50=6.13896e-05.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 51200 | 0.002359693722808424 | 6.138964090496302e-05 | 0.013668435811996458 | 0.026027302742004415 | 0.18970823287963867 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 12800 | 0.0026236883507667823 | 0.00024925731122493744 | 0.013969731330871576 | 0.02469161570072174 | 0.18970823287963867 |
| 32-63 | 12800 | 0.0016941182156768124 | 4.885194357484579e-06 | 0.0113917052745819 | 0.022168315649032592 | 0.0774226188659668 |
| 64-95 | 12800 | 0.002514315621068573 | 7.136166095733643e-05 | 0.013863027095794657 | 0.025955324172973658 | 0.07678508758544922 |
| 96-127 | 12800 | 0.0026066527037215304 | 7.626926526427269e-05 | 0.014479386806488033 | 0.029088399410247805 | 0.1470699906349182 |

## Warn Only Messages
_None captured._
