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
ref p99=0, alt p99=0, cross p50=2.41291e-05.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 512 | 0.0016061261911506364 | 2.4129054509103298e-05 | 0.010047399997711181 | 0.018295979499816883 | 0.03068828582763672 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 128 | 0.001554103345410951 | 0.000119820237159729 | 0.010054349899291992 | 0.01434896230697632 | 0.016191959381103516 |
| 32-63 | 128 | 0.0010679891388409857 | 2.8602516977116466e-06 | 0.005888573825359329 | 0.01538683891296387 | 0.019730091094970703 |
| 64-95 | 128 | 0.0018388696165168028 | 1.8320977687835693e-05 | 0.009187486767768854 | 0.014152801036834726 | 0.03068828582763672 |
| 96-127 | 128 | 0.001963542663833806 | 4.9384310841560364e-05 | 0.012555217742919917 | 0.021036906242370614 | 0.024374723434448242 |

## Warn Only Messages
- `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
