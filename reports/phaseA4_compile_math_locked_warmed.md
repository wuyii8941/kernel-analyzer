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
ref p99=0, alt p99=0, cross p50=4.3862e-05.

An additional independent-process replication ran the warmed compile path twice in separate Python processes and CUDA contexts on the same physical T4. All 51,200 token logprobs were bitwise equal (`nonzero=0`, `max=0`); PIDs were 2931946 and 2931943. Both processes discarded one full warm-up pass.

## Summary
Phase 1 produced token-level logprob deltas and self-consistency controls.

One complete ordered pass through both paths was discarded before the two measured runs. This protocol is justified by `reports/phaseA4_compile_warmstate.md`: cold-to-warm was nonzero, while warm-to-warm was bitwise equal.

## External Validity
This audit uses T4 FP16. The independent compile component is established for this execution environment; a zero-fork result on FP16 would not exclude BF16 forks.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 51200 | 0.002013081264426322 | 4.3862033635377884e-05 | 0.012304091453552244 | 0.02013585686683657 | 0.09273529052734375 | 0.0 | 0.0 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 12800 | 0.002305567987131752 | 0.00019196607172489166 | 0.01311979293823242 | 0.018869006633758546 | 0.06921100616455078 |
| 32-63 | 12800 | 0.0013794440386358958 | 3.7476420402526855e-06 | 0.009948536753654473 | 0.01566509246826172 | 0.062438011169433594 |
| 64-95 | 12800 | 0.00217555387849275 | 5.3046271204948425e-05 | 0.012748438119888303 | 0.02214709281921389 | 0.09273529052734375 |
| 96-127 | 12800 | 0.0021917591534448898 | 5.8016739785671234e-05 | 0.01289819478988647 | 0.023557826280593892 | 0.07715129852294922 |

## Warn Only Messages
- `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
