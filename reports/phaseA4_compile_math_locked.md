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
- delta_self_alt_gate: FAIL
- sample_and_token_scale_gate: PASS

## Delta Self Control
ref p99=0, alt p99=0.00156548, cross p50=4.38318e-05.

## Summary
INVALID FOR ATTRIBUTION. Although attention was locked correctly, the compile path failed the self-consistency gate. The cross delta distribution below is retained for diagnosis only and cannot establish whether the original compile and attention claim pairs are independent.

## External Validity
This failed audit ran on T4 FP16. No BF16 conclusion follows, and the determinism failure must be repaired before any numerical attribution is made.

## Delta Distribution
| n_samples | n_tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 51200 | 0.002017120923576265 | 4.383176565170288e-05 | 0.012326890230178817 | 0.020356223583221495 | 0.09273529052734375 | 0.0 | 0.001565480977296831 |

## Delta By Token Position
| token_positions | n | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0-31 | 12800 | 0.002306495457245404 | 0.00019158050417900085 | 0.01311979293823242 | 0.01902834177017216 | 0.06921100616455078 |
| 32-63 | 12800 | 0.0013888457261918125 | 3.732740879058838e-06 | 0.010011655092239374 | 0.015719652175903327 | 0.062438011169433594 |
| 64-95 | 12800 | 0.002178111297793398 | 5.325465463101864e-05 | 0.012733495235443112 | 0.022011560201644916 | 0.09273529052734375 |
| 96-127 | 12800 | 0.0021950312130744453 | 5.749892443418503e-05 | 0.012896496057510375 | 0.023645416498184215 | 0.07715129852294922 |

## Warn Only Messages
- `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
