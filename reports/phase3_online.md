# Phase 3 Controlled Calibration

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- fixed_response_tokens: PASS
- controlled_old_logp_only: PASS
- not_used_as_final_claim: PASS
- same_token_comparison: PASS

## Delta Self Control
Uses Phase 1 delta_self fields if present; this script does not establish self consistency.

## Summary
Controlled boundary construction generated certificates for detector calibration.

Controlled construction is a detector unit test only. On the canonical online-aligned population, the independent empirical convolution predicts rate `9.4096e-5`, or 3.758 forks among 39,936 applicable decisions. Phase 4 observed 5.

## External Validity
Calibration uses T4 FP16 and the canonical 300-step online population. It predicts this measured configuration only; BF16 requires new delta measurements.

## Calibration
| controlled_cases | fork_possible_rate | actual_fork_rate | delta_p50 | delta_p99 | predicted_fork_rate_overall | predicted_fork_rate_late |
| --- | --- | --- | --- | --- | --- | --- |
| 13684 | 1.0 | 0.5 | 0.0 | 0.007959556579589883 | 9.409586588541594e-05 | 9.409586588541594e-05 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000000_5732bcae85d7",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 4.768371583696585e-07,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -1.851419448852539,
  "logp_ref": -1.8514213562011719,
  "logprob_delta": 1.9073486328125e-06,
  "metadata": {
    "phase": "phase3_controlled",
    "phase1_metadata": {},
    "source": "results/phase4_online_full_logprobs.jsonl",
    "tokenization": {}
  },
  "old_logp": -2.033742436157968,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 18,
  "token_index": 4,
  "token_text": "3"
}
```
