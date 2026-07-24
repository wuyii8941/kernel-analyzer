# Phase 3 Controlled Calibration

## Confound Checklist
- fixed_response_tokens: PASS
- controlled_old_logp_only: PASS
- not_used_as_final_claim: PASS
- same_token_comparison: PASS

## Delta Self Control
Uses Phase 1 delta_self fields if present; this script does not establish self consistency.

## Summary
Controlled boundary construction generated certificates for detector calibration.

## Calibration
| controlled_cases | fork_possible_rate | actual_fork_rate | delta_p50 | delta_p99 | predicted_fork_rate_overall | predicted_fork_rate_late |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.0 | 0.5 | 0.0002 | 0.0002 | 1.0 | 1.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "sample_000001",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 5.000000000002225e-05,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -0.06757844320604539,
  "logp_ref": -0.0677784432060454,
  "logprob_delta": 0.00020000000000000573,
  "metadata": {
    "phase": "phase3_controlled",
    "phase1_metadata": {
      "source": "example"
    },
    "source": "data/phase1_logprobs.example.jsonl",
    "tokenization": {}
  },
  "old_logp": -0.25005,
  "path_alt": "dummy-alt",
  "path_ref": "dummy-ref",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 19,
  "token_index": 0,
  "token_text": " 4"
}
```
