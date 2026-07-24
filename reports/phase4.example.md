# Phase 4 Natural Scan

## Confound Checklist
- fixed_response_tokens: PASS
- real_old_logp_present: PASS
- rollout_state_matches: PASS
- rollout_token_id_matches: PASS
- advantage_sign_present: PASS
- same_token_comparison: PASS
- actual_forks_need_manual_confounds: required for every actual_fork before claim

## Delta Self Control
Uses delta_self fields from Phase 1; inspect Phase 1 report for the hard gate.

## Summary
Natural scan generated v2 certificates with clipping branch decisions.

## Rates
| n_certificates | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_unknown | predicted_fork_rate | observed_minus_predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 0 | 0 | 2 | 0 | 0.00020000000000010287 | 0.00020000000000019031 | 0.00020000000000019807 | 0.00010000000000014164 | 0.00010000000000001924 | 0.00010000000000002922 | 1.0 | 1.0 | 1000.0 | 1000.0 | 1.0 | 1.0 | 0.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "sample_000001",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.00010000000000001674,
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
    "logprobs": "data/phase1_logprobs.example.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {
      "source": "example"
    },
    "rollout": "data/rollout_dump.example.jsonl",
    "rollout_advantage": null,
    "rollout_alignment": {
      "phase1_token_id": 19,
      "rollout_state": null,
      "rollout_token_id": null,
      "token_id_match": null
    },
    "tokenization": {}
  },
  "old_logp": -0.25,
  "path_alt": "dummy-alt",
  "path_ref": "dummy-ref",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 19,
  "token_index": 0,
  "token_text": " 4"
}
```
