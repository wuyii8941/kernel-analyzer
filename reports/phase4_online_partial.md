# Phase 4 Natural Scan

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- fixed_response_tokens: PASS
- real_old_logp_present: PASS
- rollout_state_matches: PASS
- rollout_token_id_matches: PASS
- advantage_sign_present_or_zero_marked_not_applicable: PASS
- same_token_comparison: PASS
- actual_forks_need_manual_confounds: required for every actual_fork before claim

## Delta Self Control
Uses delta_self fields from Phase 1; inspect Phase 1 report for the hard gate.

## Summary
Natural scan generated v2 certificates with clipping branch decisions.

## Rates
| n_certificates | n_applicable_decisions | n_zero_advantage_not_applicable | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_not_applicable | region_rate_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11776 | 11264 | 512 | 92 | 0 | 0 | 0 | 0 | 0.0007746602323922244 | 0.0059261322021484375 | 0.013342456817626979 | 0.18645309611704713 | 0.013793820780741678 | 0.06522209671297534 | 0.00044389204545454544 | 0.0013316761363636363 | 0.44389204545454547 | 54.34782608695652 | 0.043478260869565216 | 1.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": -1,
  "case_id": "grpo_000004_50bbbbeba833",
  "clip_alt": true,
  "clip_boundary": -0.22314355131420976,
  "clip_margin": 9.820219311601486e-05,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -1.5907154083251953,
  "logp_ref": -1.5903339385986328,
  "logprob_delta": 0.0003814697265625,
  "metadata": {
    "bound_metadata": {
      "delta_bound_prob": null,
      "delta_bound_worst": null,
      "primary_bound_kind": "deterministic_worst",
      "probability_region_is_not_bug_proof": true
    },
    "logprobs": "results/phase4_online_full_logprobs.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {},
    "rollout": "results/phase4_online_full_logprobs.jsonl",
    "rollout_advantage": -0.9260299801826477,
    "rollout_alignment": {
      "phase1_token_id": 30543,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 30543,
      "token_id_match": true
    },
    "token_metrics": {
      "entropy_alt": null,
      "entropy_delta": null,
      "entropy_ref": null,
      "token_class": "punctuation"
    },
    "tokenization": {}
  },
  "old_logp": -1.367288589477539,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 30543,
  "token_index": 116,
  "token_text": "\ufe0f"
}
```

## Fragile Set Entropy
| fragile_tokens | entropy_ref_p50 | entropy_ref_p95 | entropy_ref_mean |
| --- | --- | --- | --- |
| 0 | None | None | None |

## Fragile Set Breakdown
_No rows._

## Global Attribution Context
_No rows._
