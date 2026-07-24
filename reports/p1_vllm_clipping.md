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
| n_certificates | n_applicable_decisions | n_zero_advantage_not_applicable | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 0 | 4 | 0 | 0 | 0 | 0 | 0.38583323400345737 | 1.4253466308116907 | 3.2161647105216975 | 0.1852548093153907 | 0.010074577318933258 | 0.03838293736818779 | 0.271484375 | 0.46484375 | 271.484375 | 34750.0 | 1.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000001_2817771126c0",
  "clip_alt": false,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.04120158315965866,
  "clip_ref": true,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -3.008202075958252,
  "logp_ref": -2.959591865539551,
  "logprob_delta": 0.04861021041870117,
  "metadata": {
    "bound_metadata": {
      "delta_bound_prob": null,
      "delta_bound_worst": null,
      "primary_bound_kind": "deterministic_worst",
      "probability_region_is_not_bug_proof": true
    },
    "logprobs": "results/p1_vllm/state_aligned_logprobs.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {},
    "rollout": "results/p1_vllm/state_aligned_rollout.jsonl",
    "rollout_advantage": 1.4502660036087036,
    "rollout_alignment": {
      "phase1_token_id": 7039,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 7039,
      "token_id_match": true
    },
    "token_metrics": {
      "entropy_alt": null,
      "entropy_delta": null,
      "entropy_ref": null,
      "token_class": "whitespace"
    },
    "tokenization": {}
  },
  "old_logp": -3.183115005493164,
  "path_alt": "vllm-fp16-teacher-forcing",
  "path_ref": "hf-eager-fp16-weights-sdpa-math-step5",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 7039,
  "token_index": 72,
  "token_text": null
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
