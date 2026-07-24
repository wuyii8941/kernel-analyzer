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

This 3-step smoke run found one naturally occurring clipping branch fork among 512 applicable tokens. The case has positive advantage, `logp_ref-old_logp=0.189058`, `logp_alt-old_logp=0.176403`, and boundary `log(1.2)=0.182322`; ref is clipped and alt is not. Both online self deltas are exactly zero, and token ID alignment is complete.

This is pipeline validation, not the final natural-fork claim. Setting `max_steps=3` changed the linear learning-rate schedule relative to the original 300-step Phase 0 recipe. The fork must be reproduced in the canonical 300-step online scan before entering Phase 6.

No legal B is available, so the case region remains `unknown`; it is not labelled fragile or bug.

## External Validity
The smoke run uses T4 FP16. It demonstrates the fork mechanism under FP16 but does not replace a BF16 replication.

## Rates
| n_certificates | n_applicable_decisions | n_zero_advantage_not_applicable | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 0 | 4 | 0 | 0 | 0 | 0 | 0.002024758607149124 | 0.011466693878173825 | 0.015200748443603513 | 0.1927798590988765 | 0.018330283612356287 | 0.09357840772900469 | 0.001953125 | 0.001953125 | 1.953125 | 250.0 | 1.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000000_091ccd9ffe84",
  "clip_alt": false,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.0067367470390531925,
  "clip_ref": true,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -1.8596572875976562,
  "logp_ref": -1.8470020294189453,
  "logprob_delta": 0.012655258178710938,
  "metadata": {
    "bound_metadata": {
      "delta_bound_prob": null,
      "delta_bound_worst": null,
      "primary_bound_kind": "deterministic_worst",
      "probability_region_is_not_bug_proof": true
    },
    "logprobs": "results/online_smoke_logprobs.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {},
    "rollout": "results/online_smoke_logprobs.jsonl",
    "rollout_advantage": 1.4972960948944092,
    "rollout_alignment": {
      "phase1_token_id": 6771,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 6771,
      "token_id_match": true
    },
    "token_metrics": {
      "entropy_alt": null,
      "entropy_delta": null,
      "entropy_ref": null,
      "token_class": "alphabetic"
    },
    "tokenization": {}
  },
  "old_logp": -2.036060333251953,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 6771,
  "token_index": 0,
  "token_text": " Let"
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
