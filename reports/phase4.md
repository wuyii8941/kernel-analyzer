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
Across all 51,200 online rows, both self maxima and p99 values are exactly zero. Cross-path p50 is also zero because most tokens are bitwise equal, so the original strict scalar inequality `0 < 0.1 * 0` is degenerate. Separation is established by exact-zero self runs together with nonzero cross deltas and exact-zero self on every actual-fork token.

## Summary
The canonical 300-step online scan generated 51,200 state-aligned certificates over 100 rollout batches. Of these, 39,936 have nonzero advantage and are applicable clipping decisions; 11,264 are explicitly `not_applicable`.

There are 165 tokens with margin below `1e-2`, 15 fork-possible tokens, and 5 actual clipping branch forks. The in-sample empirical independence model predicted 3.758 forks, so it is only a same-population single-digit scale check, not held-out prediction performance. All five cases pass the structured confound checklist in `reports/fork_cases.md`.

Near-boundary deltas do not shrink: their p99 is `0.012311`, versus `0.007814` away from the boundary. Log-margin/log-delta Pearson correlation is weakly negative (`-0.0746`), not evidence of protective anticorrelation.

The full online signed mean `logp_alt-logp_ref` is `2.42e-7` with prompt-cluster 95% CI `[-1.02e-5, 1.06e-5]`; neither the overall bias nor the positive-vs-negative advantage difference is significant. The final claim is existence and optimization-semantic branching, not directional bias.

No usable theoretical legal bound exists. Every applicable region therefore remains `unknown`; none of the five forks is labelled fragile or bug. The empirical envelope is not used for bug classification.

## External Validity
This scan uses T4 FP16. An FP16 fork demonstrates the decision-boundary amplification mechanism, but region rates and zero-result implications do not automatically transfer to production BF16 hardware. A BF16 hardware replication remains required.

## Rates
| n_certificates | n_applicable_decisions | n_zero_advantage_not_applicable | n_samples | missing_rollout_rows | wrong_rollout_state_rows | missing_rollout_token_id_rows | token_id_mismatch_rows | mean_logprob_delta | p95_logprob_delta | p99_logprob_delta | mean_clip_margin | p1_clip_margin | p5_clip_margin | actual_fork_rate | fork_possible_rate | forks_per_1k_tokens | forks_per_1k_samples | region_rate_not_applicable | region_rate_unknown | predicted_fork_rate | observed_minus_predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 51200 | 39936 | 11264 | 400 | 0 | 0 | 0 | 0 | 0.00021849391170037098 | 0.0002155303955078125 | 0.007959556579589883 | 0.18784804499545232 | 0.02403738740649336 | 0.09845709769086411 | 0.0001252003205128205 | 0.00037560096153846156 | 0.1252003205128205 | 12.5 | 0.22 | 1.0 | 9.409586588541594e-05 | 3.110445462740456e-05 |

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
    "logprobs": "results/phase4_online_full_enriched.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {
      "config": {
        "path_alt": {
          "attention_backend": "math",
          "attn_implementation": "sdpa",
          "autocast_dtype": "fp16",
          "compile_model": true,
          "model_name_or_path": "online_shared_training_model",
          "name": "hf-compile-fp16-sdpa-math-online",
          "parameter_dtype": "float32"
        },
        "path_ref": {
          "attention_backend": "math",
          "attn_implementation": "sdpa",
          "autocast_dtype": "fp16",
          "compile_model": false,
          "model_name_or_path": "online_shared_training_model",
          "name": "hf-eager-fp16-sdpa-math-online",
          "parameter_dtype": "float32"
        }
      },
      "env": {
        "deterministic_env": {
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
          "PYTHONHASHSEED": "0"
        },
        "torch": {
          "cuda_version": "12.6",
          "cudnn_benchmark": false,
          "deterministic_algorithms": true,
          "deterministic_warn_only": true,
          "gpu_name": "Tesla T4",
          "version": "2.13.0.dev20260609+cu126"
        }
      },
      "execution_invariants": {
        "attention_backend_locked_math_both_paths": true,
        "compiled_warmup_discarded": true,
        "default_position_ids_both_paths": true,
        "dropout_disabled_by_training_config": true,
        "model_training_mode": true,
        "same_attention_mask_both_paths": true,
        "same_input_ids_both_paths": true,
        "shared_model_object": true
      },
      "model_artifact_fingerprint_alt": {
        "aggregate_sha256": "online_shared_model_step_14",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "model_artifact_fingerprint_ref": {
        "aggregate_sha256": "online_shared_model_step_14",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "online_state": {
        "optimizer_step": 14,
        "policy_iteration": 2,
        "rollout_batch": 4,
        "state": "pre_minibatch"
      },
      "phase1_gates": {
        "delta_self_alt_gate": true,
        "delta_self_ref_gate": true,
        "gate_rule": "exact_zero_self_with_nonzero_cross_when_cross_p50_zero"
      },
      "tokenization": {
        "full_token_count": 166,
        "full_token_hash": "c9ecb5f6688ebc2d8ea5269a12ad54da7ab4dda3738c1d48e39e4c358fca773f",
        "prompt_token_count": 38,
        "prompt_token_hash": "31c95c87d0db476c577c4f6bc03aca2aa49420066a1650066eee5b20ee227f98",
        "response_token_count": 128,
        "response_token_hash": "32636a1c69cc7b26ee19cf29c25201b39a248cb00bba39710e787ffdb899a25a"
      },
      "training_metadata_sidecar": "data/phase4_online_full_dump.metadata.json"
    },
    "rollout": "results/phase4_online_full_enriched.jsonl",
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
