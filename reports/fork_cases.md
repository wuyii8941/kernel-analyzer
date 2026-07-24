# Fork Case Report

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Summary
- certificates: 51200
- actual_fork_cases: 5
- reported_cases: 5

## Case 1: grpo_000004_50bbbbeba833 token 116

- claim_ready_without_manual_review: True
- region: unknown
- token_id: 30543
- token_text: "️"
- path_ref: hf-eager-fp16-sdpa-math-online
- path_alt: hf-compile-fp16-sdpa-math-online

### One-Line Math
negative advantage: boundary=log(1-eps); old_logp=-1.3672886, logp_ref=-1.5903339, logp_alt=-1.5907154, margin=9.8202193e-05, delta=0.00038146973, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | PASS | model/tokenizer source ref=online_shared_training_model, alt=online_shared_training_model |
| model_weights_identical | PASS | ref_sha256=online_shared_model_step_14, alt_sha256=online_shared_model_step_14 |
| prompt_tokens_identical | PASS | prompt_token_hash=31c95c87d0db476c577c4f6bc03aca2aa49420066a1650066eee5b20ee227f98 |
| response_tokens_identical | PASS | response_token_hash=32636a1c69cc7b26ee19cf29c25201b39a248cb00bba39710e787ffdb899a25a |
| bos_eos_chat_template_identical | PASS | full token hash matched during Phase 1 merge |
| dropout_disabled_eval_mode | PASS | execution_invariants={'attention_backend_locked_math_both_paths': True, 'compiled_warmup_discarded': True, 'default_position_ids_both_paths': True, 'dropout_disabled_by_training_config': True, 'model_training_mode': True, 'same_attention_mask_both_paths': True, 'same_input_ids_both_paths': True, 'shared_model_object': True} |
| position_ids_identical | PASS | same full token sequence and explicit default-position invariant |
| attention_mask_identical | PASS | same full token sequence and explicit default-mask invariant |
| dtype_backend_only_intended_change | PASS | path_ref={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': False, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-eager-fp16-sdpa-math-online', 'parameter_dtype': 'float32'}, path_alt={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': True, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-compile-fp16-sdpa-math-online', 'parameter_dtype': 'float32'} |
| same_token_compared | PASS | case_id=grpo_000004_50bbbbeba833, token_index=116, token_id=30543 |
| old_logp_same_response_token | PASS | rollout_alignment={'phase1_token_id': 30543, 'rollout_state': 'pre_minibatch', 'rollout_token_id': 30543, 'token_id_match': True} |
| advantage_sign_correct | PASS | advantage_sign=-1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | torch={'cuda_version': '12.6', 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'deterministic_warn_only': True, 'gpu_name': 'Tesla T4', 'version': '2.13.0.dev20260609+cu126'}, deterministic_env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'} |
| delta_self_gate_passed | PASS | aggregate_phase1_gates={'delta_self_alt_gate': True, 'delta_self_ref_gate': True, 'gate_rule': 'exact_zero_self_with_nonzero_cross_when_cross_p50_zero'} |

### Certificate
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
  "token_text": "️"
}
```

## Case 2: grpo_000001_2817771126c0 token 80

- claim_ready_without_manual_review: True
- region: unknown
- token_id: 476
- token_text: " or"
- path_ref: hf-eager-fp16-sdpa-math-online
- path_alt: hf-compile-fp16-sdpa-math-online

### One-Line Math
positive advantage: boundary=log(1+eps); old_logp=-1.2596855, logp_ref=-1.0795212, logp_alt=-1.0754948, margin=0.0021572196, delta=0.004026413, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | PASS | model/tokenizer source ref=online_shared_training_model, alt=online_shared_training_model |
| model_weights_identical | PASS | ref_sha256=online_shared_model_step_5, alt_sha256=online_shared_model_step_5 |
| prompt_tokens_identical | PASS | prompt_token_hash=183c96d19d00dc301df099bc5b3775093d57d88cf77fe26d6044197df9bba85b |
| response_tokens_identical | PASS | response_token_hash=bb013e1aeac627b390d9381c6b7d4b2a8803a2d64b8616dd6a566e1764f8fcdc |
| bos_eos_chat_template_identical | PASS | full token hash matched during Phase 1 merge |
| dropout_disabled_eval_mode | PASS | execution_invariants={'attention_backend_locked_math_both_paths': True, 'compiled_warmup_discarded': True, 'default_position_ids_both_paths': True, 'dropout_disabled_by_training_config': True, 'model_training_mode': True, 'same_attention_mask_both_paths': True, 'same_input_ids_both_paths': True, 'shared_model_object': True} |
| position_ids_identical | PASS | same full token sequence and explicit default-position invariant |
| attention_mask_identical | PASS | same full token sequence and explicit default-mask invariant |
| dtype_backend_only_intended_change | PASS | path_ref={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': False, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-eager-fp16-sdpa-math-online', 'parameter_dtype': 'float32'}, path_alt={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': True, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-compile-fp16-sdpa-math-online', 'parameter_dtype': 'float32'} |
| same_token_compared | PASS | case_id=grpo_000001_2817771126c0, token_index=80, token_id=476 |
| old_logp_same_response_token | PASS | rollout_alignment={'phase1_token_id': 476, 'rollout_state': 'pre_minibatch', 'rollout_token_id': 476, 'token_id_match': True} |
| advantage_sign_correct | PASS | advantage_sign=1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | torch={'cuda_version': '12.6', 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'deterministic_warn_only': True, 'gpu_name': 'Tesla T4', 'version': '2.13.0.dev20260609+cu126'}, deterministic_env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'} |
| delta_self_gate_passed | PASS | aggregate_phase1_gates={'delta_self_alt_gate': True, 'delta_self_ref_gate': True, 'gate_rule': 'exact_zero_self_with_nonzero_cross_when_cross_p50_zero'} |

### Certificate
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000001_2817771126c0",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.002157219635751495,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -1.0754947662353516,
  "logp_ref": -1.0795211791992188,
  "logprob_delta": 0.0040264129638671875,
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
        "aggregate_sha256": "online_shared_model_step_5",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "model_artifact_fingerprint_ref": {
        "aggregate_sha256": "online_shared_model_step_5",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "online_state": {
        "optimizer_step": 5,
        "policy_iteration": 2,
        "rollout_batch": 1,
        "state": "pre_minibatch"
      },
      "phase1_gates": {
        "delta_self_alt_gate": true,
        "delta_self_ref_gate": true,
        "gate_rule": "exact_zero_self_with_nonzero_cross_when_cross_p50_zero"
      },
      "tokenization": {
        "full_token_count": 166,
        "full_token_hash": "4ee9ef1d454928a92f7abf8a976b3cb13df370a71974b4d687c34e581e697ea1",
        "prompt_token_count": 38,
        "prompt_token_hash": "183c96d19d00dc301df099bc5b3775093d57d88cf77fe26d6044197df9bba85b",
        "response_token_count": 128,
        "response_token_hash": "bb013e1aeac627b390d9381c6b7d4b2a8803a2d64b8616dd6a566e1764f8fcdc"
      },
      "training_metadata_sidecar": "data/phase4_online_full_dump.metadata.json"
    },
    "rollout": "results/phase4_online_full_enriched.jsonl",
    "rollout_advantage": 1.4502660036087036,
    "rollout_alignment": {
      "phase1_token_id": 476,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 476,
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
  "old_logp": -1.2596855163574219,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 476,
  "token_index": 80,
  "token_text": " or"
}
```

## Case 3: grpo_000003_692fbb817526 token 72

- claim_ready_without_manual_review: True
- region: unknown
- token_id: 3019
- token_text: " step"
- path_ref: hf-eager-fp16-sdpa-math-online
- path_alt: hf-compile-fp16-sdpa-math-online

### One-Line Math
positive advantage: boundary=log(1+eps); old_logp=-7.4692078, logp_ref=-7.2961063, logp_alt=-7.2843227, margin=0.0092201316, delta=0.0117836, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | PASS | model/tokenizer source ref=online_shared_training_model, alt=online_shared_training_model |
| model_weights_identical | PASS | ref_sha256=online_shared_model_step_11, alt_sha256=online_shared_model_step_11 |
| prompt_tokens_identical | PASS | prompt_token_hash=97f7fe704fded383b0ace5362dce3b9aa912e5fa798cbfd07cd27f1ce9b466f3 |
| response_tokens_identical | PASS | response_token_hash=40598be85d8c4ff2cfa214123fd343f4bb60d106eafbc058556534ec85052b86 |
| bos_eos_chat_template_identical | PASS | full token hash matched during Phase 1 merge |
| dropout_disabled_eval_mode | PASS | execution_invariants={'attention_backend_locked_math_both_paths': True, 'compiled_warmup_discarded': True, 'default_position_ids_both_paths': True, 'dropout_disabled_by_training_config': True, 'model_training_mode': True, 'same_attention_mask_both_paths': True, 'same_input_ids_both_paths': True, 'shared_model_object': True} |
| position_ids_identical | PASS | same full token sequence and explicit default-position invariant |
| attention_mask_identical | PASS | same full token sequence and explicit default-mask invariant |
| dtype_backend_only_intended_change | PASS | path_ref={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': False, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-eager-fp16-sdpa-math-online', 'parameter_dtype': 'float32'}, path_alt={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': True, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-compile-fp16-sdpa-math-online', 'parameter_dtype': 'float32'} |
| same_token_compared | PASS | case_id=grpo_000003_692fbb817526, token_index=72, token_id=3019 |
| old_logp_same_response_token | PASS | rollout_alignment={'phase1_token_id': 3019, 'rollout_state': 'pre_minibatch', 'rollout_token_id': 3019, 'token_id_match': True} |
| advantage_sign_correct | PASS | advantage_sign=1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | torch={'cuda_version': '12.6', 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'deterministic_warn_only': True, 'gpu_name': 'Tesla T4', 'version': '2.13.0.dev20260609+cu126'}, deterministic_env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'} |
| delta_self_gate_passed | PASS | aggregate_phase1_gates={'delta_self_alt_gate': True, 'delta_self_ref_gate': True, 'gate_rule': 'exact_zero_self_with_nonzero_cross_when_cross_p50_zero'} |

### Certificate
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000003_692fbb817526",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.009220131623056183,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -7.284322738647461,
  "logp_ref": -7.296106338500977,
  "logprob_delta": 0.011783599853515625,
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
        "aggregate_sha256": "online_shared_model_step_11",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "model_artifact_fingerprint_ref": {
        "aggregate_sha256": "online_shared_model_step_11",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "online_state": {
        "optimizer_step": 11,
        "policy_iteration": 2,
        "rollout_batch": 3,
        "state": "pre_minibatch"
      },
      "phase1_gates": {
        "delta_self_alt_gate": true,
        "delta_self_ref_gate": true,
        "gate_rule": "exact_zero_self_with_nonzero_cross_when_cross_p50_zero"
      },
      "tokenization": {
        "full_token_count": 166,
        "full_token_hash": "3c94c042781e30c04785e35903c34890bc382540b1503951f87f80d728e55da9",
        "prompt_token_count": 38,
        "prompt_token_hash": "97f7fe704fded383b0ace5362dce3b9aa912e5fa798cbfd07cd27f1ce9b466f3",
        "response_token_count": 128,
        "response_token_hash": "40598be85d8c4ff2cfa214123fd343f4bb60d106eafbc058556534ec85052b86"
      },
      "training_metadata_sidecar": "data/phase4_online_full_dump.metadata.json"
    },
    "rollout": "results/phase4_online_full_enriched.jsonl",
    "rollout_advantage": 1.465724229812622,
    "rollout_alignment": {
      "phase1_token_id": 3019,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 3019,
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
  "old_logp": -7.469207763671875,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 3019,
  "token_index": 72,
  "token_text": " step"
}
```

## Case 4: grpo_000003_692fbb817526 token 88

- claim_ready_without_manual_review: True
- region: unknown
- token_id: 9295
- token_text: " Next"
- path_ref: hf-eager-fp16-sdpa-math-online
- path_alt: hf-compile-fp16-sdpa-math-online

### One-Line Math
positive advantage: boundary=log(1+eps); old_logp=-3.6760941, logp_ref=-3.5071468, logp_alt=-3.4925766, margin=0.013374337, delta=0.014570236, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | PASS | model/tokenizer source ref=online_shared_training_model, alt=online_shared_training_model |
| model_weights_identical | PASS | ref_sha256=online_shared_model_step_11, alt_sha256=online_shared_model_step_11 |
| prompt_tokens_identical | PASS | prompt_token_hash=97f7fe704fded383b0ace5362dce3b9aa912e5fa798cbfd07cd27f1ce9b466f3 |
| response_tokens_identical | PASS | response_token_hash=40598be85d8c4ff2cfa214123fd343f4bb60d106eafbc058556534ec85052b86 |
| bos_eos_chat_template_identical | PASS | full token hash matched during Phase 1 merge |
| dropout_disabled_eval_mode | PASS | execution_invariants={'attention_backend_locked_math_both_paths': True, 'compiled_warmup_discarded': True, 'default_position_ids_both_paths': True, 'dropout_disabled_by_training_config': True, 'model_training_mode': True, 'same_attention_mask_both_paths': True, 'same_input_ids_both_paths': True, 'shared_model_object': True} |
| position_ids_identical | PASS | same full token sequence and explicit default-position invariant |
| attention_mask_identical | PASS | same full token sequence and explicit default-mask invariant |
| dtype_backend_only_intended_change | PASS | path_ref={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': False, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-eager-fp16-sdpa-math-online', 'parameter_dtype': 'float32'}, path_alt={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': True, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-compile-fp16-sdpa-math-online', 'parameter_dtype': 'float32'} |
| same_token_compared | PASS | case_id=grpo_000003_692fbb817526, token_index=88, token_id=9295 |
| old_logp_same_response_token | PASS | rollout_alignment={'phase1_token_id': 9295, 'rollout_state': 'pre_minibatch', 'rollout_token_id': 9295, 'token_id_match': True} |
| advantage_sign_correct | PASS | advantage_sign=1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | torch={'cuda_version': '12.6', 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'deterministic_warn_only': True, 'gpu_name': 'Tesla T4', 'version': '2.13.0.dev20260609+cu126'}, deterministic_env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'} |
| delta_self_gate_passed | PASS | aggregate_phase1_gates={'delta_self_alt_gate': True, 'delta_self_ref_gate': True, 'gate_rule': 'exact_zero_self_with_nonzero_cross_when_cross_p50_zero'} |

### Certificate
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000003_692fbb817526",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 0.013374336945321808,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -3.4925765991210938,
  "logp_ref": -3.5071468353271484,
  "logprob_delta": 0.014570236206054688,
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
        "aggregate_sha256": "online_shared_model_step_11",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "model_artifact_fingerprint_ref": {
        "aggregate_sha256": "online_shared_model_step_11",
        "kind": "shared_in_memory_training_state",
        "verified_local_files": true
      },
      "online_state": {
        "optimizer_step": 11,
        "policy_iteration": 2,
        "rollout_batch": 3,
        "state": "pre_minibatch"
      },
      "phase1_gates": {
        "delta_self_alt_gate": true,
        "delta_self_ref_gate": true,
        "gate_rule": "exact_zero_self_with_nonzero_cross_when_cross_p50_zero"
      },
      "tokenization": {
        "full_token_count": 166,
        "full_token_hash": "3c94c042781e30c04785e35903c34890bc382540b1503951f87f80d728e55da9",
        "prompt_token_count": 38,
        "prompt_token_hash": "97f7fe704fded383b0ace5362dce3b9aa912e5fa798cbfd07cd27f1ce9b466f3",
        "response_token_count": 128,
        "response_token_hash": "40598be85d8c4ff2cfa214123fd343f4bb60d106eafbc058556534ec85052b86"
      },
      "training_metadata_sidecar": "data/phase4_online_full_dump.metadata.json"
    },
    "rollout": "results/phase4_online_full_enriched.jsonl",
    "rollout_advantage": 1.465724229812622,
    "rollout_alignment": {
      "phase1_token_id": 9295,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 9295,
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
  "old_logp": -3.6760940551757812,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 9295,
  "token_index": 88,
  "token_text": " Next"
}
```

## Case 5: grpo_000004_50bbbbeba833 token 34

- claim_ready_without_manual_review: True
- region: unknown
- token_id: 1156
- token_text: " first"
- path_ref: hf-eager-fp16-sdpa-math-online
- path_alt: hf-compile-fp16-sdpa-math-online

### One-Line Math
negative advantage: boundary=log(1-eps); old_logp=-10.675325, logp_ref=-10.909842, logp_alt=-10.890917, margin=0.011372592, delta=0.018924713, clip_ref=True, clip_alt=False.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | PASS | model/tokenizer source ref=online_shared_training_model, alt=online_shared_training_model |
| model_weights_identical | PASS | ref_sha256=online_shared_model_step_14, alt_sha256=online_shared_model_step_14 |
| prompt_tokens_identical | PASS | prompt_token_hash=31c95c87d0db476c577c4f6bc03aca2aa49420066a1650066eee5b20ee227f98 |
| response_tokens_identical | PASS | response_token_hash=32636a1c69cc7b26ee19cf29c25201b39a248cb00bba39710e787ffdb899a25a |
| bos_eos_chat_template_identical | PASS | full token hash matched during Phase 1 merge |
| dropout_disabled_eval_mode | PASS | execution_invariants={'attention_backend_locked_math_both_paths': True, 'compiled_warmup_discarded': True, 'default_position_ids_both_paths': True, 'dropout_disabled_by_training_config': True, 'model_training_mode': True, 'same_attention_mask_both_paths': True, 'same_input_ids_both_paths': True, 'shared_model_object': True} |
| position_ids_identical | PASS | same full token sequence and explicit default-position invariant |
| attention_mask_identical | PASS | same full token sequence and explicit default-mask invariant |
| dtype_backend_only_intended_change | PASS | path_ref={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': False, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-eager-fp16-sdpa-math-online', 'parameter_dtype': 'float32'}, path_alt={'attention_backend': 'math', 'attn_implementation': 'sdpa', 'autocast_dtype': 'fp16', 'compile_model': True, 'model_name_or_path': 'online_shared_training_model', 'name': 'hf-compile-fp16-sdpa-math-online', 'parameter_dtype': 'float32'} |
| same_token_compared | PASS | case_id=grpo_000004_50bbbbeba833, token_index=34, token_id=1156 |
| old_logp_same_response_token | PASS | rollout_alignment={'phase1_token_id': 1156, 'rollout_state': 'pre_minibatch', 'rollout_token_id': 1156, 'token_id_match': True} |
| advantage_sign_correct | PASS | advantage_sign=-1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | torch={'cuda_version': '12.6', 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'deterministic_warn_only': True, 'gpu_name': 'Tesla T4', 'version': '2.13.0.dev20260609+cu126'}, deterministic_env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'} |
| delta_self_gate_passed | PASS | aggregate_phase1_gates={'delta_self_alt_gate': True, 'delta_self_ref_gate': True, 'gate_rule': 'exact_zero_self_with_nonzero_cross_when_cross_p50_zero'} |

### Certificate
```json
{
  "actual_fork": true,
  "advantage_sign": -1,
  "case_id": "grpo_000004_50bbbbeba833",
  "clip_alt": false,
  "clip_boundary": -0.22314355131420976,
  "clip_margin": 0.01137259248461836,
  "clip_ref": true,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -10.89091682434082,
  "logp_ref": -10.909841537475586,
  "logprob_delta": 0.018924713134765625,
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
      "phase1_token_id": 1156,
      "rollout_state": "pre_minibatch",
      "rollout_token_id": 1156,
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
  "old_logp": -10.675325393676758,
  "path_alt": "hf-compile-fp16-sdpa-math-online",
  "path_ref": "hf-eager-fp16-sdpa-math-online",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 1156,
  "token_index": 34,
  "token_text": " first"
}
```
