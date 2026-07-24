# Phase 3 Controlled Calibration

> Historical aligned-batch audit. The authoritative full-population calibration is `reports/phase3_online.md`, which measures every rollout in its own online state and predicts 3.758 forks versus 5 observed.

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

The detector produced `fork_possible_rate=1.0` and `actual_fork_rate=0.5` by construction, covering both advantage-sign branches. This is a unit test only and is not a natural-fork claim.

The empirical convolution uses only the state-aligned step-272 repair batch: 512 margins and 512 deltas from the same pre-minibatch checkpoint and rollout. Its predicted natural fork rate is zero because the minimum observed margin is `0.147793`, above the maximum observed compile delta `0.0306883`. The original 51,200-token pooled prediction is not reused because its model states were inconsistent.

## External Validity
This calibration is scoped to one T4 FP16 rollout batch. Zero predicted forks in this batch cannot exclude forks elsewhere in FP16 training or on BF16 hardware. A full prediction requires online per-rollout state alignment.

## Calibration
| controlled_cases | fork_possible_rate | actual_fork_rate | delta_p50 | delta_p99 | predicted_fork_rate_overall | predicted_fork_rate_late |
| --- | --- | --- | --- | --- | --- | --- |
| 1784 | 1.0 | 0.5 | 2.4129054509103298e-05 | 0.018295979499816883 | 0.0 | 0.0 |

## Minimum Actual Fork Case
```json
{
  "actual_fork": true,
  "advantage_sign": 1,
  "case_id": "grpo_000090_a95cab1ca2f7",
  "clip_alt": true,
  "clip_boundary": 0.18232155679395462,
  "clip_margin": 2.60770320892334e-08,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -0.08912485092878342,
  "logp_ref": -0.08912495523691177,
  "logprob_delta": 1.043081283569336e-07,
  "metadata": {
    "phase": "phase3_controlled",
    "phase1_metadata": {
      "config": {
        "path_alt": {
          "attn_implementation": "sdpa",
          "compile_model": true,
          "device": "cuda",
          "dtype": "fp16",
          "logits_upcast_fp32": true,
          "model_name_or_path": "data/phase0_policy_step272_pre",
          "name": "hf-compile-fp16-step272"
        },
        "path_ref": {
          "attn_implementation": "sdpa",
          "compile_model": false,
          "device": "cuda",
          "dtype": "fp16",
          "logits_upcast_fp32": true,
          "model_name_or_path": "data/phase0_policy_step272_pre",
          "name": "hf-eager-fp16-step272"
        },
        "same_weights_config_expected": true,
        "seed": 0
      },
      "env": {
        "deterministic_env": {
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
          "CUDA_VISIBLE_DEVICES": "8",
          "HF_HOME": "/data1/tzh/forkcert/cache/huggingface",
          "HF_HUB_CACHE": "/data1/tzh/forkcert/cache/huggingface/hub",
          "PYTHONHASHSEED": "0",
          "TRANSFORMERS_CACHE": null
        },
        "packages": {
          "accelerate": "1.14.0",
          "datasets": "5.0.0",
          "flash-attn": null,
          "torch": "2.13.0.dev20260609+cu126",
          "transformers": "5.13.0",
          "trl": "1.8.0",
          "verl": null,
          "vllm": null
        },
        "torch": {
          "cuda_available": true,
          "cuda_version": "12.6",
          "cudnn_benchmark": false,
          "deterministic_algorithms": true,
          "deterministic_warn_only": true,
          "device_count": 1,
          "device_names": [
            "Tesla T4"
          ],
          "version": "2.13.0.dev20260609+cu126"
        }
      },
      "execution_invariants": {
        "default_causal_attention_mask_both_paths": true,
        "default_position_ids_both_paths": true,
        "dropout_disabled_by_eval": true,
        "fixed_response_tokens": true,
        "model_eval_called": true
      },
      "metadata_sidecar": "results/phaseA3_compile_step272.metadata.json",
      "model_artifact_fingerprint_alt": {
        "aggregate_sha256": "6a7fb928f28488ebf4c1791ffab0518accfdf8335e164831af14bb9219fdce11",
        "kind": "local_checkpoint_files",
        "path": "/data1/tzh/forkcert/data/phase0_policy_step272_pre",
        "verified_local_files": true
      },
      "model_artifact_fingerprint_ref": {
        "aggregate_sha256": "6a7fb928f28488ebf4c1791ffab0518accfdf8335e164831af14bb9219fdce11",
        "kind": "local_checkpoint_files",
        "path": "/data1/tzh/forkcert/data/phase0_policy_step272_pre",
        "verified_local_files": true
      },
      "phase1_gates": {
        "delta_self_alt_gate": true,
        "delta_self_ref_gate": true,
        "sample_and_token_scale_gate": false
      },
      "warnings": [
        "`torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`."
      ]
    },
    "source": "results/phaseA3_compile_step272.jsonl",
    "tokenization": {
      "full_token_count": 166,
      "full_token_hash": "e61949ac15d48d48ab39c87f12ce77497df6366bcec3eb06108ad2bc5135fb3f",
      "prompt_token_count": 38,
      "prompt_token_hash": "7bb8502be6f4917f6d040f637246a76d5c7085391b2251307cde65acb5be0fb6",
      "response_token_count": 128,
      "response_token_hash": "9afc99ad4974cd0fd397a0a7b9f5b41fe4438decf72105ec02c53dbdbf466ad5"
    }
  },
  "old_logp": -0.2714464859538343,
  "path_alt": "hf-compile-fp16-step272",
  "path_ref": "hf-eager-fp16-step272",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 79,
  "token_index": 106,
  "token_text": "p"
}
```
