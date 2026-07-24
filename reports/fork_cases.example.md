# Fork Case Report

## Summary
- certificates: 2
- actual_fork_cases: 2
- reported_cases: 2

## Case 1: sample_000001 token 0

- claim_ready_without_manual_review: False
- region: unknown
- token_id: 19
- token_text: " 4"
- path_ref: dummy-ref
- path_alt: dummy-alt

### One-Line Math
positive advantage: boundary=log(1+eps); old_logp=-0.25, logp_ref=-0.067778443, logp_alt=-0.067578443, margin=0.0001, delta=0.0002, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | FAIL | model/tokenizer source ref=None, alt=None |
| model_weights_identical | FAIL | same model path=False, same_weights_config_expected=None |
| prompt_tokens_identical | FAIL | prompt_token_hash=None |
| response_tokens_identical | FAIL | response_token_hash=None |
| bos_eos_chat_template_identical | FAIL | tokenization hash missing |
| dropout_disabled_eval_mode | PASS | HF runner calls model.eval() for both paths |
| position_ids_identical | FAIL | tokenization hash missing |
| attention_mask_identical | FAIL | tokenization hash missing |
| dtype_backend_only_intended_change | FAIL | path_ref={}, path_alt={} |
| same_token_compared | PASS | case_id=sample_000001, token_index=0, token_id=19 |
| old_logp_same_response_token | FAIL | rollout_alignment={'phase1_token_id': 19, 'rollout_state': None, 'rollout_token_id': None, 'token_id_match': None} |
| advantage_sign_correct | PASS | advantage_sign=1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | environment metadata present |
| delta_self_gate_passed | PASS | delta_self_ref=0.0, delta_self_alt=0.0, delta=0.00020000000000000573 |

### Certificate
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

## Case 2: sample_000002 token 0

- claim_ready_without_manual_review: False
- region: unknown
- token_id: 2500
- token_text: " small"
- path_ref: dummy-ref
- path_alt: dummy-alt

### One-Line Math
negative advantage: boundary=log(1-eps); old_logp=-1, logp_ref=-1.2230436, logp_alt=-1.2232436, margin=0.0001, delta=0.0002, clip_ref=False, clip_alt=True.

### Confound Checklist
| name | status | evidence |
| --- | --- | --- |
| tokenizer_identical | FAIL | model/tokenizer source ref=None, alt=None |
| model_weights_identical | FAIL | same model path=False, same_weights_config_expected=None |
| prompt_tokens_identical | FAIL | prompt_token_hash=None |
| response_tokens_identical | FAIL | response_token_hash=None |
| bos_eos_chat_template_identical | FAIL | tokenization hash missing |
| dropout_disabled_eval_mode | PASS | HF runner calls model.eval() for both paths |
| position_ids_identical | FAIL | tokenization hash missing |
| attention_mask_identical | FAIL | tokenization hash missing |
| dtype_backend_only_intended_change | FAIL | path_ref={}, path_alt={} |
| same_token_compared | PASS | case_id=sample_000002, token_index=0, token_id=2500 |
| old_logp_same_response_token | FAIL | rollout_alignment={'phase1_token_id': 2500, 'rollout_state': None, 'rollout_token_id': None, 'token_id_match': None} |
| advantage_sign_correct | PASS | advantage_sign=-1 |
| clipping_formula_correct | PASS | detector uses log-space PPO sign-specific boundary |
| deterministic_env_recorded | PASS | environment metadata present |
| delta_self_gate_passed | PASS | delta_self_ref=0.0, delta_self_alt=0.0, delta=0.00020000000000020002 |

### Certificate
```json
{
  "actual_fork": true,
  "advantage_sign": -1,
  "case_id": "sample_000002",
  "clip_alt": true,
  "clip_boundary": -0.22314355131420976,
  "clip_margin": 0.00010000000000026654,
  "clip_ref": false,
  "delta_bound_legal": null,
  "delta_self_alt": 0.0,
  "delta_self_ref": 0.0,
  "eps": 0.2,
  "fork_possible": true,
  "grad_contribution_alt": null,
  "grad_contribution_diff": null,
  "grad_contribution_ref": null,
  "logp_alt": -1.2232435513142097,
  "logp_ref": -1.2230435513142095,
  "logprob_delta": 0.00020000000000020002,
  "metadata": {
    "logprobs": "data/phase1_logprobs.example.jsonl",
    "phase": "phase4_natural_scan",
    "phase1_metadata": {
      "source": "example"
    },
    "rollout": "data/rollout_dump.example.jsonl",
    "rollout_alignment": {
      "phase1_token_id": 2500,
      "rollout_state": null,
      "rollout_token_id": null,
      "token_id_match": null
    },
    "tokenization": {}
  },
  "old_logp": -1.0,
  "path_alt": "dummy-alt",
  "path_ref": "dummy-ref",
  "region": "unknown",
  "schema_version": "forkcert.v2.0",
  "token_id": 2500,
  "token_index": 0,
  "token_text": " small"
}
```
