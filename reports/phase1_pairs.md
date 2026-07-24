# Phase 1 Path Pair Manifest

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- debug_pair_excluded_from_claim: PASS
- compile_claim_pair_present: PASS
- sdpa_flash_claim_pair_present: PASS
- vllm_measured_or_unavailable: PASS

## Delta Self Control
Every measured pair independently enforces self p99 < 0.1 * cross p50.

## Summary
Required debug and claim path pairs were validated independently.

## Measured Pairs
| name | role | path | samples | tokens | delta_p50 | delta_p99 | self_ref_p99 | self_alt_p99 | scale_gate | self_gate | weights_gate | determinism_gate | pair_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| debug_fp32_bf16 | debug_only | results/phase1_debug_fp32_bf16.jsonl | 400 | 51200 | 0.0002489350736141205 | 0.07054763793945314 | 0.0 | 0.0 | True | True | True | True | True |
| claim_eager_compile_same_dtype | claim | results/phase1_logprobs.jsonl | 400 | 51200 | 4.697870463132858e-05 | 0.02150791049003602 | 0.0 | 0.0 | True | True | True | True | True |
| claim_sdpa_backend_same_dtype | claim | results/phase1_sdpa_logprobs.jsonl | 400 | 51200 | 6.138964090496302e-05 | 0.026027302742004415 | 0.0 | 0.0 | True | True | True | True | True |

## Optional vLLM
| name | role | available | version | status |
| --- | --- | --- | --- | --- |
| claim_hf_vllm_bf16 | claim_optional | False | None | skipped_package_unavailable |
