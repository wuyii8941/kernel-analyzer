# Phase 1.5 Attribution Canaries

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- known perturbation magnitude is 1e-3: PASS
- every switched module produced observable nonzero logprob delta: FAIL

## Delta Self Control
Canaries are positive controls, not self-consistency estimates. Phase A1 remains authoritative.

## External Validity
Canary pass/fail validates instrumentation on T4 FP16 only; it does not estimate production BF16 effects.

## Results
| level | config | target | path_key | module | injected_magnitude | token_count | observed_delta_max | observed_nonzero_tokens | canary_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | results/configs/hf_sdpa_math_flash.phase0_final.yaml | attention | path_alt | model.layers.0.self_attn | 0.001 | 128 | 0.02477884292602539 | 111 | True |
| L2a | configs/hf_rmsnorm_no_upcast_audit.yaml | rmsnorm | path_alt | model.layers.0.self_attn.q_norm | 0.001 | 128 | 11.746067300438881 | 44 | True |
| L2b | configs/hf_rmsnorm_submodule_compile_audit.yaml | rmsnorm | path_alt | model.layers.0.self_attn.q_norm | 0.001 | 128 | 0.031154632568359375 | 111 | True |
| L3 | configs/hf_materialization_cross_format_audit.yaml | projection | path_alt | model.layers.0.self_attn.q_proj | 0.001 | 128 | 0.1978616714477539 | 120 | True |
| L4 | results/configs/hf_logsoftmax_upcast.phase0_final.yaml | lm_head | path_alt | lm_head | 0.001 | 128 | 0.0 | 0 | False |
| L5 | results/configs/hf_matmul_reduction.phase0_final.yaml | linear | path_alt | model.layers.0.self_attn.q_proj | 0.001 | 128 | 0.030882835388183594 | 114 | True |
| L6 | configs/hf_compile_sdpa_math_audit.yaml | decoder_layer | path_alt | _orig_mod.model.layers.0 | 0.001 | 128 | 0.0 | 0 | False |
