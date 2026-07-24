# Phase 1.5 Attribution Canaries

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- known perturbation magnitude is 1e-3: PASS
- every switched module produced observable nonzero logprob delta: PASS

## Delta Self Control
Canaries are positive controls, not self-consistency estimates. Phase A1 remains authoritative.

## External Validity
Canary pass/fail validates instrumentation on T4 FP16 only; it does not estimate production BF16 effects.

## Results
| level | config | target | path_key | module | injected_magnitude | token_count | observed_delta_max | observed_nonzero_tokens | canary_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4 | results/configs/hf_logsoftmax_upcast.phase0_final.yaml | lm_head | path_alt | lm_head | 0.004 | 128 | 1.9073486328125e-06 | 1 | True |
| L6 | configs/hf_compile_sdpa_math_audit.yaml | root_model | path_alt | <root_model> | 0.004 | 128 | 3.9577484130859375e-05 | 9 | True |
