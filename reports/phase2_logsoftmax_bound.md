# Phase 2 Log-Softmax Conditional Bound

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- shared upstream logits: PASS
- runtime input/accumulator/output kernel types profiled: PASS
- arbitrary reduction order covered: PASS
- CUDA expf/logf ULP values recorded: PASS
- vendor values are guaranteed mathematical bounds: FAIL
- certified bug classification enabled: FAIL / prohibited

## Delta Self Control
Both half-input and float-input log-softmax outputs are bitwise reproducible across the two measured calls.

## Summary
| certificate_kind | vocabulary_size | deterministic_conditional_bound | probability_conditional_bound | empirical_target_delta_p99 | empirical_target_delta_max | deterministic_tightness_over_p99 | probability_tightness_over_p99 | analytic_legal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vendor_documented_conditional | 151936 | 0.018293388146037714 | 0.0001918047424971219 | 4.76837158203125e-07 | 4.76837158203125e-07 | 38364.015537239284 | 402.2436993373242 | False |

## Legal Status
This is a vendor-documented conditional envelope, not `analytic_legal`. CUDA 12.6 lists `expf` at 2 ULP and `logf` at 1 ULP, but explicitly says the table is based on extensive, non-exhaustive testing and is not guaranteed. Consequently this result may support conservative stable/unknown analysis but cannot prove that a larger delta is a bug.

## Formula
B_det = log((1+rho)/(1-rho)) + transcendental/output terms, rho=(1+gamma_(n-1))*(1+4u)-1; arbitrary positive-sum order per path

## External Validity
The measured range and kernels are from CUDA 12.6 on Tesla T4 with FP16 autocast. Native BF16 uses different input types and requires a new isolation audit.
