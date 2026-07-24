# Phase 2 v2 Differential Probability Bound

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- only declared path-difference injection points included: PASS
- shared rounding cancellation established: PASS
- local cross-source independence established: FAIL
- propagation gains empirically calibrated: PASS
- propagation gain kept inside each RSS term: PASS
- empirical envelope prohibited from bug classification: PASS

## Delta Self Control
Consumes the warmed A4 and Phase 1 self gates; no self delta is reinterpreted as a legal bound.

## External Validity
Inputs are T4 FP16. A zero-fork conclusion is limited to FP16 and cannot exclude BF16 behavior.

## Summary
| certificate_kind | source_validation | logprob_bound_prob | empirical_delta_p99 | tightness_prob_over_p99 | decision |
| --- | --- | --- | --- | --- | --- |
| unverified_diagnostic | False | 262568.8948547631 | 0.02013585686683657 | 13039866.969217975 | DOWNGRADE: P4 numerical assembly is diagnostic only because source-level assumptions are unverified. |

## Validation Failures
- L1_attention_algorithm_difference_draft: cross-source local independence not established
- L5_matmul_reduction_difference_draft: cross-source local independence not established

## Per Source
| name | mechanism | dtype | reduction_length | sum_abs | reduction | materialization_count_delta | local_scale | propagation | logprob_lipschitz | reduction_path_count | assumptions_verified | algorithm_order_known | input_norm_measured | propagation_certified | difference_injection | shared_rounding_cancelled | local_error_independent | propagation_empirically_calibrated | notes | local_bound_worst | local_bound_prob | probability_failure_budget | propagated_bound_worst | propagated_bound_prob | logprob_bound_worst_term | logprob_bound_prob_term |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1_attention_algorithm_difference_draft | algorithm_structure | fp16 | 4096 | 69230.58984375 | tree | 0 | 16.901999473571777 | 144.76535111236754 | 2.0 | 2 | False | False | False | False | True | True | False | True | Diagnostic only: algorithm order and operator input norm remain unverified. | 816.0776798870334 | 816.0776798870334 | 5e-07 | 118139.77186381267 | 118139.77186381267 | 236279.54372762534 | 236279.54372762534 |
| L5_matmul_reduction_difference_draft | reduction_precision | fp16 | 3072 | 7598.930572509766 | tree | 0 | 14.636686086654663 | 639.2316969645769 | 2.0 | 2 | False | False | True | False | True | True | False | True | Diagnostic only: conservative measured product-sum upper from model.layers.27.mlp.down_proj; exact kernel order and cross-source independence remain unverified. | 89.57482010817013 | 89.57482010817013 | 5e-07 | 57259.0642630423 | 57259.0642630423 | 114518.1285260846 | 114518.1285260846 |

## Post-Audit Correction

The two-source payload above is not a complete source model for the canonical L6 eager-versus-compile pair: L1 and L5 are sensitivity probes, while L6 fusion/materialization injection points have not been causally enumerated. Therefore the `1.30e7` tightness ratio is retained only as a failed diagnostic and must not be described as the canonical pair's legal bound.

A separately isolated standalone near-output pair is documented in `reports/phase2_logsoftmax_isolation.md` and `reports/phase2_logsoftmax_bound.md`. Half-input and float-input log-softmax both accumulate and output FP32 but dispatch different CUDA kernels. A vendor-documented conditional envelope gives deterministic `B=0.0182934` and probability `B=1.91805e-4`; probability tightness is `402x`, while deterministic tightness is `3.84e4x`. CUDA 12.6 explicitly says its `expf`/`logf` ULP table is based on non-exhaustive testing and is not guaranteed, so this envelope is not `analytic_legal` and cannot classify bugs.

Applied hypothetically to the 39,936 real iteration-2 clipping margins, it covers 39,620 (`99.2087%`) and leaves 316 unknown. A canonical Trainer canary then showed that both sides receive FP32 logits and produce exact-zero cross delta over 512 tokens, so L4 is a no-op in this recipe under P1. The hypothetical coverage is retained as an engine-portability calculation, not a canonical result.

## Canonical Compile Source Audit

`reports/phase2_compile_graph_audit.md` and `reports/phase2_compile_source_inventory.md` replace the earlier two-proxy picture with the actual compiled artifact for one exact step-5 shape. Dynamo emits one graph with no graph breaks and 93 graph-level ops. Inductor expands it into 23 unique Triton templates invoked 453 times plus 197 external MM and 56 external BMM calls, for 706 compiled kernel calls. Thirteen unique templates contain reductions/transcendentals and thirteen materialize FP16 outputs. The eager and warmed-compile logits are each self-bitwise-stable, while their max cross-logit delta is `0.0625`.

This inventory proves that L1 and L5 alone cannot cover the canonical L6 difference. A legal bound would require an eager-to-compiled arithmetic map and legal local contracts for every differing fused reduction/transcendental/materialization template and GEMM path, followed by invocation-specific propagation. Those contracts are not present, so B1 ends in the predeclared stable/unknown + empirical downgrade; it does not enable certified bug labels.
