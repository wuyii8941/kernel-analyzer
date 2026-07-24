# Phase 2 Compile Source Inventory

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- generated output code from exact audited graph: PASS
- every unique Triton template inventoried: PASS
- every Triton/external invocation counted: PASS
- eager-to-compiled arithmetic equivalence map: FAIL / pending
- per-template legal arithmetic contracts: FAIL / pending

## Delta Self Control
The parent graph audit reports bitwise eager self and warmed-compile self logits for the measured input.

## Summary
| unique_triton_templates | triton_invocations | external_mm_calls | external_bmm_calls | total_compiled_kernel_calls | numerically_relevant_templates | reduction_templates | transcendental_templates | fp16_materialization_templates | causal_difference_sources_proven |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | 453 | 197 | 56 | 706 | 13 | 12 | 13 | 13 | False |

## Numerical Templates
| kernel | invocations | tl_sum | tl_max | rsqrt | exp | sin | cos | fp16_casts | fp32_casts | fp16_output_ptrs | has_reduction | source_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 3 | True | _to_copy_add_embedding_mean_mul_pow_rsqrt_0 |
| triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2 | 28 | 1 | 0 | 3 | 0 | 1 | 1 | 0 | 17 | 0 | True | _to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2 |
| triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4 | 28 | 1 | 0 | 3 | 0 | 1 | 1 | 0 | 15 | 0 | True | _to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4 |
| triton_per_fused__safe_softmax_add_arange_bitwise_and_eq_expand_index_le_new_ones_prepare_softmax_online_scalar_tensor_unsqueeze_view_where_9 | 28 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | True | _safe_softmax_add_arange_bitwise_and_eq_expand_index_le_new_ones_prepare_softmax_online_scalar_tensor_unsqueeze_view_where_9 |
| triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 4 | 2 | True | _to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12 |
| triton_poi_fused__unsafe_view_mul_silu_14 | 28 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | False | _unsafe_view_mul_silu_14 |
| triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_15 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 6 | 3 | True | _to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_15 |
| triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_16 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 8 | 2 | True | _to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_16 |
| triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17 | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 4 | 3 | True | _to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17 |
| triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18 | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 6 | 2 | True | _to_copy__unsafe_view_add_mean_mul_pow_rsqrt_18 |
| triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19 | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 8 | 3 | True | _to_copy__unsafe_view_add_mean_mul_pow_rsqrt_19 |
| triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_20 | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 2 | True | _to_copy__unsafe_view_add_mean_mul_pow_rsqrt_20 |
| triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_21 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 4 | 0 | True | _to_copy__unsafe_view_add_mean_mul_pow_rsqrt_21 |

## Remaining Requirement
Inventory identifies compiled kernels, but a legal difference certificate still needs an eager-to-compiled operation mapping, arithmetic/error contract for every differing fused template and GEMM path, and propagation bounds for each invocation.

## External Validity
This inventory is shape-, model-, compiler-build-, cache-, GPU-, and dtype-specific. Dynamic shapes or native BF16 produce a different inventory.
