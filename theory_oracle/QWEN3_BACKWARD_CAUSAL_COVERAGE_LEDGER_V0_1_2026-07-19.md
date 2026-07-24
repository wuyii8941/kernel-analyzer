# Qwen3 backward causal coverage ledger v0.1

## Verdict

BACKWARD_RUNTIME_DENOMINATOR_COMPLETE_FOR_SELECTED_STATE; ROLE_MAPPING_PARTIAL; CAUSAL_COVERAGE_UNINSTANTIATED

All rows are dynamically observed treatment candidates, not causal evidence.

| Treatment | Kind | Runtime calls | Role mapping | Representatives | State |
|---|---|---:|---|---|---|
| `triton_poi_fused__to_copy_26` | generated_triton_family | 84 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_t_4` | generated_triton_family | 84 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_mul_slice_backward_sum_view_35` | generated_triton_family | 57 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_25` | generated_triton_family | 56 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_29` | generated_triton_family | 56 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_clone_transpose_view_13` | generated_triton_family | 56 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_t_2` | generated_triton_family | 56 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_t_3` | generated_triton_family | 56 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__softmax_backward_data_view_20` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_6` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_7` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_div_expand_mean_mul_pow_rsqrt_sum_view_18` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_1` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused_sum_34` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mean_mul_pow_rsqrt_sin_transpose_unsqueeze_view_10` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_21` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_view_8` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_clone_squeeze_sum_transpose_view_27` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_clone_transpose_view_19` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused_clone_9` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mean_mul_neg_pow_rsqrt_sin_slice_slice_backward_sum_transpose_unsqueeze_view_23` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_sum_transpose_unsqueeze_view_24` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_sum_transpose_unsqueeze_view_33` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_clone_div_expand_mean_mul_pow_rsqrt_sum_view_22` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused_sum_28` | generated_triton_family | 28 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,14,27 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy_add_div_expand_mean_mul_pow_rsqrt_sum_view_31` | generated_triton_family | 27 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,13,26 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32` | generated_triton_family | 27 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,13,26 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__safe_softmax_add_prepare_softmax_online_view_11` | generated_triton_family | 27 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,13,26 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sum_view_36` | generated_triton_family | 27 | CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED | 0,13,26 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_slice_backward_sum_view_16` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_per_fused__to_copy_add_div_embedding_dense_backward_expand_mean_mul_pow_rsqrt_sum_view_37` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_add_38` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__to_copy_view_0` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__unsafe_view_mul_silu_15` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_17` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_poi_fused_embedding_dense_backward_5` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__safe_softmax_add_prepare_softmax_online_view_12` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_slice_backward_sum_view_30` | generated_triton_family | 1 | SINGLETON | 0 | RUNTIME_OBSERVED_ONLY |
| `extern:bmm` | external_kernel_family | 168 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |
| `extern:mm` | external_kernel_family | 563 | MULTI_ROLE_OR_ORDERING_UNRESOLVED | UNDECLARED | RUNTIME_OBSERVED_ONLY |

The selected-state runtime denominator is 41 family names and 1,857 calls. No family currently has backward repair or injection credit.
