# 新 Triton signature 测量的去重结果

本表把有效位置、结构 signature、实际编译计算和独立根因分开。
模型、层号、shape 或同一计算的不同输出不会自动增加问题数。

- 有效位置：31
- 目录家族：4
- 结构 signature：31
- 实际编译计算：30
- 独立根因：未由本轮覆盖测量判定

| 阶段 | 实际 Triton 计算 | 位置数 | 目录家族 | 参数写入 RMS 范围 |
|---|---|---:|---|---:|
| BACKWARD | `triton_per_fused__unsafe_view_add_bmm_exp_mul_neg_select_backward_softplus_squeeze_sum_transpose_unsqueeze_zeros_70` | 1 | ELEMENTWISE | 0–0 |
| BACKWARD | `triton_per_fused__unsafe_view_add_exp_mul_neg_softplus_squeeze_sum_transpose_unsqueeze_86` | 1 | ELEMENTWISE | 0.265238–0.265238 |
| BACKWARD | `triton_red_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_12` | 2 | DATA_MOVEMENT_LAYOUT, ELEMENTWISE | 0–0.449831 |
| BACKWARD | `triton_per_fused__to_copy_mul_sum_view_4` | 1 | REDUCTION | 0.443209–0.443209 |
| BACKWARD | `triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0` | 1 | FUSED_MIXED | 0.117355–0.117355 |
| BACKWARD | `triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9` | 1 | ELEMENTWISE | 0.324546–0.324546 |
| BACKWARD | `triton_red_fused__to_copy__unsafe_view_mul_sum_transpose_view_12` | 1 | REDUCTION | 0.269679–0.269679 |
| BACKWARD | `triton_poi_fused__to_copy__unsafe_view_add_clone_div_expand_mul_pow_transpose_view_13` | 1 | DATA_MOVEMENT_LAYOUT | 0.375489–0.375489 |
| BACKWARD | `triton_per_fused__to_copy__unsafe_view_mul_sum_transpose_view_15` | 1 | REDUCTION | 0.339131–0.339131 |
| BACKWARD | `triton_per_fused__to_copy_mul_sum_2` | 1 | REDUCTION | 0.279883–0.279883 |
| BACKWARD | `triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_17` | 1 | ELEMENTWISE | 0.360885–0.360885 |
| BACKWARD | `triton_red_fused__to_copy_add_mul_sum_view_18` | 1 | REDUCTION | 0.417043–0.417043 |
| BACKWARD | `triton_per_fused_sum_19` | 1 | REDUCTION | 0.423826–0.423826 |
| BACKWARD | `triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4` | 1 | ELEMENTWISE | 0.37214–0.37214 |
| BACKWARD | `triton_poi_fused_clone_squeeze_sum_transpose_view_7` | 1 | DATA_MOVEMENT_LAYOUT | 0.38325–0.38325 |
| BACKWARD | `triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8` | 1 | ELEMENTWISE | 0.416246–0.416246 |
| BACKWARD | `triton_poi_fused__to_copy_arange_bmm_cat_expand_mul_sin_squeeze_sum_transpose_unsqueeze_view_9` | 1 | ELEMENTWISE | 0.220257–0.220257 |
| BACKWARD | `triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10` | 1 | ELEMENTWISE | 0.394251–0.394251 |
| BACKWARD | `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_11` | 1 | ELEMENTWISE | 0.41642–0.41642 |
| BACKWARD | `triton_red_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_13` | 1 | REDUCTION | 0.281426–0.281426 |
| BACKWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_div_expand_mul_neg_pow_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14` | 1 | DATA_MOVEMENT_LAYOUT | 0.413939–0.413939 |
| BACKWARD | `triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_20` | 1 | ELEMENTWISE | 0.506389–0.506389 |
| BACKWARD | `triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_21` | 1 | REDUCTION | 0.508517–0.508517 |
| BACKWARD | `triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_22` | 1 | ELEMENTWISE | 0.507902–0.507902 |
| BACKWARD | `triton_red_fused__to_copy_mul_sum_view_2` | 1 | REDUCTION | 0.275905–0.275905 |
| BACKWARD | `triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3` | 1 | REDUCTION | 0.225786–0.225786 |
| BACKWARD | `triton_poi_fused__unsafe_view_cat_mul_silu_silu_backward_split_view_2` | 1 | DATA_MOVEMENT_LAYOUT | 0.312191–0.312191 |
| BACKWARD | `triton_poi_fused__unsafe_view_add_clone_mul_neg_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_6` | 1 | ELEMENTWISE | 0.440418–0.440418 |
| BACKWARD | `triton_per_fused__to_copy_add_mul_sum_view_18` | 1 | REDUCTION | 0.391332–0.391332 |
| BACKWARD | `triton_per_fused__to_copy__unsafe_view_add_mul_sum_view_21` | 1 | REDUCTION | 0.47812–0.47812 |

这些结果只覆盖固定状态集合；不自动证明总体 bias、根因或训练后果。
