# 所有偏差候选的 4096 步复核摘要

审计表包含 226 行：23 个主矩阵案例、11 个历史候选，以及新增模型的复核行。
严格持久性 bias 标签要求 bias-bearing 的直接作用或反馈作用本身在长程仍有方向；如果一个早先已有 bias 证据的候选在长程出现配对 loss 分叉，即使直接 bias 后来停止，也计入结果受影响的 bias 记录，但不计入严格持久性组件数量。两条轨迹不需要收敛到不同的最终 loss。

| 模型 | 算子或位置 | 长程状态 | 结果 |
|---|---|---|---|
| Qwen3-1.7B | fused CE dW accumulation | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Phi-4-mini | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | seq64 v_proj MM + output rounding | COMPLETE_4096 | 结果受影响的 bias 候选：loss 分叉但直接 bias 未保持 |
| Qwen3-1.7B | v_proj MM/output rounding | COMPLETE_4096 | 结果受影响的 bias 候选：loss 分叉但直接 bias 未保持 |
| Mamba-130M | in_proj matrix multiply | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | COMPLETE_4096 | 结果受影响的 bias 候选：loss 分叉但直接 bias 未保持 |
| Qwen3-VL-Reranker-2B | SiLU backward | COMPLETE_4096 | 反馈长程维持 + loss 分叉 |
| Qwen3-1.7B | attention S_bwd/K to q_proj | ABSTAIN | 无法安全重放 |
| DeepSeek-R1-Qwen3-8B | attention dV BMM | NOT_RUN | UNRESOLVED_FORMATION |
| Llama-3.2-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Ministral-3-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Gemma-4 E2B | RMSNorm / projection feedback region | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0057; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0103; input LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0153; attention k-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0190; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0191; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_RESOURCE | 运行环境或资源未决 |
| Mamba-130M | Mamba backward cell 0450; dt-projection bias carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0501; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0508; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0543; final norm carrier | UNRESOLVED_LONG_REPLAY_RESOURCE | 运行环境或资源未决 |
| Qwen3-1.7B | Qwen backward cell 0654; input LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | Qwen backward cell 0745; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | Qwen backward cell 0747; attention k-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | GELU/loss backward region 1401; projection carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused__unsafe_view_mul_silu_silu_backward_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused__unsafe_view_mul_silu_silu_backward_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__unsafe_view_mul_silu [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_mul_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused_embedding_dense_backward [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_add_mul_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused_clone_squeeze_sum_transpose_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_mul_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_mul_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_add_mul_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_add_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [in_out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_select_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_mul_pow_sum_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_mul_pow_sum_view [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__softmax__softmax_backward_data__to_copy_add_arange_bitwise_and_eq_expand_gt_index_le_lift_fresh_mul_new_ones_scalar_tensor_sub_unsqueeze_view_where [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_expand_gt_index_le_lift_fresh_mul_new_ones_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_expand_gt_index_le_lift_fresh_mul_new_ones_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused_add_squeeze_sum_transpose_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_div_embedding_dense_backward_expand_gelu_mul_nll_loss_forward_pow_select_backward_sum_view [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [in_out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused_add_squeeze_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_transpose_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_squeeze_sum_transpose_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_poi_fused__unsafe_view_gelu_mul_select [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_squeeze_sum_transpose_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused_add_squeeze_sum_transpose_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_select_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_add_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy_add_bitwise_or_embedding_eq_mean_mul_pow_scalar_tensor_where [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy_add_bitwise_or_embedding_eq_mean_mul_pow_scalar_tensor_where [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_red_fused__to_copy_add_bitwise_or_embedding_eq_mean_mul_pow_scalar_tensor_where [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_add_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_per_fused_sum_transpose_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_poi_fused__unsafe_view_gelu_mul [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_sin_slice_transpose_unsqueeze_view [in_out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr3] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_div_eq_expand_mul_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_tanh_tanh_backward_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy_mul_pow_sum_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | FORWARD triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Gemma-4 E2B | BACKWARD triton_poi_fused__to_copy__unsafe_view_add_mul_pow_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused__unsafe_view_mul_silu_silu_backward_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused__unsafe_view_mul_silu_silu_backward_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__unsafe_view_mul_silu [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_per_fused__to_copy_mul_sum_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused_embedding_dense_backward [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr2] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt [out_ptr1] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused_clone_squeeze_sum_transpose_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | FORWARD triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view [out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Llama-3.2-3B (text512) | BACKWARD triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view [in_out_ptr0] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:21457:out_ptr0 [backbone.layers.19.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:1306:output_0 [model.layers.18.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:681:output_0 [model.layers.23.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:20909:output_0 [backbone.layers.20.mixer.conv1d.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:1430:out_ptr1 [model.layers.0.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:871:output_0 [model.layers.30.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:18750:output_0 [backbone.norm_f.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:669:output_0 [model.layers.23.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:20908:output_0 [backbone.layers.20.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:1136:out_ptr0 [model.layers.0.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:5461:output_0 [backbone.layers.20.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1429:out_ptr0 [model.layers.0.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | ATTENTION_PROJECTION_BACKWARD backward:846:output_0 [model.layers.16.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | LOSS_CE_BACKWARD backward:517:output_0 [model.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:996:in_out_ptr0 [model.layers.13.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1000:output_0 [model.layers.13.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:1014:out_ptr0 [model.layers.13.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:10613:output_0 [backbone.layers.20.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:1306:out_ptr0 [model.layers.4.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1296:output_0 [model.layers.4.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:20906:output_0 [backbone.layers.20.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | LOSS_HEAD_BACKWARD backward:518:out_ptr0 [model.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:699:in_out_ptr0 [model.layers.22.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:1004:out_ptr0 [model.layers.13.self_attn.k_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:19281:out_ptr0 [backbone.layers.23.mixer.dt_proj.bias] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:606:out_ptr0 [model.layers.26.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:707:out_ptr0 [model.layers.22.self_attn.k_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:10605:output_0 [backbone.layers.20.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:1009:out_ptr0 [model.layers.13.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:1006:out_ptr0 [model.layers.6.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:1136:out_ptr0 [model.layers.0.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_PROJECTION_BACKWARD backward:851:output_0 [model.layers.30.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:684:out_ptr0 [model.layers.23.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:10607:output_0 [backbone.layers.20.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | LOSS_CE_BACKWARD backward:494:output_0 [model.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:868:out_ptr0 [model.layers.30.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | ATTENTION_PROJECTION_BACKWARD backward:817:output_0 [model.layers.16.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:19287:out_ptr0 [backbone.layers.23.mixer.conv1d.bias] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | NORMALIZATION_BACKWARD backward:16235:out_ptr0 [backbone.layers.0.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1293:output_0 [model.layers.18.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | ATTENTION_PROJECTION_BACKWARD backward:1037:output_0 [model.layers.5.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:18752:out_ptr1 [backbone.norm_f.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:1006:out_ptr0 [model.layers.6.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_PROJECTION_BACKWARD backward:1292:output_0 [model.layers.18.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:883:in_out_ptr0 [model.layers.29.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | ATTENTION_PROJECTION_BACKWARD backward:617:output_0 [model.layers.26.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:1276:out_ptr0 [model.layers.18.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:20913:output_0 [backbone.layers.20.mixer.in_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:874:output_0 [model.layers.30.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:678:output_0 [model.layers.23.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | NORMALIZATION_BACKWARD backward:497:out_ptr0 [model.norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:23604:in_out_ptr0 [backbone.layers.15.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | NORMALIZATION_BACKWARD backward:1884:out_ptr0 [model.layers.0.post_attention_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:19022:in_out_ptr0 [backbone.layers.23.mixer.A_log] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1257:in_out_ptr0 [model.layers.18.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:10604:output_0 [backbone.layers.20.mixer.dt_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:19280:in_out_ptr0 [backbone.layers.23.mixer.A_log] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | LOSS_HEAD_BACKWARD backward:519:in_out_ptr0 [model.layers.27.mlp.down_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_PROJECTION_BACKWARD backward:851:output_0 [model.layers.30.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:24770:out_ptr0 [backbone.layers.12.mixer.A_log] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:541:out_ptr0 [model.layers.27.self_attn.k_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1257:in_out_ptr0 [model.layers.18.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_PROJECTION_BACKWARD backward:675:output_0 [model.layers.23.input_layernorm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | ATTENTION_STATE_OR_TRANSPORT_BACKWARD backward:1293:in_out_ptr0 [model.layers.4.self_attn.q_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | NORMALIZATION_BACKWARD backward:712:out_ptr0 [model.layers.22.self_attn.k_norm.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | STATE_SPACE_RECURRENT_BACKWARD backward:19276:in_out_ptr0 [backbone.layers.23.mixer.x_proj.weight] | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |

短筛候选总数为 69；这些候选全部要求长程复核，当前仍未完成的 66 个保持未决，不改判为阴性。
32 步只叫短程方向性；4096 步是长程复核，不等于完整全参数训练收敛。
