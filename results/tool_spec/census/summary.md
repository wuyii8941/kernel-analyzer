# 对规格检查普查表（工具：e_num = K − K_R，e_sem = K_R − f）

## census/inductor

用例 86，输出 89；e_sem 分档：candidate 1，constant-level 22，none 60，small 6；报错 0；不评（非 Triton 写出或之后被改）5

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| ind_adaptive_avg_pool2d | out | 1.00 | none | 5.8e-17 | 是 | 8.0e-08 |
| ind_avg_pool1d_ceil | out | 1.00 | none | 0.0e+00 | 否 | 4.0e-08 |
| ind_avg_pool1d_ceil_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 1.9e-08 |
| ind_avg_pool2d_ceil_nopad | out | 1.00 | none | 6.8e-17 | 是 | 5.9e-08 |
| ind_avg_pool2d_ceil_nopad_bwd | dx | 1.00 | none | 1.0e-16 | 是 | 3.9e-08 |
| ind_avg_pool2d_ceil_pad | out | 1.00 | none | 7.6e-17 | 是 | 6.2e-08 |
| ind_avg_pool2d_ceil_pad_bwd | dx | 1.00 | candidate | 9.4e-02 | 是 | 3.6e-08 |
| ind_avg_pool2d_divisor | out | 1.00 | constant-level | 1.5e-08 | 是 | 6.2e-08 |
| ind_avg_pool2d_divisor_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 3.8e-08 |
| ind_bce_logits_posweight | out | 1.00 | none | 1.3e-16 | 是 | 5.8e-08 |
| ind_bce_logits_posweight_bwd | dx | 1.00 | none | 1.9e-16 | 是 | 9.0e-08 |
| ind_ce_mean_smooth | out | 1.00 | constant-level | 2.6e-08 | 是 | 6.3e-08 |
| ind_ce_mean_smooth_bwd | dx | 1.00 | constant-level | 2.6e-08 | 是 | 7.6e-08 |
| ind_ce_none | out | 1.00 | none | 6.9e-17 | 否 | 3.5e-08 |
| ind_ce_none_bwd | dx | 1.00 | none | 1.0e-16 | 否 | 4.0e-08 |
| ind_ce_weighted_smooth | out | 1.00 | constant-level | 2.6e-08 | 否 | 6.3e-08 |
| ind_ce_weighted_smooth_bwd | dx | 1.00 | constant-level | 2.6e-08 | 否 | 8.0e-08 |
| ind_cosine_similarity | out | 1.00 | none | 2.5e-16 | 否 | 1.1e-07 |
| ind_cosine_similarity_bwd | dx | 1.00 | none | 1.6e-16 | 否 | 8.1e-08 |
| ind_cosine_similarity_bwd | dy | 1.00 | none | 1.6e-16 | 否 | 8.0e-08 |
| ind_cumsum | out | 1.00 | none | 0.0e+00 | 否 | 7.8e-08 |
| ind_cumsum_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 7.8e-08 |
| ind_elu | out | 1.00 | constant-level | 3.1e-09 | 是 | 6.1e-09 |
| ind_elu_bwd | dx | 1.00 | constant-level | 4.2e-09 | 是 | 1.3e-08 |
| ind_gelu_erf | out | 1.00 | constant-level | 1.3e-09 | 是 | 2.9e-08 |
| ind_gelu_erf_bwd | dx | 1.00 | constant-level | 2.1e-09 | 是 | 4.7e-08 |
| ind_gelu_tanh | out | 1.00 | constant-level | 1.6e-09 | 是 | 2.8e-08 |
| ind_gelu_tanh_bwd | dx | 1.00 | constant-level | 5.7e-09 | 是 | 1.4e-07 |
| ind_grid_sample | out | 1.00 | none | 8.4e-17 | 否 | 4.7e-07 |
| ind_group_norm | out | 1.00 | none | 1.3e-13 | 是 | 7.0e-08 |
| ind_group_norm_bwd | dx | 1.00 | constant-level | 2.1e-09 | 是 | 6.5e-08 |
| ind_group_norm_bwd | dw | 1.00 | none | 1.3e-13 | 是 | 1.1e-07 |
| ind_group_norm_bwd | db | 1.00 | none | 0.0e+00 | 是 | 8.4e-08 |
| ind_hardswish | out | 1.00 | none | 2.3e-17 | 是 | 4.1e-08 |
| ind_hardswish_bwd | dx | 1.00 | constant-level | 1.7e-08 | 是 | 2.8e-08 |
| ind_huber | out | 1.00 | constant-level | 1.4e-08 | 是 | 5.1e-08 |
| ind_huber_bwd | dx | 1.00 | constant-level | 1.6e-08 | 是 | 2.7e-08 |
| ind_interp_area | out | 1.00 | none | 6.6e-17 | 是 | 6.4e-08 |
| ind_interp_bicubic | out | 1.00 | small | 6.6e-07 | 是 | 6.7e-07 |
| ind_interp_bicubic_bwd | dx | 1.00 | small | 6.6e-07 | 是 | 6.7e-07 |
| ind_interp_bilinear | out | 1.00 | small | 6.7e-07 | 是 | 5.6e-07 |
| ind_interp_bilinear_ac | out | 1.00 | small | 5.0e-07 | 是 | 5.5e-07 |
| ind_interp_bilinear_ac_bwd | dx | 1.00 | small | 5.0e-07 | 是 | 5.5e-07 |
| ind_interp_bilinear_bwd | dx | 1.00 | small | 6.7e-07 | 是 | 5.6e-07 |
| ind_interp_nearest_exact | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_interp_nearest_exact_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 3.6e-08 |
| ind_kl_div_logtarget | out | 1.00 | none | 8.6e-17 | 是 | 2.9e-08 |
| ind_kl_div_logtarget_bwd | dx | 1.00 | none | 2.6e-16 | 是 | 1.1e-07 |
| ind_layer_norm | out | 1.00 | none | 1.4e-14 | 是 | 6.6e-08 |
| ind_layer_norm_bwd | dx | 1.00 | constant-level | 3.0e-08 | 是 | 7.5e-08 |
| ind_layer_norm_bwd | dw | 1.00 | none | 1.4e-14 | 是 | 9.3e-08 |
| ind_layer_norm_bwd | db | 1.00 | none | 0.0e+00 | 是 | 6.7e-08 |
| ind_log_softmax | out | 1.00 | none | 7.5e-17 | 是 | 3.6e-08 |
| ind_log_softmax_bwd | dx | 1.00 | none | 1.3e-16 | 是 | 6.4e-08 |
| ind_logcumsumexp | out | 1.00 | none | 2.4e-16 | 是 | 3.9e-08 |
| ind_logcumsumexp_bwd | dx | 0.49 | none | 4.5e-15 | 是 | 1.1e-06 |
| ind_logsigmoid | out | 1.00 | none | 8.1e-17 | 是 | 2.4e-08 |
| ind_logsigmoid_bwd | dx | 1.00 | none | 8.6e-17 | 是 | 3.4e-08 |
| ind_logsumexp | out | 1.00 | none | 6.3e-17 | 是 | 2.3e-08 |
| ind_logsumexp_bwd | dx | 1.00 | none | 8.2e-16 | 否 | 3.1e-07 |
| ind_max_pool2d_ceil_dil | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_mish | out | 1.00 | none | 8.2e-17 | 是 | 3.1e-08 |
| ind_mish_bwd | dx | 1.00 | none | 3.1e-16 | 是 | 9.8e-08 |
| ind_nll_weighted | out | 1.00 | none | 1.1e-16 | 是 | 7.0e-08 |
| ind_nll_weighted_bwd | dx | 1.00 | none | 1.4e-16 | 否 | 6.5e-08 |
| ind_normalize | out | 1.00 | constant-level | 9.1e-09 | 是 | 4.8e-08 |
| ind_normalize_bwd | dx | 1.00 | constant-level | 4.7e-08 | 是 | 2.9e-08 |
| ind_pad_circular | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_pad_circular_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 3.2e-08 |
| ind_pad_reflect | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_pad_reflect_bwd | dx | 1.00 | none | 0.0e+00 | 是 | 3.4e-08 |
| ind_rms_norm | out | 1.00 | none | 1.3e-15 | 是 | 6.1e-08 |
| ind_rms_norm_bwd | dx | 1.00 | constant-level | 1.5e-09 | 是 | 6.7e-08 |
| ind_rms_norm_bwd | dw | 1.00 | none | 1.3e-15 | 是 | 8.8e-08 |
| ind_selu | out | 1.00 | constant-level | 3.1e-08 | 是 | 2.6e-08 |
| ind_selu_bwd | dx | 1.00 | constant-level | 2.9e-08 | 是 | 3.5e-08 |
| ind_silu | out | 1.00 | none | 1.2e-16 | 是 | 4.8e-08 |
| ind_silu_bwd | dx | 1.00 | none | 2.9e-16 | 是 | 1.0e-07 |
| ind_smooth_l1 | out | 1.00 | none | 1.0e-18 | 是 | 3.2e-08 |
| ind_smooth_l1_bwd | dx | 1.00 | none | 2.2e-18 | 否 | 1.0e-08 |
| ind_softmax | out | 1.00 | none | 2.2e-16 | 是 | 8.1e-08 |
| ind_softmax_bwd | dx | 1.00 | none | 4.2e-16 | 是 | 1.6e-07 |
| ind_softplus | out | 1.00 | none | 1.9e-17 | 是 | 7.3e-09 |
| ind_softplus_bwd | dx | 1.00 | none | 6.5e-17 | 是 | 3.2e-08 |
| ind_swiglu | out | 1.00 | none | 1.4e-16 | 是 | 5.4e-08 |
| ind_swiglu_bwd | da | 1.00 | none | 2.9e-16 | 是 | 1.0e-07 |
| ind_swiglu_bwd | db | 1.00 | none | 1.4e-16 | 是 | 5.4e-08 |
| ind_var_unbiased | out | 1.00 | none | 1.3e-16 | 是 | 5.0e-08 |
| ind_var_unbiased_bwd | dx | 1.00 | constant-level | 1.5e-08 | 否 | 1.7e-07 |

不评：ind_adaptive_avg_pool2d_bwd:dx，ind_grid_sample_bwd:dgrid，ind_grid_sample_bwd:dx，ind_interp_area_bwd:dx，ind_max_pool2d_ceil_dil_bwd:dx

## census/inductor2

用例 77，输出 100；e_sem 分档：constant-level 54，none 43，small 2，unresolved 1；报错 2；不评（非 Triton 写出或之后被改）10

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| ind_amax_ties | out | 1.00 | none | 1.4e-17 | 否 | 0.0e+00 |
| ind_amax_ties_bwd | dx | 1.00 | none | 1.9e-17 | 是 | 1.1e-08 |
| ind_batch_norm_train_1d | out | 1.00 | none | 3.5e-14 | 是 | 7.6e-08 |
| ind_batch_norm_train_1d | running_mean | 1.00 | constant-level | 2.3e-08 | 是 | 3.9e-08 |
| ind_batch_norm_train_1d | running_var | 1.00 | constant-level | 5.5e-08 | 是 | 5.5e-08 |
| ind_batch_norm_train_2d | out | 1.00 | none | 3.2e-14 | 是 | 8.1e-08 |
| ind_batch_norm_train_2d | running_mean | 1.00 | constant-level | 2.6e-08 | 是 | 2.7e-08 |
| ind_batch_norm_train_2d | running_var | 1.00 | constant-level | 2.5e-08 | 是 | 3.5e-08 |
| ind_bce_probs_edges | out | 0.94 | none | 1.2e-16 | 是 | 3.9e-08 |
| ind_celu | out | 1.00 | none | 8.8e-18 | 是 | 2.8e-09 |
| ind_celu_bwd | dx | 1.00 | none | 2.6e-17 | 是 | 1.1e-08 |
| ind_cosine_embedding | out | 1.00 | constant-level | 1.1e-09 | 否 | 3.1e-08 |
| ind_cosine_embedding_bwd | dx1 | 1.00 | none | 1.5e-16 | 是 | 8.8e-08 |
| ind_cosine_embedding_bwd | dx2 | 1.00 | none | 1.5e-16 | 是 | 8.8e-08 |
| ind_glu | out | 1.00 | none | 1.3e-16 | 是 | 4.8e-08 |
| ind_glu_bwd | dx | 1.00 | none | 2.6e-16 | 是 | 9.2e-08 |
| ind_hardtanh | out | 1.00 | constant-level | 2.9e-08 | 否 | 0.0e+00 |
| ind_hardtanh_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_hinge_embedding | out | 1.00 | constant-level | 8.5e-09 | 否 | 8.4e-09 |
| ind_hinge_embedding_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_instance_norm | out | 1.00 | none | 1.4e-13 | 是 | 7.6e-08 |
| ind_instance_norm_bwd | dx | 1.00 | constant-level | 4.6e-09 | 是 | 7.3e-08 |
| ind_instance_norm_bwd | dw | 1.00 | none | 1.4e-13 | 是 | 1.1e-07 |
| ind_instance_norm_bwd | db | 1.00 | none | 0.0e+00 | 否 | 7.4e-08 |
| ind_local_response_norm | out | 1.00 | constant-level | 3.2e-10 | 是 | 6.1e-08 |
| ind_local_response_norm_bwd | dx | 1.00 | constant-level | 3.4e-10 | 是 | 8.0e-08 |
| ind_logit_eps | out | 1.00 | constant-level | 3.6e-08 | 是 | 3.3e-08 |
| ind_logit_eps_bwd | dx | 1.00 | none | 8.2e-17 | 是 | 4.6e-08 |
| ind_margin_ranking | out | 1.00 | constant-level | 7.7e-09 | 是 | 3.7e-08 |
| ind_margin_ranking_bwd | da | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_margin_ranking_bwd | db | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_max_dim_ties | out | 1.00 | none | 1.4e-17 | 否 | 0.0e+00 |
| ind_max_dim_ties_bwd | dx | 0.00 | unresolved | — | — | — |
| ind_multi_margin_p2 | out | 1.00 | constant-level | 9.4e-09 | 是 | 7.1e-08 |
| ind_multi_margin_p2_bwd | dx | 1.00 | constant-level | 8.6e-09 | 否 | 5.7e-08 |
| ind_multilabel_soft_margin | out | 1.00 | constant-level | 1.5e-08 | 是 | 5.0e-08 |
| ind_multilabel_soft_margin_bwd | dx | 1.00 | constant-level | 1.5e-08 | 是 | 5.2e-08 |
| ind_poisson_nll_full | out | 1.00 | constant-level | 2.4e-09 | 是 | 4.6e-08 |
| ind_poisson_nll_full_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 5.1e-08 |
| ind_poisson_nll_rate | out | 1.00 | constant-level | 2.8e-09 | 是 | 4.1e-08 |
| ind_poisson_nll_rate_bwd | dx | 1.00 | constant-level | 9.6e-09 | 是 | 5.7e-08 |
| ind_prelu | out | 1.00 | none | 0.0e+00 | 否 | 1.3e-08 |
| ind_prelu_bwd | dx | 1.00 | none | 1.9e-20 | 否 | 1.3e-08 |
| ind_prelu_bwd | dw | 1.00 | none | 1.7e-16 | 否 | 8.2e-08 |
| ind_scatter_reduce_sum_noself | out | 1.00 | none | 5.5e-21 | 否 | 4.5e-08 |
| ind_scatter_reduce_sum_self | out | 1.00 | none | 0.0e+00 | 否 | 4.8e-08 |
| ind_soft_margin | out | 1.00 | none | 7.3e-17 | 是 | 2.8e-08 |
| ind_soft_margin_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 5.3e-08 |
| ind_softshrink | out | 1.00 | constant-level | 7.0e-09 | 是 | 3.0e-08 |
| ind_softshrink_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_softsign | out | 1.00 | none | 7.6e-17 | 是 | 4.2e-08 |
| ind_softsign_bwd | dx | 1.00 | none | 1.6e-16 | 是 | 6.2e-08 |
| ind_std_mean_c0 | out | 1.00 | none | 8.6e-17 | 否 | 3.4e-08 |
| ind_std_mean_c0_bwd | dx | 1.00 | none | 1.2e-16 | 否 | 1.2e-07 |
| ind_tanhshrink | out | 1.00 | none | 8.5e-17 | 是 | 3.1e-08 |
| ind_tanhshrink_bwd | dx | 1.00 | none | 1.5e-16 | 是 | 5.3e-08 |
| ind_threshold | out | 1.00 | none | 6.0e-17 | 否 | 0.0e+00 |
| ind_threshold_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_triplet_swap | out | 1.00 | none | 2.3e-15 | 否 | 2.5e-07 |
| ind_triplet_swap_bwd | da | 1.00 | none | 1.3e-15 | 是 | 7.7e-08 |
| ind_triplet_swap_bwd | dp | 1.00 | none | 2.8e-15 | 是 | 7.9e-08 |
| ind_triplet_swap_bwd | dn | 1.00 | none | 2.0e-15 | 是 | 6.6e-08 |
| ind_vector_norm_half | out | 1.00 | none | 1.8e-16 | 否 | 7.2e-08 |
| ind_vector_norm_half_bwd | dx | 1.00 | none | 1.4e-16 | 是 | 7.0e-08 |
| ind_vector_norm_p3 | out | 1.00 | constant-level | 6.0e-08 | 是 | 4.2e-08 |
| ind_vector_norm_p3_bwd | dx | 1.00 | small | 1.2e-07 | 是 | 1.0e-07 |
| opt_adadelta | param | 1.00 | constant-level | 2.6e-11 | 否 | 2.5e-08 |
| opt_adadelta | square_avg | 1.00 | constant-level | 1.2e-08 | 是 | 5.4e-08 |
| opt_adadelta | acc_delta | 1.00 | constant-level | 1.1e-08 | 是 | 6.5e-08 |
| opt_adagrad | param | 1.00 | constant-level | 1.5e-10 | 否 | 2.5e-08 |
| opt_adagrad | sum | 1.00 | constant-level | 1.8e-10 | 否 | 4.5e-08 |
| opt_adam_amsgrad | param | 1.00 | constant-level | 3.6e-08 | 是 | 3.4e-08 |
| opt_adam_amsgrad | exp_avg | 1.00 | constant-level | 1.2e-08 | 是 | 3.8e-08 |
| opt_adam_amsgrad | exp_avg_sq | 1.00 | constant-level | 3.8e-08 | 否 | 5.2e-08 |
| opt_adam_amsgrad | max_exp_avg_sq | 1.00 | constant-level | 3.8e-08 | 是 | 5.2e-08 |
| opt_adam_maximize | param | 1.00 | constant-level | 3.6e-08 | 是 | 3.4e-08 |
| opt_adam_maximize | exp_avg | 1.00 | constant-level | 1.2e-08 | 是 | 3.3e-08 |
| opt_adam_maximize | exp_avg_sq | 1.00 | constant-level | 3.8e-08 | 否 | 3.7e-08 |
| opt_adamax | param | 1.00 | constant-level | 6.7e-10 | 否 | 2.5e-08 |
| opt_adamax | exp_avg | 1.00 | constant-level | 1.2e-08 | 是 | 3.8e-08 |
| opt_adamax | exp_inf | 1.00 | constant-level | 7.2e-09 | 是 | 2.5e-08 |
| opt_adamw | param | 1.00 | constant-level | 1.3e-08 | 否 | 2.5e-08 |
| opt_adamw | exp_avg | 1.00 | constant-level | 1.2e-08 | 是 | 3.3e-08 |
| opt_adamw | exp_avg_sq | 1.00 | constant-level | 1.0e-08 | 是 | 3.3e-08 |
| opt_asgd | param | 1.00 | constant-level | 7.4e-12 | 是 | 2.9e-08 |
| opt_asgd | ax | 1.00 | constant-level | 3.7e-12 | 是 | 3.4e-08 |
| opt_radam | param | 1.00 | small | 7.3e-07 | 是 | 3.5e-07 |
| opt_radam | exp_avg | 1.00 | constant-level | 1.0e-08 | 是 | 3.5e-08 |
| opt_radam | exp_avg_sq | 1.00 | constant-level | 3.0e-08 | 否 | 4.1e-08 |
| opt_radam_early | param | 1.00 | constant-level | 1.9e-09 | 否 | 2.5e-08 |
| opt_radam_early | exp_avg | 1.00 | constant-level | 1.4e-08 | 否 | 3.5e-08 |
| opt_radam_early | exp_avg_sq | 1.00 | constant-level | 4.6e-08 | 否 | 4.3e-08 |
| opt_rmsprop_centered | param | 1.00 | constant-level | 2.1e-09 | 否 | 2.5e-08 |
| opt_rmsprop_centered | square_avg | 1.00 | constant-level | 1.2e-08 | 是 | 5.4e-08 |
| opt_rmsprop_centered | momentum_buffer | 1.00 | constant-level | 2.3e-08 | 是 | 3.7e-08 |
| opt_rmsprop_centered | grad_avg | 1.00 | constant-level | 1.2e-08 | 是 | 3.8e-08 |
| opt_sgd_dampening | param | 1.00 | constant-level | 1.3e-09 | 否 | 2.5e-08 |
| opt_sgd_dampening | momentum_buffer | 1.00 | constant-level | 2.1e-08 | 是 | 3.8e-08 |
| opt_sgd_nesterov | param | 1.00 | constant-level | 2.9e-09 | 否 | 2.5e-08 |
| opt_sgd_nesterov | momentum_buffer | 1.00 | constant-level | 1.6e-08 | 否 | 3.2e-08 |

报错：
- opt_nadam: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <built-in function imul>(*(FakeTensor(.
- opt_nadam_decoupled: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <built-in function imul>(*(FakeTensor(.

不评：ind_cdist_p1:out，ind_cdist_p1_bwd:da，ind_cdist_p1_bwd:db，ind_gaussian_nll_full:out，ind_gaussian_nll_full_bwd:dvar，ind_gaussian_nll_full_bwd:dx，ind_scatter_reduce_amax_noself:out，ind_scatter_reduce_amax_self:out，ind_scatter_reduce_mean_noself:out，ind_scatter_reduce_mean_self:out

## census/inductor3

用例 68，输出 55；e_sem 分档：candidate 1，constant-level 6，mixed (external re-entry) 3，none 40，small 1，unresolved 4；报错 0；不评（非 Triton 写出或之后被改）8

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| ind_atan2_bwd | dx | 1.00 | none | 1.0e-16 | 是 | 4.9e-08 |
| ind_atan2_bwd | dy | 1.00 | none | 1.6e-16 | 是 | 5.1e-08 |
| ind_bucketize_left | out | 1.00 | none | 2.1e-17 | 否 | 0.0e+00 |
| ind_cosh_sinh | out | 1.00 | none | 1.1e-16 | 是 | 5.8e-08 |
| ind_cosh_sinh_bwd | dx | 1.00 | none | 2.1e-16 | 否 | 3.5e-08 |
| ind_cummax_idx_ties | out | 1.00 | none | 5.2e-18 | 否 | 0.0e+00 |
| ind_cummax_vals | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_cummax_vals_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 2.0e-07 |
| ind_digamma_bwd | dx | 1.00 | small | 1.0e-07 | 否 | 2.5e-08 |
| ind_expm1_small | out | 1.00 | none | 8.2e-17 | 否 | 2.5e-08 |
| ind_expm1_small_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 3.7e-08 |
| ind_floor_divide_neg | out | 1.00 | none | 2.4e-20 | 是 | 1.2e-07 |
| ind_fmod_neg | out | 1.00 | none | 7.6e-23 | 否 | 0.0e+00 |
| ind_frac_trunc | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_hypot_bwd | dx | 0.00 | unresolved | — | — | — |
| ind_hypot_bwd | dy | 0.00 | unresolved | — | — | — |
| ind_i0e_bwd | dx | 1.00 | mixed (external re-entry) | 1.4e-07 | 否 | 3.5e-08 |
| ind_i1_bwd | dx | 0.00 | unresolved | — | — | — |
| ind_i1e_bwd | dx | 1.00 | mixed (external re-entry) | 3.8e-07 | 是 | 9.6e-08 |
| ind_lgamma_bwd | dx | 1.00 | mixed (external re-entry) | 5.4e-08 | 否 | 1.6e-08 |
| ind_log1p_small | out | 1.00 | none | 8.2e-17 | 否 | 2.5e-08 |
| ind_log1p_small_bwd | dx | 1.00 | none | 8.2e-17 | 是 | 4.7e-08 |
| ind_log_ndtr_bwd | dx | 0.00 | unresolved | 0.0e+00 | — | 0.0e+00 |
| ind_logaddexp | out | 1.00 | none | 8.2e-17 | 是 | 1.8e-08 |
| ind_logaddexp2 | out | 1.00 | constant-level | 3.0e-10 | 是 | 2.2e-08 |
| ind_logaddexp2_bwd | dx | 1.00 | none | 1.3e-16 | 是 | 4.5e-08 |
| ind_logaddexp2_bwd | dy | 1.00 | none | 1.3e-16 | 是 | 4.5e-08 |
| ind_logaddexp_bwd | dx | 1.00 | none | 1.3e-16 | 是 | 4.0e-08 |
| ind_logaddexp_bwd | dy | 1.00 | none | 1.3e-16 | 是 | 4.1e-08 |
| ind_nan_to_num | out | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_pow_frac | out | 1.00 | constant-level | 4.5e-08 | 是 | 9.9e-08 |
| ind_pow_frac_bwd | dx | 1.00 | constant-level | 5.3e-08 | 是 | 7.5e-08 |
| ind_remainder_neg | out | 1.00 | none | 1.1e-22 | 否 | 6.1e-09 |
| ind_round_decimals | out | 1.00 | constant-level | 2.2e-08 | 否 | 6.8e-06 |
| ind_round_half_even | out | 1.00 | none | 2.9e-17 | 否 | 0.0e+00 |
| ind_rsqrt | out | 1.00 | none | 8.1e-17 | 是 | 3.4e-08 |
| ind_rsqrt_bwd | dx | 1.00 | none | 2.8e-16 | 是 | 1.1e-07 |
| ind_searchsorted_right | out | 1.00 | none | 1.5e-17 | 否 | 0.0e+00 |
| ind_sigmoid_tail | out | 1.00 | none | 1.3e-16 | 是 | 2.5e-08 |
| ind_sigmoid_tail_bwd | dx | 1.00 | none | 1.7e-15 | 是 | 3.6e-07 |
| ind_sinc | out | 1.00 | constant-level | 4.9e-08 | 是 | 6.0e-08 |
| ind_sinc_bwd | dx | 1.00 | constant-level | 9.1e-08 | 是 | 3.3e-06 |
| ind_sort_desc_stable_idx | out | 1.00 | none | 3.5e-18 | 否 | 0.0e+00 |
| ind_sort_stable_idx | out | 1.00 | none | 3.5e-18 | 否 | 0.0e+00 |
| ind_sort_values | out | 1.00 | none | 5.5e-21 | 否 | 0.0e+00 |
| ind_sort_values_bwd | dx | 1.00 | candidate | 1.4e-03 | 否 | 0.0e+00 |
| ind_tanh_tail | out | 1.00 | none | 6.6e-17 | 是 | 1.6e-08 |
| ind_tanh_tail_bwd | dx | 1.00 | none | 4.9e-16 | 是 | 1.2e-07 |
| ind_topk_values_bwd | dx | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| ind_xlog1py | out | 1.00 | none | 1.1e-16 | 是 | 3.7e-08 |
| ind_xlog1py_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 3.7e-08 |
| ind_xlog1py_bwd | dy | 1.00 | none | 8.2e-17 | 是 | 4.8e-08 |
| ind_xlogy | out | 1.00 | none | 1.1e-16 | 是 | 3.6e-08 |
| ind_xlogy_bwd | dx | 1.00 | none | 1.1e-16 | 是 | 3.6e-08 |
| ind_xlogy_bwd | dy | 1.00 | none | 8.2e-17 | 是 | 4.7e-08 |

不评：ind_digamma:out，ind_digamma_neg:out，ind_erfcx_bwd:dx，ind_i0e:out，ind_i1e:out，ind_ndtri:out，ind_polygamma1:out，ind_topk_values:out

## census/flex

用例 27，输出 51；e_sem 分档：constant-level 47，mixed (external re-entry) 4；报错 0；不评（非 Triton 写出或之后被改）8

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| flex_bwd_alibi | dq | 1.00 | mixed (external re-entry) | 1.2e-08 | 是 | 3.5e-07 |
| flex_bwd_alibi | dk | 1.00 | mixed (external re-entry) | 1.2e-08 | 是 | 4.3e-07 |
| flex_bwd_alibi | dv | 1.00 | mixed (external re-entry) | 1.4e-08 | 是 | 4.0e-07 |
| flex_bwd_bias | dq | 1.00 | constant-level | 1.6e-08 | 否 | 4.0e-07 |
| flex_bwd_bias | dk | 1.00 | constant-level | 1.6e-08 | 是 | 4.0e-07 |
| flex_bwd_bias | dv | 1.00 | constant-level | 1.8e-08 | 否 | 3.7e-07 |
| flex_bwd_bias | dbias | 1.00 | constant-level | 1.6e-08 | 否 | 3.4e-07 |
| flex_bwd_causal | dq | 1.00 | constant-level | 1.2e-08 | 是 | 3.6e-07 |
| flex_bwd_causal | dk | 1.00 | constant-level | 1.2e-08 | 是 | 4.7e-07 |
| flex_bwd_causal | dv | 1.00 | constant-level | 1.2e-08 | 是 | 4.3e-07 |
| flex_bwd_d96 | dq | 1.00 | constant-level | 5.0e-08 | 是 | 4.0e-07 |
| flex_bwd_d96 | dk | 1.00 | constant-level | 5.1e-08 | 是 | 5.0e-07 |
| flex_bwd_d96 | dv | 1.00 | constant-level | 2.1e-08 | 是 | 4.5e-07 |
| flex_bwd_document | dq | 1.00 | constant-level | 9.2e-09 | 是 | 3.1e-07 |
| flex_bwd_document | dk | 1.00 | constant-level | 9.3e-09 | 是 | 3.3e-07 |
| flex_bwd_document | dv | 1.00 | constant-level | 1.0e-08 | 是 | 2.6e-07 |
| flex_bwd_gqa | dq | 1.00 | constant-level | 1.2e-08 | 是 | 3.6e-07 |
| flex_bwd_gqa | dk | 1.00 | constant-level | 1.2e-08 | 是 | 5.4e-07 |
| flex_bwd_gqa | dv | 1.00 | constant-level | 1.2e-08 | 是 | 5.1e-07 |
| flex_bwd_learned_alibi | dq | 1.00 | constant-level | 1.3e-08 | 是 | 3.5e-07 |
| flex_bwd_learned_alibi | dk | 1.00 | constant-level | 1.3e-08 | 是 | 4.3e-07 |
| flex_bwd_learned_alibi | dv | 1.00 | constant-level | 1.5e-08 | 是 | 4.0e-07 |
| flex_bwd_learned_alibi | dslopes | 1.00 | constant-level | 1.5e-08 | 否 | 3.9e-06 |
| flex_bwd_len200_causal | dq | 1.00 | constant-level | 1.1e-08 | 否 | 3.4e-07 |
| flex_bwd_len200_causal | dk | 1.00 | constant-level | 1.1e-08 | 是 | 4.3e-07 |
| flex_bwd_len200_causal | dv | 1.00 | constant-level | 1.2e-08 | 是 | 3.9e-07 |
| flex_bwd_lse_grad | dq | 1.00 | constant-level | 1.2e-08 | 是 | 3.6e-07 |
| flex_bwd_lse_grad | dk | 1.00 | constant-level | 1.2e-08 | 是 | 4.7e-07 |
| flex_bwd_lse_grad | dv | 1.00 | constant-level | 1.2e-08 | 是 | 4.3e-07 |
| flex_bwd_lse_grad_plain | dq | 1.00 | constant-level | 1.7e-08 | 是 | 4.9e-07 |
| flex_bwd_lse_grad_plain | dk | 1.00 | constant-level | 1.7e-08 | 是 | 4.9e-07 |
| flex_bwd_lse_grad_plain | dv | 1.00 | constant-level | 1.8e-08 | 是 | 4.7e-07 |
| flex_bwd_plain | dq | 1.00 | constant-level | 1.6e-08 | 是 | 5.0e-07 |
| flex_bwd_plain | dk | 1.00 | constant-level | 1.7e-08 | 是 | 4.9e-07 |
| flex_bwd_plain | dv | 1.00 | constant-level | 1.8e-08 | 是 | 4.7e-07 |
| flex_bwd_sliding_causal | dq | 1.00 | constant-level | 1.1e-08 | 是 | 3.3e-07 |
| flex_bwd_sliding_causal | dk | 1.00 | constant-level | 1.1e-08 | 是 | 3.4e-07 |
| flex_bwd_sliding_causal | dv | 1.00 | constant-level | 1.3e-08 | 是 | 2.9e-07 |
| flex_bwd_softcap | dq | 1.00 | constant-level | 1.5e-08 | 是 | 3.6e-07 |
| flex_bwd_softcap | dk | 1.00 | constant-level | 1.5e-08 | 是 | 4.7e-07 |
| flex_bwd_softcap | dv | 1.00 | constant-level | 1.2e-09 | 是 | 4.4e-07 |
| flex_fwd_alibi | out | 1.00 | mixed (external re-entry) | 1.4e-08 | 是 | 2.6e-07 |
| flex_fwd_bias | out | 1.00 | constant-level | 1.8e-08 | 是 | 3.3e-07 |
| flex_fwd_causal | out | 1.00 | constant-level | 1.2e-08 | 是 | 2.4e-07 |
| flex_fwd_d96 | out | 1.00 | constant-level | 2.1e-08 | 是 | 2.8e-07 |
| flex_fwd_document | out | 1.00 | constant-level | 1.0e-08 | 是 | 1.8e-07 |
| flex_fwd_gqa | out | 1.00 | constant-level | 1.2e-08 | 是 | 2.4e-07 |
| flex_fwd_len200_causal | out | 1.00 | constant-level | 1.2e-08 | 是 | 2.3e-07 |
| flex_fwd_plain | out | 1.00 | constant-level | 1.8e-08 | 是 | 4.0e-07 |
| flex_fwd_sliding_causal | out | 1.00 | constant-level | 1.3e-08 | 是 | 2.4e-07 |
| flex_fwd_softcap | out | 1.00 | constant-level | 1.2e-09 | 是 | 2.6e-07 |

不评：flex_bwd_dv32:dk，flex_bwd_dv32:dq，flex_bwd_dv32:dv，flex_bwd_q100_kv300:dk，flex_bwd_q100_kv300:dq，flex_bwd_q100_kv300:dv，flex_fwd_dv32:out，flex_fwd_q100_kv300:out

## census/tridao

用例 56，输出 78；e_sem 分档：constant-level 9，none 69；报错 0；不评（非 Triton 写出或之后被改）22

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| fa_ce_all | loss | 1.00 | constant-level | 1.1e-08 | 是 | 5.7e-08 |
| fa_ce_all | z_loss | 1.00 | constant-level | 3.8e-08 | 否 | 7.3e-08 |
| fa_ce_all_bwd | dx | 1.00 | constant-level | 2.3e-08 | 是 | 8.4e-08 |
| fa_ce_plain | loss | 1.00 | none | 7.1e-17 | 否 | 3.6e-08 |
| fa_ce_plain | z_loss | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| fa_ce_plain_bwd | dx | 1.00 | none | 1.5e-16 | 是 | 1.0e-07 |
| fa_ce_scale_zloss | loss | 1.00 | constant-level | 1.1e-08 | 是 | 4.9e-08 |
| fa_ce_scale_zloss | z_loss | 1.00 | constant-level | 3.8e-08 | 否 | 7.3e-08 |
| fa_ce_scale_zloss_bwd | dx | 1.00 | constant-level | 2.1e-08 | 是 | 7.7e-08 |
| fa_ce_smooth | loss | 1.00 | constant-level | 3.9e-10 | 是 | 4.3e-08 |
| fa_ce_smooth | z_loss | 1.00 | none | 0.0e+00 | 否 | 0.0e+00 |
| fa_ce_smooth_bwd | dx | 1.00 | constant-level | 1.5e-09 | 是 | 1.1e-07 |
| fa_ln_parallel | out | 1.00 | none | 6.4e-16 | 是 | 6.9e-08 |
| fa_ln_parallel | out1 | 1.00 | none | 6.4e-16 | 是 | 6.9e-08 |
| fa_ln_parallel_bwd | dx | 1.00 | none | 6.9e-16 | 是 | 6.4e-08 |
| fa_ln_parallel_bwd | dx1 | 1.00 | none | 6.9e-16 | 是 | 6.4e-08 |
| fa_ln_plain | out | 1.00 | none | 1.3e-15 | 是 | 6.6e-08 |
| fa_ln_plain_bwd | dx | 1.00 | none | 1.3e-15 | 是 | 6.2e-08 |
| fa_ln_residual_prenorm | out | 1.00 | none | 6.4e-16 | 是 | 6.8e-08 |
| fa_ln_residual_prenorm | residual_out | 1.00 | none | 2.6e-20 | 否 | 2.7e-08 |
| fa_ln_residual_prenorm_bwd | dx | 1.00 | none | 6.9e-16 | 是 | 5.8e-08 |
| fa_ln_residual_prenorm_bwd | dres | 1.00 | none | 6.9e-16 | 是 | 5.8e-08 |
| fa_rms_parallel | out | 1.00 | none | 4.4e-16 | 否 | 6.9e-08 |
| fa_rms_parallel | out1 | 1.00 | none | 4.5e-16 | 否 | 6.9e-08 |
| fa_rms_parallel_bwd | dx | 1.00 | none | 4.5e-16 | 否 | 6.5e-08 |
| fa_rms_parallel_bwd | dres | 1.00 | none | 4.5e-16 | 否 | 6.5e-08 |
| fa_rms_parallel_bwd | dx1 | 1.00 | none | 4.5e-16 | 否 | 6.5e-08 |
| fa_rms_plain | out | 1.00 | none | 1.2e-15 | 是 | 6.1e-08 |
| fa_rms_plain_bwd | dx | 1.00 | none | 1.3e-15 | 是 | 6.7e-08 |
| fa_rms_residual_prenorm | out | 1.00 | none | 6.3e-16 | 是 | 6.4e-08 |
| fa_rms_residual_prenorm | residual_out | 1.00 | none | 2.6e-20 | 否 | 2.7e-08 |
| fa_rms_residual_prenorm_bwd | dx | 1.00 | none | 6.4e-16 | 是 | 6.3e-08 |
| fa_rms_residual_prenorm_bwd | dres | 1.00 | none | 6.4e-16 | 是 | 6.3e-08 |
| fa_rms_rowscale | out | 1.00 | none | 6.6e-16 | 是 | 6.3e-08 |
| fa_rms_rowscale_bwd | dx | 1.00 | none | 6.2e-16 | 是 | 6.8e-08 |
| fa_rms_rowscale_bwd | dres | 1.00 | none | 7.2e-16 | 是 | 6.4e-08 |
| fa_rms_zero_centered | out | 1.00 | none | 1.2e-15 | 是 | 6.7e-08 |
| fa_rms_zero_centered_bwd | dx | 1.00 | none | 1.3e-15 | 是 | 7.2e-08 |
| fa_rotary_basic | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_conjugate | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_d80_ro48 | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_inplace | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_interleaved | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_interleaved_partial | out | 1.00 | none | 3.2e-17 | 否 | 2.9e-08 |
| fa_rotary_offset_int | out | 1.00 | none | 2.7e-17 | 是 | 2.9e-08 |
| fa_rotary_offset_tensor | out | 1.00 | none | 2.8e-17 | 是 | 2.9e-08 |
| fa_rotary_partial32 | out | 1.00 | none | 3.1e-17 | 是 | 2.9e-08 |
| fa_rotary_varlen | out | 1.00 | none | 3.6e-17 | 否 | 2.8e-08 |
| fa_rotary_varlen_offsets | out | 1.00 | none | 3.5e-17 | 否 | 2.8e-08 |
| mamba_gatednorm_after | out | 1.00 | none | 7.8e-14 | 是 | 7.9e-08 |
| mamba_gatednorm_after_bwd | dx | 1.00 | none | 8.3e-14 | 是 | 8.7e-08 |
| mamba_gatednorm_after_bwd | dz | 1.00 | none | 8.3e-14 | 否 | 1.0e-07 |
| mamba_gatednorm_before | out | 1.00 | none | 1.3e-13 | 是 | 8.2e-08 |
| mamba_gatednorm_before_bwd | dx | 1.00 | none | 1.3e-13 | 是 | 8.7e-08 |
| mamba_gatednorm_before_bwd | dz | 1.00 | none | 1.3e-13 | 是 | 9.6e-08 |
| mamba_gatednorm_group96_after_bias | out | 1.00 | none | 9.7e-14 | 是 | 7.7e-08 |
| mamba_gatednorm_group96_after_bias_bwd | dx | 1.00 | none | 1.3e-13 | 是 | 9.3e-08 |
| mamba_gatednorm_group96_after_bias_bwd | dz | 1.00 | none | 1.3e-13 | 否 | 1.1e-07 |
| mamba_gatednorm_group96_before | out | 1.00 | none | 1.3e-13 | 是 | 8.1e-08 |
| mamba_gatednorm_group96_before_bwd | dx | 1.00 | none | 1.4e-13 | 是 | 8.6e-08 |
| mamba_gatednorm_group96_before_bwd | dz | 1.00 | none | 1.3e-13 | 是 | 9.5e-08 |
| mamba_gatednorm_noz | out | 1.00 | none | 1.3e-13 | 否 | 6.2e-08 |
| mamba_gatednorm_noz_bwd | dx | 1.00 | none | 1.3e-13 | 是 | 6.7e-08 |
| mamba_ssd_D_z | y | 1.00 | none | 1.2e-15 | 是 | 9.4e-04 |
| mamba_ssd_basic | y | 1.00 | none | 1.8e-15 | 是 | 1.5e-03 |
| mamba_ssd_dt_limit | y | 1.00 | constant-level | 2.6e-08 | 是 | 1.5e-03 |
| mamba_ssd_dtbias_softplus | y | 1.00 | none | 1.3e-14 | 是 | 1.5e-03 |
| mamba_ssd_groups2 | y | 1.00 | none | 1.8e-15 | 是 | 1.5e-03 |
| mamba_ssd_h0_final | y | 1.00 | none | 1.5e-15 | 是 | 1.3e-03 |
| mamba_ssd_h0_final | final_state | 1.00 | none | 2.3e-15 | 是 | 7.7e-04 |
| mamba_ssd_len100 | y | 1.00 | none | 1.9e-15 | 是 | 1.5e-03 |
| mamba_ssd_seq_idx | y | 1.00 | none | 1.9e-15 | 是 | 1.5e-03 |
| mamba_ssu_full | y | 1.00 | none | 2.3e-16 | 是 | 1.0e-07 |
| mamba_ssu_full | state | 1.00 | none | 1.4e-16 | 是 | 6.7e-08 |
| mamba_ssu_groups2 | y | 1.00 | none | 2.3e-16 | 是 | 1.1e-07 |
| mamba_ssu_groups2 | state | 1.00 | none | 1.4e-16 | 是 | 6.7e-08 |
| mamba_ssu_plain | y | 1.00 | none | 2.1e-16 | 否 | 2.5e-07 |
| mamba_ssu_plain | state | 1.00 | none | 1.4e-16 | 是 | 2.6e-07 |

不评：fa_ln_parallel_bwd:db，fa_ln_parallel_bwd:dw，fa_ln_plain_bwd:db，fa_ln_plain_bwd:dw，fa_ln_residual_prenorm_bwd:db，fa_ln_residual_prenorm_bwd:dw，fa_rms_parallel_bwd:db，fa_rms_parallel_bwd:dw，fa_rms_plain_bwd:db，fa_rms_plain_bwd:dw，fa_rms_residual_prenorm_bwd:db，fa_rms_residual_prenorm_bwd:dw，fa_rms_rowscale_bwd:db，fa_rms_rowscale_bwd:dw，fa_rms_zero_centered_bwd:db，fa_rms_zero_centered_bwd:dw，mamba_gatednorm_after_bwd:dw，mamba_gatednorm_before_bwd:dw，mamba_gatednorm_group96_after_bias_bwd:db，mamba_gatednorm_group96_after_bias_bwd:dw，mamba_gatednorm_group96_before_bwd:dw，mamba_gatednorm_noz_bwd:dw

## census/fla

用例 24，输出 50；e_sem 分档：constant-level 41，mixed (external re-entry) 5，none 3，small 1；报错 1；不评（非 Triton 写出或之后被改）0

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| fla_gdr_chunk_basic | o | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_basic_bwd | dq | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_basic_bwd | dk | 1.00 | constant-level | 1.3e-08 | 是 | 1.7e-03 |
| fla_gdr_chunk_basic_bwd | dv | 1.00 | constant-level | 1.4e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_basic_bwd | dg | 1.00 | mixed (external re-entry) | 1.8e-03 | 是 | 9.8e-04 |
| fla_gdr_chunk_basic_bwd | dbeta | 1.00 | constant-level | 1.3e-08 | 是 | 2.0e-03 |
| fla_gdr_chunk_gate | o | 1.00 | constant-level | 1.7e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_gva | o | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_h0_bwd | dq | 1.00 | constant-level | 1.3e-08 | 是 | 1.4e-03 |
| fla_gdr_chunk_h0_bwd | dk | 1.00 | constant-level | 1.3e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_h0_bwd | dv | 1.00 | constant-level | 1.4e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_h0_bwd | dg | 1.00 | mixed (external re-entry) | 1.6e-03 | 是 | 9.3e-04 |
| fla_gdr_chunk_h0_bwd | dbeta | 1.00 | constant-level | 1.3e-08 | 是 | 2.0e-03 |
| fla_gdr_chunk_h0_bwd | dh0 | 1.00 | constant-level | 1.3e-08 | 是 | 8.8e-04 |
| fla_gdr_chunk_h0_final | o | 1.00 | constant-level | 1.3e-08 | 是 | 1.4e-03 |
| fla_gdr_chunk_h0_final | final_state | 1.00 | constant-level | 7.7e-09 | 是 | 7.9e-04 |
| fla_gdr_chunk_l2_sigmoid | o | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_l2_sigmoid_bwd | dq | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_l2_sigmoid_bwd | dk | 1.00 | constant-level | 1.3e-08 | 是 | 1.7e-03 |
| fla_gdr_chunk_l2_sigmoid_bwd | dv | 1.00 | constant-level | 1.4e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_l2_sigmoid_bwd | dg | 1.00 | mixed (external re-entry) | 1.8e-03 | 是 | 9.2e-04 |
| fla_gdr_chunk_l2_sigmoid_bwd | dbeta | 1.00 | constant-level | 1.3e-08 | 是 | 1.9e-03 |
| fla_gdr_chunk_neg_eig | o | 1.00 | constant-level | 1.4e-08 | 是 | 1.6e-03 |
| fla_gdr_chunk_t100 | o | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_t100_bwd | dq | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_t100_bwd | dk | 1.00 | constant-level | 1.3e-08 | 是 | 1.7e-03 |
| fla_gdr_chunk_t100_bwd | dv | 1.00 | constant-level | 1.4e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_t100_bwd | dg | 1.00 | mixed (external re-entry) | 1.7e-03 | 是 | 9.5e-04 |
| fla_gdr_chunk_t100_bwd | dbeta | 1.00 | constant-level | 1.3e-08 | 是 | 2.0e-03 |
| fla_gdr_chunk_varlen | o | 1.00 | constant-level | 1.3e-08 | 是 | 1.3e-03 |
| fla_gdr_chunk_varlen | final_state | 1.00 | constant-level | 7.7e-09 | 是 | 8.0e-04 |
| fla_gdr_chunk_varlen_bwd | dq | 1.00 | constant-level | 1.4e-08 | 是 | 1.5e-03 |
| fla_gdr_chunk_varlen_bwd | dk | 1.00 | constant-level | 1.3e-08 | 是 | 1.7e-03 |
| fla_gdr_chunk_varlen_bwd | dv | 1.00 | constant-level | 1.4e-08 | 是 | 1.8e-03 |
| fla_gdr_chunk_varlen_bwd | dg | 1.00 | mixed (external re-entry) | 1.7e-03 | 是 | 9.5e-04 |
| fla_gdr_chunk_varlen_bwd | dbeta | 1.00 | constant-level | 1.4e-08 | 是 | 2.0e-03 |
| fla_gdr_chunk_vfirst | o | 1.00 | constant-level | 1.3e-08 | 是 | 1.4e-03 |
| fla_gdr_chunk_vfirst | final_state | 1.00 | constant-level | 7.7e-09 | 是 | 7.9e-04 |
| fla_gdr_recurrent_basic | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.1e-07 |
| fla_gdr_recurrent_gate | o | 1.00 | constant-level | 1.9e-08 | 是 | 1.1e-07 |
| fla_gdr_recurrent_gva | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.1e-07 |
| fla_gdr_recurrent_h0_final | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.1e-07 |
| fla_gdr_recurrent_h0_final | final_state | 1.00 | none | 2.7e-16 | 否 | 2.0e-07 |
| fla_gdr_recurrent_l2_sigmoid | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.1e-07 |
| fla_gdr_recurrent_neg_eig | o | 1.00 | small | 2.0e-07 | 否 | 2.9e-07 |
| fla_gdr_recurrent_t100 | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.0e-07 |
| fla_gdr_recurrent_varlen | o | 1.00 | constant-level | 1.7e-08 | 是 | 2.1e-07 |
| fla_gdr_recurrent_varlen | final_state | 1.00 | none | 2.7e-16 | 是 | 2.0e-07 |
| fla_gdr_recurrent_vfirst | o | 1.00 | constant-level | 1.7e-08 | 否 | 2.1e-07 |
| fla_gdr_recurrent_vfirst | final_state | 1.00 | none | 2.7e-16 | 否 | 2.1e-07 |

报错：
- fla_gdr_chunk_gate_bwd: IndexError: index 2 is out of bounds for axis 0 with size 2

## census/tutorials

用例 9，输出 17；e_sem 分档：constant-level 3，mixed (external re-entry) 6，none 2，unresolved 6；报错 0；不评（非 Triton 写出或之后被改）0

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| tut_attention_causal | o | 1.00 | constant-level | 1.2e-08 | 否 | 2.4e-04 |
| tut_attention_causal_bwd | dq | 1.00 | mixed (external re-entry) | 3.8e-04 | 是 | 3.2e-04 |
| tut_attention_causal_bwd | dk | 1.00 | mixed (external re-entry) | 3.1e-04 | 否 | 3.1e-04 |
| tut_attention_causal_bwd | dv | 1.00 | mixed (external re-entry) | 3.0e-04 | 否 | 2.9e-04 |
| tut_attention_d128_causal | o | 1.00 | constant-level | 2.8e-08 | 否 | 2.4e-04 |
| tut_attention_plain | o | 1.00 | constant-level | 1.8e-08 | 是 | 2.7e-04 |
| tut_attention_plain_bwd | dq | 1.00 | mixed (external re-entry) | 9.9e-01 | 是 | 3.0e-04 |
| tut_attention_plain_bwd | dk | 1.00 | mixed (external re-entry) | 9.9e-01 | 是 | 2.9e-04 |
| tut_attention_plain_bwd | dv | 1.00 | mixed (external re-entry) | 9.9e-01 | 否 | 2.9e-04 |
| tut_layernorm_fp32 | y | 1.00 | none | 3.2e-14 | 是 | 6.8e-08 |
| tut_layernorm_fp32_bwd | dx | 0.00 | unresolved | — | — | — |
| tut_layernorm_fp32_bwd | dw | 0.00 | unresolved | — | — | — |
| tut_layernorm_fp32_bwd | db | 0.00 | unresolved | — | — | — |
| tut_layernorm_n1000 | y | 1.00 | none | 3.2e-14 | 是 | 7.0e-08 |
| tut_layernorm_n1000_bwd | dx | 0.00 | unresolved | — | — | — |
| tut_layernorm_n1000_bwd | dw | 0.00 | unresolved | — | — | — |
| tut_layernorm_n1000_bwd | db | 0.00 | unresolved | — | — | — |

## census/radam

用例 4，输出 12；e_sem 分档：candidate 3，constant-level 6，none 2，small 1；报错 0；不评（非 Triton 写出或之后被改）0

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| opt_radam | param | 1.00 | small | 7.3e-07 | 是 | 3.5e-07 |
| opt_radam | exp_avg | 1.00 | constant-level | 1.0e-08 | 是 | 3.5e-08 |
| opt_radam | exp_avg_sq | 1.00 | constant-level | 6.9e-11 | 是 | 5.1e-08 |
| opt_radam_b9995_step5 | param | 1.00 | candidate | 1.5e-02 | 是 | 2.4e-07 |
| opt_radam_b9995_step5 | exp_avg | 1.00 | constant-level | 1.1e-08 | 是 | 3.2e-08 |
| opt_radam_b9995_step5 | exp_avg_sq | 1.00 | constant-level | 2.8e-08 | 否 | 3.4e-08 |
| opt_radam_b9999_step2 | param | 1.00 | candidate | 8.2e-03 | 是 | 8.2e-03 |
| opt_radam_b9999_step2 | exp_avg | 1.00 | constant-level | 1.4e-08 | 否 | 3.5e-08 |
| opt_radam_b9999_step2 | exp_avg_sq | 1.00 | none | 7.9e-17 | 是 | 5.0e-08 |
| opt_radam_b9999_step6 | param | 1.00 | candidate | 2.9e-05 | 是 | 7.3e-06 |
| opt_radam_b9999_step6 | exp_avg | 1.00 | constant-level | 1.1e-08 | 是 | 3.1e-08 |
| opt_radam_b9999_step6 | exp_avg_sq | 1.00 | none | 8.2e-17 | 是 | 3.9e-08 |

## census/vllm

用例 15，输出 15；e_sem 分档：candidate 3，constant-level 1，none 11；报错 0；不评（非 Triton 写出或之后被改）0

| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |
|---|---|---|---|---|---|---|
| ua_bidir | out | 1.00 | none | 2.9e-16 | 否 | 2.6e-04 |
| ua_bidir_sw24_gqa4 | out | 1.00 | candidate | 1.1e-01 | 否 | 2.5e-04 |
| ua_bidir_sw8_mha | out | 1.00 | candidate | 9.1e-01 | 否 | 2.5e-04 |
| ua_bidir_sw8_qpkv16 | out | 1.00 | none | 2.2e-16 | 否 | 2.5e-04 |
| ua_causal | out | 1.00 | none | 2.4e-16 | 是 | 2.3e-04 |
| ua_causal_alibi | out | 1.00 | none | 2.0e-16 | 否 | 2.3e-04 |
| ua_causal_gqa4_d80 | out | 1.00 | constant-level | 9.0e-09 | 是 | 2.3e-04 |
| ua_causal_sinks | out | 1.00 | none | 2.6e-16 | 是 | 2.5e-04 |
| ua_causal_sinks_sw24 | out | 1.00 | none | 2.3e-16 | 否 | 2.5e-04 |
| ua_causal_softcap | out | 1.00 | none | 3.0e-16 | 否 | 2.3e-04 |
| ua_causal_sw24 | out | 1.00 | none | 2.2e-16 | 否 | 2.3e-04 |
| ua_causal_sw24_nan_stale | out | 1.00 | none | 2.2e-16 | 否 | 2.3e-04 |
| ua_decode_3d | out | 1.00 | none | 2.6e-16 | 是 | 2.5e-04 |
| ua_decode_3d_sinks_sw | out | 1.00 | none | 2.5e-16 | 是 | 2.5e-04 |
| ua_perseq_causal_sw8 | out | 1.00 | candidate | 2.2e-01 | 是 | 2.3e-04 |

