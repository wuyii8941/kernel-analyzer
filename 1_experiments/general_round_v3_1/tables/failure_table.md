# 第 6 项：失败分类表（通用能力轮的全部运行）

五类计数：{'enclosure too wide': 6, 'not counted (special values: target undefined or infinite)': 11, 'not counted (no f / clause pending: outside the measurement)': 17, 'binding': 101, 'binding (semantic verdict only)': 85, 'statistics insufficient': 4}

分类规则：`kernel_analyzer.measure.classify_failure`（原因字符串 → 类别，映射表随每份报告输出）。「binding (semantic verdict only)」是读到非 Triton 中间值的输出：数值差异照常测量，只是不出语义结论；「not counted」是规格不覆盖或条款待审阅的条件，不属于工具失败。

| 来源 | 用例 | 输出 | 原因 | 类别 |
|---|---|---|---|---|
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_gelu_erf_gauss_contiguous | out | resolved fraction 0.9966 < 1 | enclosure too wide |
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_gelu_tanh_gauss_contiguous | out | resolved fraction 0.9966 < 1 | enclosure too wide |
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_geglu_gauss_contiguous | out | resolved fraction 0.9983 < 1 | enclosure too wide |
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_gelu_erf_huge_contiguous | out | resolved fraction 0.5051 < 1 | enclosure too wide |
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_gelu_tanh_huge_contiguous | out | resolved fraction 0.4663 < 1 | enclosure too wide |
| fr_modeB_v3_1/activations | inductor_cuda_float32/acti_geglu_huge_contiguous | out | resolved fraction 0.5067 < 1 | enclosure too wide |
| fr_modeB_v3_1/attention | flex_attention_float32/att_equal_fully_masked_row_mha_sNone_d16 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/attention | flex_attention_float32/att_equal_none_mha_sNone_d72 | — | 2.10 does not compile flex decoding for head_dim 72 (#164931) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/attention | flex_attention_float32/att_qltk_fully_masked_row_mha_sNone_d16 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/attention | flex_attention_float32/att_qgtk_fully_masked_row_mha_sNone_d16 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/attention | flex_attention_float32/att_qgtk_sliding_window_mha_sNone_d16 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/attention | flex_attention_float32/att_qeq1_fully_masked_row_mha_sNone_d16 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_none_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qltk_none_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qgtk_none_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qeq1_none_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_causal_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_padding_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_fully_masked_row_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_sliding_window_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_none_gqa_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_none_mha_s0.5_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_none_mha_sNone_d64 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_equal_none_mha_sNone_d72 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qltk_causal_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qltk_padding_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qltk_fully_masked_row_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qltk_sliding_window_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qgtk_causal_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qgtk_padding_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qgtk_fully_masked_row_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qgtk_sliding_window_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qeq1_causal_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qeq1_padding_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qeq1_fully_masked_row_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/attention | inductor_attention_float32/att_qeq1_sliding_window_mha_sNone_d16 | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_sum_normal_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_padNone_mn1.0_nofreq | weight_after | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_padNone_mn1.0_nofreq | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_mean_normal_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_max_normal_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_sum_empty_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_sum_single_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_sum_normal_psw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_pad0_mnNone_nofreq_sum_normal_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_bag_padNone_mnNone_nofreq_mean_empty_nopsw | out | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_pad0_mn1.0_nofreq | weight_after | not written by Triton | binding |
| fr_modeB_v3_1/embedding | inductor_cuda_float32/emb_embedding_pad0_mn1.0_nofreq | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_cat_split_distinct_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_view_alias_distinct_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_gather_distinct_transposed | dx | not written by Triton | binding |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_gather_distinct_transposed | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_cat_split_repeated_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_cat_split_all_same_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_view_alias_repeated_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/gather_layout | inductor_cuda_float32/gath_view_alias_all_same_contiguous | — | outside the contract | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_contiguous_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_contiguous_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_contiguous_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_batch-broadcast_contiguous_None_gauss | — | refused: BASE-A4 (broadcast outside the spec) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_offset_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_offset_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_offset_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_vector_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_vector_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_vector_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_broadcast_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_broadcast_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_broadcast_gauss | dbias | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_broadcast_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_small_ints | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_small_ints | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_small_ints | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_cancel | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_cancel | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_cancel | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_mixed_scale | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_mixed_scale | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_contiguous_None_mixed_scale | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_transposed_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_transposed_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_transposed_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_offset_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_offset_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_offset_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_transposed_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_transposed_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_transposed_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_offset_None_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_offset_None_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_37x129x129x61_offset_None_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_batch-broadcast_transposed_None_gauss | — | refused: BASE-A4 (broadcast outside the spec) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_batch-broadcast_offset_None_gauss | — | refused: BASE-A4 (broadcast outside the spec) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_vector_gauss | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_vector_gauss | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_mx0x0xn_contiguous_vector_gauss | out | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_small_ints | da | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_small_ints | db | not written by Triton | binding |
| fr_modeB_v3_1/matmul_linear | inductor_cuda_float32/matm_1xkxkx1_transposed_None_small_ints | out | not written by Triton | binding |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_None_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_2_4_None_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_8_None_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_exact_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_overflow_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_None_ties | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_None_one_expert | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_2_8_None_random | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_1_4_overflow_one_expert | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/moe | vectorised_compiled/moe_2_4_None_ties | out | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state0.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state0.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state1.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state1.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state2.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.0 | state2.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state0.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state0.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state1.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state1.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state2.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_cold_min_wd0.0 | state2.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state0.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state0.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state0.max_exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state1.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state1.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state1.max_exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state2.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state2.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adam_amsgrad_cold_min_wd0.0 | state2.max_exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | state0.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | state1.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_min_wd0.0 | state2.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | state0.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | state1.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_dampening_cold_min_wd0.0 | state2.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state0.square_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state0.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state0.grad_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state1.square_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state1.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state1.grad_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state2.square_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state2.momentum_buffer | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_rmsprop_centered_cold_min_wd0.0 | state2.grad_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adafactor_cold_min_wd0.0 | — | contract v2: X 契约外，待审阅 | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state0.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state0.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state1.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state1.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state2.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_max_wd0.0 | state2.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | param0 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | param1 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | param2 | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state0.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state0.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state1.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state1.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state2.exp_avg | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_adamw_cold_min_wd0.1 | state2.exp_avg_sq | reads a non-Triton intermediate: no semantic verdict (numerical only) | binding (semantic verdict only) |
| fr_modeB_v3_1/optimizers | torch_compiled_step_cuda/opt_sgd_nesterov_cold_max_wd0.0 | — | O-D2 (SGD maximize placement), 3_audits/spec_issues_phase2.md | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_var_single_gauss_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_std_single_gauss_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_sum_single_neg_inf_row_1 | — | contract v2: X 契约外，待审阅 (not judged against f) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_mean_empty-extent_gauss_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_amax_empty-extent_gauss_1 | — | TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <built-in method amax of type object at 0x7f903d3f3980>(*(FakeTensor(..., devic | binding |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_amin_empty-extent_gauss_1 | — | TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <built-in method amin of type object at 0x7f903d3f3980>(*(FakeTensor(..., devic | binding |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_var_empty-extent_gauss_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_std_empty-extent_gauss_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_softmax_empty-extent_gauss_1 | — | contract v2: X 契约外，待审阅 (not judged against f) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_log_softmax_empty-extent_gauss_1 | — | contract v2: X 契约外，待审阅 (not judged against f) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_logsumexp_empty-extent_gauss_1 | — | contract v2: X 契约外，待审阅 (not judged against f) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_cumsum_empty-extent_gauss_1 | — | contract v2: X 契约外，待审阅 (not judged against f) | not counted (no f / clause pending: outside the measurement) |
| fr_modeB_v3_1/reductions | inductor_cuda_float32/redu_logsumexp_single_neg_inf_row_1 | out | missing elements are special values (NaN / inf: target undefined or infinite, contract class C) | not counted (special values: target undefined or infinite) |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wd | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wg | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wk | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wq | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wu | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | prenorm_block | d_wv | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | rope_attention | d_wo | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | rope_attention | out | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | gelu_mlp_layernorm | d_w1 | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | gelu_mlp_layernorm | d_w2 | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | groupnorm_silu | d_lin | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | ce_head | d_h | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | ce_head | d_w | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | moe_block | d_router | not written by Triton | binding |
| probe/G6 compositions (float32 Inductor, fwd + bwd) | moe_block | d_x | not written by Triton | binding |
| calibration | rare_tail effect 0.0 | R5 | cannot judge in 0.599 of 1000 replicates | statistics insufficient |
| calibration | rare_tail effect 0.005 | R5 | cannot judge in 0.612 of 1000 replicates | statistics insufficient |
| calibration | rare_tail effect 0.02 | R5 | cannot judge in 0.621 of 1000 replicates | statistics insufficient |
| calibration | rare_tail effect 0.25 | R5 | cannot judge in 0.621 of 1000 replicates | statistics insufficient |
