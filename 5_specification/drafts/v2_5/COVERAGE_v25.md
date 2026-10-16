# v2.5 on the 15-library survey (families that needed a definition or are random)

This batch was used to design v2.5; the numbers below say where v2.5 stands on it and are not a first-contact measurement.

| class | name | families | mapped to a v2.5 contract | still open |
|---|---|---|---|---|
| A | 线性注意力与状态递推 | 103 | 70 | 33 |
| B | 分页 / 变长 KV 注意力 | 23 | 13 | 10 |
| C | 稀疏注意力、索引器与 KV 压缩 | 53 | 21 | 32 |
| E | KV cache 写入、读出与 KV 量化 | 13 | 13 | 0 |
| F | 量化格式与量化 / 反量化 | 41 | 40 | 1 |
| G | 量化与低比特 GEMM | 13 | 13 | 0 |
| H | MoE：路由、排序对齐、分组 GEMM、EP 分发 | 39 | 32 | 7 |
| K | 块稀疏矩阵乘 | 3 | 0 | 3 |
| I | 采样、投机解码、哈希与水印 | 12 | 2 | 10 |
| J | 布局、元数据与调度 | 16 | 0 | 16 |
| M | RoPE 与位置编码变体 | 6 | 4 | 2 |
| L | LoRA | 2 | 0 | 2 |
| O | 其他 | 5 | 0 | 5 |
| R | 随机 | 15 | 11 | 4 |
| all | | 344 | 219 (64%) | 125 |

Before v2.5 (first contact), 453 of 797 families had a reference program (torch operation or composition) and 344 needed a definition or were random.

## Per family

| class | library | family | v2.5 contract or open item |
|---|---|---|---|
| A | flash-linear-attention (fla) | Common chunk state passing (chunk_fwd_h / chunk_bwd_dh) | A1_LINEAR |
| A | flash-linear-attention (fla) | Common chunk output forward (chunk_fwd_o) | A1_LINEAR |
| A | flash-linear-attention (fla) | Common chunk backward helpers (dq/dk/dw/dg, dv, dv_local) | A1_LINEAR |
| A | flash-linear-attention (fla) | Common chunk_h parallel/split variants (unreferenced) | A1_LINEAR |
| A | flash-linear-attention (fla) | Common fused_recurrent gated linear attention (fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | Common fused_chunk scalar-gated linear attention (fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] common chunk h/o (fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | GLA chunk forward (intra-chunk A, output o) | A1_LINEAR |
| A | flash-linear-attention (fla) | GLA chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] GLA chunk (fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | Simple GLA parallel (quadratic form, fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] Simple GLA fused_recurrent & parallel | A1_LINEAR |
| A | flash-linear-attention (fla) | Based parallel (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Based fused_chunk (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | ReBased parallel (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | HGRN fused_recurrent (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | HGRN chunk (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | ABC chunk forward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | ABC chunk backward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | GSA chunk (k-pass kernels, fwd+bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | GSA fused_recurrent single-step inference | A1_LINEAR |
| A | flash-linear-attention (fla) | RWKV4 fused_recurrent WKV (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | RWKV6 chunk forward (cumsum, intra-chunk A) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | RWKV6 chunk backward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | RWKV6 fused_recurrent (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | TTT-Linear chunk forward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | TTT-Linear chunk backward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | TTT-Linear fused_chunk (fwd+bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Log-linear attention chunk forward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Log-linear attention chunk backward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Delta-rule chunk state recurrence h (fwd h/v_new/ht, bwd dh/dh0/dv) [shared by DeltaNet, GDN, Comba, PGDN, KDA, GDN2, PKDA] | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Context-parallel / intra-card CP state pre-process + merge (S_ext, M affine chain) for delta-rule chunk h | A1_LINEAR |
| A | flash-linear-attention (fla) | DeltaNet fused_recurrent forward+backward | A1_LINEAR |
| A | flash-linear-attention (fla) | DeltaNet chunk WY representation (recompute_w_u fwd, prepare_wy_repr bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | DeltaNet parallel forward (materialised attention, head-first) [broken/dead] | A1_LINEAR |
| A | flash-linear-attention (fla) | Gated DeltaNet chunk intra-chunk WY (fused kkt+solve_tril, recompute_w_u fwd, prepare_wy_repr bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | Gated DeltaNet fused_recurrent forward (inference; supports g, gk, gv gates) | A1_LINEAR |
| A | flash-linear-attention (fla) | Gated DeltaProduct chunk forward (h over interleaved Householder steps, o) + unused bwd dhu | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Comba chunk WY (p·k^T kkt with exclusive decay, recompute_w_u fwd, prepare_wy_repr bwd) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Comba fused_recurrent forward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Gated Oja rule chunk forward (v v^T kkt with per-V gate, WY, h, o) | A1_LINEAR |
| A | flash-linear-attention (fla) | Gated Oja rule chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | Gated Oja rule fused_recurrent forward | A1_LINEAR |
| A | flash-linear-attention (fla) | MesaNet chunk forward (h_kk/h_kv states, chunked CG solve, output) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | MesaNet chunk backward (adjoint CG solve, intra-chunk dH_kk/dH_kv) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | MesaNet decoding one step | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | Preconditioned Gated DeltaNet chunk (asymmetric precond kkt fwd, precond WY bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | Preconditioned Gated DeltaNet fused_recurrent forward | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] delta-rule chunk state recurrence h fwd/bwd (common) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] GDN WY representation (recompute_w_u fwd, split prepare_wy_repr bwd) | A1_LINEAR |
| A | flash-linear-attention (fla) | KDA chunk forward (intra Aqk/Akk + fused solve_tril, WY w/u) | A1_LINEAR |
| A | flash-linear-attention (fla) | KDA chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | KDA fused_recurrent forward (inference/decoding) | A1_LINEAR |
| A | flash-linear-attention (fla) | Preconditioned KDA chunk forward (intra + solve, WY) | A1_LINEAR |
| A | flash-linear-attention (fla) | Preconditioned KDA chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | Preconditioned KDA fused_recurrent forward (inference) | A1_LINEAR |
| A | flash-linear-attention (fla) | GDN2 (Gated DeltaNet-2) chunk forward (intra + solve, WY) | A1_LINEAR |
| A | flash-linear-attention (fla) | GDN2 chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | GDN2 fused_recurrent forward (inference) | A1_LINEAR |
| A | flash-linear-attention (fla) | DPLR (generalized delta rule) chunk forward (A, WY, h, o) | A1_LINEAR |
| A | flash-linear-attention (fla) | DPLR chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | DPLR fused_recurrent forward (inference) | A1_LINEAR |
| A | flash-linear-attention (fla) | IPLR (identity-plus-low-rank delta rule) chunk forward | A1_LINEAR |
| A | flash-linear-attention (fla) | IPLR fused_recurrent forward+backward | A1_LINEAR |
| A | flash-linear-attention (fla) | RWKV7 fused_mul_recurrent forward (kk/a parameterised DPLR) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | [Ascend] KDA chunk forward (intra/solve, token-parallel, WY) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] KDA chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] KDA fused_recurrent forward | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] GDN2 chunk forward (intra/solve) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] GDN2 chunk backward (WY) | A1_LINEAR |
| A | flash-linear-attention (fla) | [Ascend] GDN2 fused_recurrent forward | A1_LINEAR |
| A | flash-linear-attention (fla) | DeltaFormer u-recurrence forward (chunked softmax-weighted delta) | A1_LINEAR |
| A | flash-linear-attention (fla) | DeltaFormer backward | A1_LINEAR |
| A | flash-linear-attention (fla) | PaTH attention forward (intra-chunk preprocess + parallel inter-chunk) | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | PaTH key-cache preparation for decoding | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | PaTH attention backward | open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...) |
| A | flash-linear-attention (fla) | CyFA chunk forward (point write + fused readout) | A1_LINEAR |
| A | flash-linear-attention (fla) | CyFA chunk backward | A1_LINEAR |
| A | flash-linear-attention (fla) | CyFA fused recurrent inference / single-token decode | A1_LINEAR |
| A | flash-linear-attention (fla) | ATK preconditioner forward (chunked elementwise k^2 accumulator) | A1_LINEAR |
| A | flash-linear-attention (fla) | ATK preconditioner backward | open: not matched |
| A | lightllm | gated_delta_rule_chunk_prefill | A1_LINEAR |
| A | lightllm | gated_delta_rule_recurrent_decode | A1_LINEAR |
| A | lightllm | kda_kimi_delta_attention | A1_LINEAR |
| A | lightllm | causal_conv1d_state_update | open: convolution state (A1.D11 / D7) |
| A | Lightning Attention | lightning attention-2 with per-head exponential decay (chunk-loop variant) | A1_LINEAR |
| A | Lightning Attention | lightning attention-2 with decay, fully-parallel variant | A1_LINEAR |
| A | sglang | gdn_fused_proj_causal_conv1d_update | open: convolution state (A1.D11 / D7) |
| A | sglang | xpu_gdn_chunk_intermediates | A1_LINEAR |
| A | sglang | lightning_linear_attention_decay | A1_LINEAR |
| A | sglang | seg_la_segmented_linear_attention | A1_LINEAR |
| A | sglang | inkling_sconv_causal_conv_with_cache | open: convolution state (A1.D11 / D7) |
| A | sglang | gdn_chunked_prefill | A1_LINEAR |
| A | sglang | kda_chunked_prefill | A1_LINEAR |
| A | sglang | gated_delta_rule_recurrent_fwd | A1_LINEAR |
| A | sglang | gdn_kda_packed_decode | A1_LINEAR |
| A | sglang | gdn_kda_spec_verify_recurrent | A1_LINEAR |
| A | sglang | replayssm_ring_decode_fold | A1_LINEAR |
| A | sglang | causal_conv1d | open: convolution state (A1.D11 / D7) |
| A | sglang | lfm2_short_conv | open: convolution state (A1.D11 / D7) |
| A | sglang | mamba_selective_state_update | A1_LINEAR |
| A | sglang | mamba2_ssd_chunk_scan | A1_LINEAR |
| A | sglang | sana_wm_bidirectional_gdn_chunkwise | A1_LINEAR |
| B | flash-linear-attention (fla) | Wall attention cached decode (pre-rescaled KV cache) | B1_PAGED |
| B | kernl | experimental llama: flash attention with start_position, fp16 KV | B1_PAGED |
| B | lightllm | prefill_paged_extend_attention | B1_PAGED |
| B | lightllm | paged_decode_attention_mha_gqa | B1_PAGED |
| B | lightllm | quantized_kv_paged_decode_attention | open: quantized KV target (B1.D8) |
| B | lightllm | mla_paged_decode_attention | open: MLA absorbed form |
| B | lightllm | mla_prefill_attention | open: MLA absorbed form |
| B | lightllm | mla_fp8_kv_attention | open: MLA absorbed form |
| B | lightllm | prefill_attention_image_bidirectional | B1_PAGED |
| B | lightllm | alibi_attention | B1_PAGED |
| B | LMDeploy | varlen_flash_attention_prefill | B1_PAGED |
| B | LMDeploy | paged_decode_attention | B1_PAGED |
| B | LMDeploy | paged_decode_attention_quant_kv | open: quantized KV target (B1.D8) |
| B | SageAttention (Triton parts) | INT8-QK / FP16-PV attention on pre-quantised inputs, non-causal, optional mask + LSE (dense) | open: quantized KV target (B1.D8) |
| B | SageAttention (Triton parts) | INT8-QK / FP16-PV attention on pre-quantised inputs, causal (dense) | open: quantized KV target (B1.D8) |
| B | SageAttention (Triton parts) | INT8-QK / FP16-PV attention on pre-quantised inputs, varlen (causal and non-causal) | open: quantized KV target (B1.D8) |
| B | sglang | triton_decode_attention_paged | B1_PAGED |
| B | sglang | triton_extend_attention_paged_prefix | B1_PAGED |
| B | sglang | dense_varlen_prefill_attention | B1_PAGED |
| B | sglang | spec_verify_splitkv_attention | B1_PAGED |
| B | sglang | rocm_mla_decode_fused_rope | open: MLA absorbed form |
| B | sglang | aiter_unified_attention_3d_mtp | B1_PAGED |
| B | sglang | swa_mla_fallback_gather_and_mask | open: MLA absorbed form |
| C | flash-linear-attention (fla) | NSA compressed-block attention forward | open: compressed KV and ring buffers (C1.D7) |
| C | flash-linear-attention (fla) | NSA compressed-block attention backward | open: compressed KV and ring buffers (C1.D7) |
| C | flash-linear-attention (fla) | NSA top-k block selection | C1_SPARSE |
| C | flash-linear-attention (fla) | NSA selected-block attention forward | C1_SPARSE |
| C | flash-linear-attention (fla) | NSA selected-block attention backward | C1_SPARSE |
| C | lightllm | dsa_fp8_paged_mqa_logits | open: KV byte layouts (C1.D5) |
| C | lightllm | flashmla_fp8_kv_cache_quant | open: KV byte layouts (C1.D5) |
| C | lightllm | glm5_kpool_compress | open: compressed KV and ring buffers (C1.D7) |
| C | lightllm | glm5_expand_topk | C1_SPARSE |
| C | lightllm | nsa_indexer_ks_ke | C1_SPARSE |
| C | LMDeploy | bitonic_topk_varlen | C1_SPARSE |
| C | LMDeploy | dsv4_sparse_index_builders | C1_SPARSE |
| C | LMDeploy | dsv4_window_pack_fp8 | open: KV byte layouts (C1.D5) |
| C | LMDeploy | dsv4_flatten_kv | open: not matched |
| C | LMDeploy | dsv4_kv_compressor | open: compressed KV and ring buffers (C1.D7) |
| C | LMDeploy | dsv4_fill_compressed_kv_fp8 | open: compressed KV and ring buffers (C1.D7) |
| C | LMDeploy | dsa_indexer_qk_prep_fp8 | open: KV byte layouts (C1.D5) |
| C | LMDeploy | dsa_indexer_k_cache_flatten | C1_SPARSE |
| C | LMDeploy | dsa_fp8_index_logits | open: KV byte layouts (C1.D5) |
| C | sglang | flashmla_sm120_sparse_decode_fp8 | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_fused_qk_norm_rope_fp8_paged_store | open: KV byte layouts (C1.D5) |
| C | sglang | dsa_kpool_compress_fp8_write | open: compressed KV and ring buffers (C1.D7) |
| C | sglang | dsa_kpool_bookkeeping | C1_SPARSE |
| C | sglang | minicpm_sparse_compress_k | open: compressed KV and ring buffers (C1.D7) |
| C | sglang | minimax_gluon_prefill_layout | open: not matched |
| C | sglang | qsa_sparse_gqa_attention | C1_SPARSE |
| C | sglang | qsa_kv_gather_and_metadata | open: not matched |
| C | sglang | dsv4_fp4_indexer_quantize_and_index_k_cache | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_hip_flydsl_fp4_qk_pack | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_rope_tail_fp4_fake_quant | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_fp4_index_logits | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_indexer_topk_finalize | C1_SPARSE |
| C | sglang | dsv4_compressed_attn_metadata | open: compressed KV and ring buffers (C1.D7) |
| C | sglang | dsv4_compressor_softmax_pool_hip | open: compressed KV and ring buffers (C1.D7) |
| C | sglang | dsv4_compress_norm_rope_inplace_hip | open: compressed KV and ring buffers (C1.D7) |
| C | sglang | dsv41_c2_decode_pair_pool | open: not matched |
| C | sglang | dsv4_v4_kv_cache_quant_store_dequant | open: KV byte layouts (C1.D5) |
| C | sglang | dsv4_unified_kv_sparse_attention_with_sink | C1_SPARSE |
| C | sglang | dsv4_swapab_decode_combine_sink_inverse_rope | open: not matched |
| C | sglang | dsv4_sparse_prefill_swa_topk_index_combine | C1_SPARSE |
| C | sglang | dsv4_hip_fp4_indexer_schedule | open: KV byte layouts (C1.D5) |
| C | sglang | dsa_fp8_kv_cache_pack_quant_dequant | open: KV byte layouts (C1.D5) |
| C | sglang | dsa_sparse_mla_attention_fwd | C1_SPARSE |
| C | sglang | dsa_kpool_fused_metadata | C1_SPARSE |
| C | sglang | minimax_sparse_indexer_block_score | C1_SPARSE |
| C | sglang | minimax_sparse_block_topk_select | C1_SPARSE |
| C | sglang | minimax_sparse_indexer_context_parallel | C1_SPARSE |
| C | sglang | minimax_sparse_topk_attention | C1_SPARSE |
| C | sglang | dsv4_fused_gather_dequant_sparse_decode | open: KV byte layouts (C1.D5) |
| C | sglang | qwen38_qsa_packed_decode | open: not matched |
| C | sglang | sla_block_sparse_attention_fwd | C1_SPARSE |
| C | sglang | vsa_h3_block_sparse_attention_fwd | C1_SPARSE |
| C | sglang | dsv4_fp8_paged_store | open: KV byte layouts (C1.D5) |
| E | lightllm | int8_int4_kv_cache_quant | F4_INTQ + B1_PAGED |
| E | lightllm | mla_kv_fp8_quant_copy | F7_FP8SCALE + B1_PAGED |
| E | lightllm | destindex_copy_kv_fp8_quant | F7_FP8SCALE + B1_PAGED |
| E | LMDeploy | fill_paged_kv_cache | B1_PAGED |
| E | LMDeploy | fill_paged_kv_cache_int_turbo_quant | F4_INTQ + B1_PAGED |
| E | LMDeploy | fill_paged_kv_cache_fp8_per_tensor | F7_FP8SCALE + B1_PAGED |
| E | LMDeploy | fill_paged_kv_cache_blocked_fp8 | F7_FP8SCALE + B1_PAGED |
| E | LMDeploy | flatten_paged_kv_cache | B1_PAGED |
| E | LMDeploy | flatten_paged_kv_cache_int_turbo_dequant | F4_INTQ + B1_PAGED |
| E | LMDeploy | flatten_paged_kv_cache_fp8_per_tensor | F7_FP8SCALE + B1_PAGED |
| E | LMDeploy | flatten_mla_fp8_kv_cache | F7_FP8SCALE + B1_PAGED |
| E | sglang | mxfp8_paged_kv_store | F7_FP8SCALE + B1_PAGED |
| E | sglang | kv_cache_fp8_scaled_write | F7_FP8SCALE + B1_PAGED |
| F | applied-ai | fp8_groupwise_quant_1x128 | F7_FP8SCALE |
| F | gemlite | act_quant_per_token (int8/fp8 absmax per row) | F4_INTQ |
| F | gemlite | act_quant_per_block (int8/fp8, 1x128 groups) | F4_INTQ |
| F | gemlite | act_quant_mxfp8 (e4m3 + E8M0, group 32) | F2_MX |
| F | gemlite | act_quant_mxfp4 (e2m1 packed + E8M0, group 32) | F2_MX |
| F | gemlite | act_quant_nvfp4 (e2m1 packed + e4m3 scale, group 16, global meta scale) | F3_NVFP4 |
| F | kernl | experimental llama: FP8 (e5m2 bit-pattern) weight / KV variants (USE_FP8=True) | F7_FP8SCALE |
| F | kernl | experimental fp8 <-> fp16 conversion kernels | F7_FP8SCALE |
| F | lightllm | moe_silu_mul_fp8_group_quant_masked | F7_FP8SCALE |
| F | lightllm | fp8_activation_group_quant | F7_FP8SCALE |
| F | lightllm | fp8_weight_quant | F7_FP8SCALE |
| F | lightllm | fp8_q_per_head_quant | F7_FP8SCALE |
| F | lightllm | hadamard_fp8_ue8m0_quant | F2_MX |
| F | LMDeploy | dsv4_swiglu_quant_fp8 | F7_FP8SCALE |
| F | LMDeploy | per_token_quant | F7_FP8SCALE |
| F | LMDeploy | rms_norm_dynamic_quant | v2.4 NORM + F7_FP8SCALE / F4_INTQ |
| F | LMDeploy | blocked_fp8_activation_quant | F7_FP8SCALE |
| F | SageAttention (Triton parts) | per-block INT8 quantisation of Q and K (dense) | F4_INTQ |
| F | SageAttention (Triton parts) | per-block INT8 quantisation of Q and K (varlen) | F4_INTQ |
| F | SageAttention (Triton parts) | per-thread INT8 quantisation of Q and K (for CUDA mma attention kernels) | F4_INTQ |
| F | sglang | mla_kv_pack_fp8_quant | F7_FP8SCALE |
| F | sglang | situ_mul_fp8_group_quant | F7_FP8SCALE |
| F | sglang | fp8_fa_sm120_qkv_prepare | F7_FP8SCALE |
| F | sglang | fp8_per_token_group_act_quant_model_local | F7_FP8SCALE |
| F | sglang | moe_act_quant_fp8 | F7_FP8SCALE |
| F | sglang | moe_fp8_requant | F7_FP8SCALE |
| F | sglang | mxfp4_upscale_decode | F2_MX |
| F | sglang | fp8_rowwise_quant | F7_FP8SCALE |
| F | sglang | aiter_per_tensor_fp8_quant | F7_FP8SCALE |
| F | sglang | swiglu_limit_fp8grid_hip | F7_FP8SCALE |
| F | sglang | attn_output_sigmoid_gate | open: not a quantization kernel (gating) |
| F | sglang | dsv4_wo_a_mxfp8_epilogue | F2_MX |
| F | sglang | rmsnorm_mxfp8_epilogue | F2_MX |
| F | sglang | per_token_group_quant_8bit | F7_FP8SCALE |
| F | sglang | int8_activation_quant | F4_INTQ |
| F | sglang | per_tensor_dynamic_fp8_mla | F7_FP8SCALE |
| F | sglang | mxfp8_quantize | F2_MX |
| F | sglang | mxfp8_fused_producers | F2_MX |
| F | sglang | rmsnorm_fp8grid_fake_quant_hip | F7_FP8SCALE |
| F | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | quantize to MX (mxfp4/mxfp8 with E8M0 scales; NVFP4 with E4M3 scales) | F3_NVFP4 |
| F | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | dequantize from MX (mxfp4/mxfp8 values, E8M0 or E4M3 scales) | F2_MX |
| G | applied-ai | gptq_w4a16_dequant_gemm | F5_PACK + F4_INTQ |
| G | DeepSpeed | fp8_weight_only_gemm (DeepSpeed FP_Quantize 8-bit weights x bf16/fp16 activations) | F5_PACK + F4_INTQ |
| G | gemlite | int_dequant_matmul (A16Wn / A8Wn / A8W8, grouped or channel/block scaled) | F5_PACK + F4_INTQ |
| G | gemlite | mx_scaled_matmul (MXFP8/MXFP4/NVFP4 via tl.dot_scaled) | F2_MX / F3_NVFP4 + MATMUL |
| G | gemlite | mx_dequant_gemv (software-dequant MX GEMV) | F2_MX / F3_NVFP4 + MATMUL |
| G | gemlite | bitplane_lowbit_gemm (experimental A16W3 / A16W5) | F5_PACK + F4_INTQ |
| G | LMDeploy | w8a8_linear | F5_PACK + F4_INTQ |
| G | LMDeploy | awq_w4a16_linear | F5_PACK + F4_INTQ |
| G | LMDeploy | blocked_fp8_gemm | F5_PACK + F4_INTQ |
| G | LMDeploy | blocked_fp8_gemm_gluon_hopper | F5_PACK + F4_INTQ |
| G | sglang | bf16_batched_gemm_fp8grid_gfx95 | F5_PACK + F4_INTQ |
| G | sglang | awq_int4 | F5_PACK + F4_INTQ |
| G | sglang | mxfp8_shuffled_gemm_gfx950 | F2_MX / F3_NVFP4 + MATMUL |
| H | applied-ai | gptq_w4a16_fused_moe_gemm | H2_ALIGN + F5_PACK |
| H | lightllm | fused_moe_grouped_gemm | H1_ROUTING |
| H | lightllm | moe_softmax_topk_routing | H1_ROUTING |
| H | lightllm | moe_grouped_topk_routing | H1_ROUTING |
| H | lightllm | eplb_redundant_expert_remap | H2_ALIGN |
| H | LMDeploy | moe_route_sort_metadata | H2_ALIGN |
| H | LMDeploy | moe_grouped_gemm_w8a8 | H1_ROUTING |
| H | LMDeploy | moe_grouped_gemm_blocked_fp8 | open: quantized MoE layouts (H2.D5) |
| H | LMDeploy | moe_grouped_gemm_fp8_x_fp4_v4 | open: quantized MoE layouts (H2.D5) |
| H | LMDeploy | moe_grouped_gemm_w4a16_compressed_tensors | H2_ALIGN + F5_PACK |
| H | LMDeploy | ep_token_scatter | H2_ALIGN |
| H | LMDeploy | ep_token_scatter_fp8 | open: quantized MoE layouts (H2.D5) |
| H | LMDeploy | moe_router_noaux_tc | H1_ROUTING |
| H | sglang | musa_moe_topk_gating | H1_ROUTING |
| H | sglang | moe_lora_virtual_expert_expand | H2_ALIGN |
| H | sglang | llada2_block_expert_routing | H1_ROUTING |
| H | sglang | fused_moe_triton_gemm | H2_ALIGN |
| H | sglang | fused_moe_wna16_gemm | H2_ALIGN |
| H | sglang | moe_atomic_dispatch_scatter | K14_COLL |
| H | sglang | moe_align_block_size_small | H2_ALIGN |
| H | sglang | aiter_moe_sorting_small_mxfp8 | open: quantized MoE layouts (H2.D5) |
| H | sglang | mxfp8_grouped_moe_gemm_gfx950 | open: quantized MoE layouts (H2.D5) |
| H | sglang | fused_router_gate_topk | H1_ROUTING |
| H | sglang | rocm_router_gate_sqrtsoftplus | H1_ROUTING |
| H | sglang | softcap_router_gemm_topk | H1_ROUTING |
| H | sglang | waterfill_shared_expert_expand | H2_ALIGN |
| H | sglang | group_limited_topk_routing | H1_ROUTING |
| H | sglang | subblock_sparse_router_approx_topk | H1_ROUTING |
| H | sglang | grouped_fp8_quant_moe_layouts | open: quantized MoE layouts (H2.D5) |
| H | sglang | aot_ref_moe_align_block_size | H2_ALIGN |
| H | sglang | lora_moe_grouped_gemm | H1_ROUTING |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | matmul_ogs MoE GEMM, non-persistent (gather/scatter/ragged/MX/flex) | H2_ALIGN |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | matmul_ogs MoE GEMM, persistent TMA | H2_ALIGN |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | masked/scaled reduction over one axis (split-K reducer, MoE combine) | K14_COLL |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | reduce backward | open: not matched |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | top-k routing (sorted top-k + optional softmax + bitmatrix) | H2_ALIGN |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | top-k backward (softmax gradient scattered to selected experts) | H2_ALIGN |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | expert-parallel token dispatch/combine over symmetric memory | K14_COLL |
| H | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | ragged segment pad/unpad to block-aligned rows | H2_ALIGN |
| K | stk | block_sparse_matmul_sdd (dense x dense -> BCSR-sparse output on given topology) | open: block-sparse metadata |
| K | stk | block_sparse_matmul_dsd (BCSR-sparse x dense -> dense) | open: block-sparse metadata |
| K | stk | block_sparse_matmul_dds (dense x BCSR-sparse -> dense) | open: block-sparse metadata |
| I | lightllm | sampling_penalties | R1_RANDOM (D2 open) |
| I | sglang | sampling.murmur_hash32 | open: speculative decoding / hashing / watermark |
| I | sglang | sampling.textseal_watermark_select | open: speculative decoding / hashing / watermark |
| I | sglang | sampling.textseal_context_state | R1_RANDOM (D2 open) |
| I | sglang | spec.eagle_tree_build_and_greedy_verify | open: speculative decoding / hashing / watermark |
| I | sglang | spec.chain_rejection_sampling | open: speculative decoding / hashing / watermark |
| I | sglang | spec.selector_walk | open: speculative decoding / hashing / watermark |
| I | sglang | spec.lilicorr_topk_lse | open: speculative decoding / hashing / watermark |
| I | sglang | spec.dspark_schedule_verify_lens | open: speculative decoding / hashing / watermark |
| I | sglang | engram_ngram_hash | open: speculative decoding / hashing / watermark |
| I | sglang | qwen4_ple_ngram_hash | open: speculative decoding / hashing / watermark |
| I | sglang | gpu_tensor_hash | open: speculative decoding / hashing / watermark |
| J | lightllm | flashinfer_mla_decode_plan | open: layout and metadata formats |
| J | sglang | dcp_kv_index_builders | open: layout and metadata formats |
| J | sglang | decode_num_kv_splits_heuristic | open: layout and metadata formats |
| J | sglang | vattn_asm_mtp_verify_segment_plan | open: layout and metadata formats |
| J | sglang | dsa_cp_interleave_q_split | open: layout and metadata formats |
| J | sglang | spec.align_evict_mask_to_page_size | open: layout and metadata formats |
| J | sglang | spec.draft_decode_kv_indices | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | reduce row scheduler (active-input counts + heavy-first row order) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | bitmatrix metadata (col-sorted / row-sorted nonzero indices) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | ragged tensor metadata (slice/block offsets, packed block schedule) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | ragged tensor metadata remap (drop/renumber slices) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | Hopper MXFP4 value swizzle (MMA tile permutation + fp4 bit interleave) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | Hopper MX scale swizzle | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | Blackwell MX scale swizzle (weights and activations, batched or ragged) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | Blackwell MXFP4 value repack (K-packed <-> N-packed) | open: layout and metadata formats |
| J | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | Blackwell MX4 shuffled weight layout conversion | open: layout and metadata formats |
| M | kernl | experimental llama: rotary embedding (RoPE, interleaved pairs) | v2.4 ROPE role |
| M | kernl | experimental llama: fused rmsnorm + matmul (+ RoPE epilogue), fp16 weights | v2.4 ROPE role |
| M | kernl | experimental llama: fused rmsnorm + QKV projection + RoPE on q,k, fp16 | v2.4 ROPE role |
| M | lightllm | mrope_position_ids | open: mrope positions |
| M | lightllm | neo_mrope_position_ids | open: mrope positions |
| M | sglang | fused_qk_rope_reshape_and_cache | v2.4 ROPE role |
| L | sglang | lora_dense_sgmm | open: LoRA layouts |
| L | sglang | lora_mla_kv_b_absorbed | open: LoRA layouts |
| O | lightllm | mhc_pre_norm | open: other |
| O | sglang | mhc_sinkhorn_coefficients | open: other |
| O | sglang | mhc_boundary_hip | open: other |
| O | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | SwiGLU (GPT-OSS variant, interleaved gate/linear, clamp limit, flexpoint) | open: other |
| O | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | FPSan embed/unembed (gluon) | open: other |
| R | attorch | activation forward with fused dropout (dropout=True) | R1_RANDOM |
| R | attorch | activation backward with fused dropout (dropout=True) | R1_RANDOM |
| R | attorch | dropout forward | R1_RANDOM |
| R | attorch | dropout backward | R1_RANDOM |
| R | flash-attention (Triton code) | fused LayerNorm / RMSNorm with dropout (dropout_p > 0) | R1_RANDOM |
| R | LMDeploy | multinomial_sampling_sorted | R1_RANDOM (D2 open) |
| R | sglang | simulated_balanced_routing | open: other randomness |
| R | sglang | cache_consistency_random_jitter | open: other randomness |
| R | sglang | spec.markov_sample_step | R1_RANDOM (D2 open) |
| R | sglang | spec.simulated_bonus_sample | R1_RANDOM (D2 open) |
| R | sglang | flash_attn_fused_add_norm_fwd | open: other randomness |
| R | trident | dropout forward | R1_RANDOM |
| R | trident | dropout backward | R1_RANDOM |
| R | trident | attention with dropout (dropout_p>0, forward+backward) | R1_RANDOM |
| R | OpenAI triton_kernels (python/triton_kernels in triton-lang/triton) | random block signs (Philox) | open: other randomness |
