# Qwen3 operator equivalence ledger v0.1

## Verdict

CANDIDATE_EQUIVALENCE_PARTITION_COMPLETE_FOR_FROZEN_FORWARD; TRANSPORT_UNVALIDATED; CAUSAL_COVERAGE_INCOMPLETE

The classes below are candidate experimental strata, not proven equivalence classes. They receive no causal-coverage credit until the transport gate passes.

## Semantic-role denominator

| Class | Family | Role | Calls | Original-valid | State | Candidate representatives | Fusion context |
|---|---|---|---:|---:|---|---|---|
| `linear.q_proj` | Linear | attention query projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.k_proj` | Linear | attention key projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.v_proj` | Linear | attention value projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.o_proj` | Linear | attention output projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.gate_proj` | Linear | MLP gate projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.up_proj` | Linear | MLP up projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.down_proj` | Linear | MLP down projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | extern:mm plus role-specific cast/layout consumers |
| `linear.lm_head` | Linear | vocabulary projection | 1 | 1 | VALID_NULL_EFFECT | singleton | extern:mm plus logits conversion/slice |
| `norm.input.layer0` | RMSNorm | layer-0 input normalization | 1 | 0 | BARRIER_CONDITIONED | singleton | embedding fused with input RMSNorm |
| `norm.input.layers1_27` | RMSNorm | decoder input normalization after previous-layer residual | 27 | 0 | BARRIER_CONDITIONED | 1,14,27 | cross-layer previous residual/add fused with next input RMSNorm |
| `norm.post_attention` | RMSNorm | post-attention normalization | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | attention residual fused with RMSNorm and downstream cast |
| `norm.q_norm` | RMSNorm | query normalization | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | q projection, RMSNorm and rotary application fused |
| `norm.k_norm` | RMSNorm | key normalization | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | k projection, RMSNorm, rotary and layout fused |
| `norm.final` | RMSNorm | final hidden-state normalization | 1 | 1 | VALID_NULL_EFFECT | singleton | last residual plus final RMSNorm |
| `attention.qk_bmm` | attention_bmm | query-key score product | 28 | 0 | INVALID_TREATMENT | 0,14,27 | extern:bmm with fused q/k producers and softmax consumer |
| `attention.pv_bmm` | attention_bmm | probability-value product | 28 | 0 | INVALID_TREATMENT | 0,14,27 | extern:bmm with softmax/value producers and layout consumers |
| `attention.softmax` | attention_softmax | masked safe softmax | 28 | 0 | INVALID_TREATMENT | 0,14,27 | mask add and online safe-softmax reduction fused |
| `attention.rotary` | rotary_embedding | paired query/key rotary application | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | q/k RMSNorm, trig, slicing and pointwise application fused |
| `mlp.silu` | MLP_SiLU | gate activation | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | SiLU fused with gate multiplication |
| `mlp.gate_multiply` | MLP_gate_multiply | activated-gate times up projection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | SiLU fused with gate multiplication |
| `residual.attention` | residual_add | attention residual connection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | residual add fused with post-attention RMSNorm |
| `residual.mlp` | residual_add | MLP residual connection | 28 | 0 | BARRIER_CONDITIONED | 0,14,27 | residual add fused across decoder-layer boundary |
| `embedding.token` | token_embedding | token lookup | 1 | 0 | BARRIER_CONDITIONED | singleton | embedding fused with layer-0 input RMSNorm |
| `mask.causal` | causal_mask_construction | causal/padding mask construction | 1 | 0 | INVALID_TREATMENT | singleton | index/comparison/where construction plus per-layer mask consumption |

The denominator is 536 forward invocations in 24 semantic-role/context classes. If all transport assumptions succeed, at least 62 representative invocations are still required; 2 original-candidate-valid singleton interventions and 50 non-transported barrier-conditioned interventions currently count in their separate categories.
Additionally, 3 fixed-boundary SDPA invocations have joint composite evidence. They receive no separate qk-bmm, softmax, or pv-bmm coverage credit.

## Generated-treatment denominator

| Treatment | Kind | Calls | State |
|---|---|---:|---|
| `triton_poi_fused__to_copy_13` | generated_triton_family | 84 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_9_OF_84__0_EFFECT_9_NULL |
| `triton_poi_fused__to_copy_1` | generated_triton_family | 56 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_6_OF_56__0_EFFECT_6_NULL |
| `triton_poi_fused__to_copy_3` | generated_triton_family | 56 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_6_OF_56__0_EFFECT_6_NULL |
| `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__3_EFFECT_0_NULL |
| `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__3_EFFECT_0_NULL |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__3_EFFECT_0_NULL |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mean_mul_pow_rsqrt_sin_transpose_unsqueeze_view_7` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__0_EFFECT_3_NULL |
| `triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_view_10` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__0_EFFECT_3_NULL |
| `triton_poi_fused__to_copy_clone_transpose_view_11` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__0_EFFECT_3_NULL |
| `triton_poi_fused__unsafe_view_mul_silu_14` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__3_EFFECT_0_NULL |
| `triton_poi_fused_clone_6` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__0_EFFECT_3_NULL |
| `triton_red_fused__safe_softmax_add_prepare_softmax_online_view_8` | generated_triton_family | 28 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_28__3_EFFECT_0_NULL |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15` | generated_triton_family | 27 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_3_OF_27__3_EFFECT_0_NULL |
| `triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__1_EFFECT_0_NULL |
| `triton_per_fused__unsafe_view_add_mean_pow_rsqrt_16` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `triton_poi_fused__safe_softmax_9` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `triton_poi_fused__to_copy__unsafe_view_19` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `triton_poi_fused__to_copy__unsafe_view_add_mul_slice_18` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `triton_poi_fused__to_copy_add_arange_bitwise_and_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_5` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `triton_poi_fused__to_copy_t_17` | generated_triton_family | 1 | PARTIAL_ORIGINAL_CANDIDATE_REPAIR_1_OF_1__0_EFFECT_1_NULL |
| `extern:bmm` | external_kernel_family | 56 | PARTIAL_SHARED_PATH_REEXECUTION_6_OF_56__0_EFFECT_6_NULL |
| `extern:mm` | external_kernel_family | 197 | PARTIAL_SHARED_PATH_REEXECUTION_22_OF_197__0_EFFECT_22_NULL |

No generated treatment family is fully covered. A valid module-level null effect may partially constrain several generated kernels, but it cannot identify which constituent kernel or primitive is null.

## Structural and precision operations

Casts, views/layout operations, clones, indexing and mask construction remain in the denominator. They must be covered through a generated treatment family or a valid primitive intervention; they are not silently treated as harmless bookkeeping.
