# Gemma / Llama 未重复算子扫描

这一步重新检查了两个模型已有的全量 pattern-screen 结果，并优先查看未出现在历史主案例中的算子族。升级规则仍是预先冻结的多重比较门，不因为单个 nominal p 值或 A>1 就把行列为 bias case。

| 模型 | 首轮扫描行数 | 通过升级门的行数 | 结论 |
|---|---:|---:|---|
| Llama-3.2-3B | 64 | 0 | 32 行有非零差异但缺合法长程重放，暂记未决 |
| Gemma-4 E2B | 115 | 0 | 72 行有非零差异但缺合法长程重放，暂记未决 |
| Llama-3.2-3B (text512) | 63 | 0 | 37 行有非零差异但缺合法长程重放，暂记未决 |

## 最强但未升级的算子族

下面只列出排序最靠前的行，作为后续可重放入口；它们不被称为阴性，也不被称为 bias。

### Llama-3.2-3B

| 阶段 | 算子族 | 方向分数 | nominal p |
|---|---|---:|---:|
| FORWARD | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt` | 1.258 | 0.0078 |
| FORWARD | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt` | 1.188 | 0.0078 |
| FORWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view` | 1.239 | 0.0156 |
| FORWARD | `extern_kernels.mm` | 1.178 | 0.0312 |
| FORWARD | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt` | 1.286 | 0.0391 |

### Gemma-4 E2B

| 阶段 | 算子族 | 方向分数 | nominal p |
|---|---|---:|---:|
| BACKWARD | `extern_kernels.mm` | 1.109 | 0.0234 |
| FORWARD | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow` | 1.178 | 0.0391 |
| FORWARD | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow` | 1.169 | 0.0391 |
| FORWARD | `triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_view` | 1.157 | 0.0703 |
| BACKWARD | `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view` | 1.146 | 0.0703 |

### Llama-3.2-3B (text512)

| 阶段 | 算子族 | 方向分数 | nominal p |
|---|---|---:|---:|
| FORWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view` | 1.680 | 0.0078 |
| FORWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view` | 1.373 | 0.0078 |
| FORWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view` | 1.223 | 0.0312 |
| FORWARD | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view` | 1.249 | 0.0391 |
| BACKWARD | `triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view` | 1.117 | 0.1250 |

当前结果：Gemma 115 行、Llama text128 的 64 行和 text512 的 63 行都完成了首轮扫描，但没有新增通过冻结升级门的候选。部分行虽然有非零差异，却没有合法的参数可达 repair/长程重放边界；这些行明确记为未决，不能当作阴性。若后续要扩大分母，应先建立合法 repair、载体和 live replay，而不是把 pattern-screen 直接当作训练 bias 证据。
