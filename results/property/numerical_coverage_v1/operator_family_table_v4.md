# 算子族覆盖表

来源：`results/property/numerical_coverage_v1/operator_family_report_v11.json`。

| family_id | label | classified_positions | valid_measurement_positions | valid_triton_positions | valid_other_or_undeclared_positions | bound_not_measured_positions | reference_only_positions | additional_fixed_suite_evidence | historical_role_records | historical_artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LINEAR | 矩阵乘法与线性层 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| NORMALIZATION | Normalization | 1302 | 263 | 263 | 0 | 1039 | 0 | 0 | 2 | 0 |
| SOFTMAX | Softmax | 120 | 28 | 28 | 0 | 92 | 0 | 0 | 1 | 0 |
| CROSS_ENTROPY | Cross-entropy / fused loss | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| SILU_GATING | SiLU 与门控乘法 | 10944 | 64 | 64 | 0 | 128 | 10752 | 0 | 0 | 0 |
| SOFTPLUS | Softplus | 70 | 36 | 36 | 0 | 34 | 0 | 0 | 0 | 0 |
| RECURRENCE | 状态递推 / scan | 10593 | 8 | 8 | 0 | 10585 | 0 | 0 | 0 | 0 |
| ROTARY | Rotary / RoPE | 32 | 24 | 24 | 0 | 8 | 0 | 1 | 0 | 0 |
| REDUCTION | 独立求和与平方和归约 | 36 | 36 | 36 | 0 | 0 | 0 | 0 | 0 | 0 |
| INDEXED_ACCUMULATION | 索引梯度累加 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GELU | GELU | 34 | 34 | 34 | 0 | 0 | 0 | 0 | 0 | 1 |
| CONVOLUTION | 卷积（含逐通道与视觉卷积） | 24 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 2 |
| EMBEDDING | Embedding lookup | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| SELECTION | Top-k / sort selection | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| OPTIMIZER_UPDATE | Optimizer parameter and moment update | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| FUSED_ATTENTION | Fused causal attention | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| ELEMENTWISE_BIAS | 按通道偏置加法 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`classified_positions` 是 release-qualified 输出位置，不是独立算子数。
`valid_triton_positions` 只按已保存的实际 implementation kind 计数；未知不会被猜成 Triton。
`additional_fixed_suite_evidence` 用于没有编译图位置的常规实现测量，不加入位置数。
有效测量不等于 bias 阳性、总体等价或训练质量结论。
