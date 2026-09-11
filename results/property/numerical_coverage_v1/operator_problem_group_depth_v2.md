# 算子族与重点问题组证据深度

当前清单有 520 个已核验位置、17 个目录家族。位置数不是独立问题数。

## 目录家族

| 家族 | 已归类位置 | 已核验 | 其中明确 Triton | 其他或未声明 | 清单外专项证据 |
|---|---:|---:|---:|---:|---:|
| 矩阵乘法与线性层 | 0 | 0 | 0 | 0 | 0 |
| Normalization | 1302 | 263 | 263 | 0 | 0 |
| Softmax | 120 | 28 | 28 | 0 | 0 |
| Cross-entropy / fused loss | 1 | 1 | 1 | 0 | 0 |
| SiLU 与门控乘法 | 10944 | 64 | 64 | 0 | 0 |
| Softplus | 70 | 36 | 36 | 0 | 0 |
| 状态递推 / scan | 10593 | 8 | 8 | 0 | 0 |
| Rotary / RoPE | 32 | 24 | 24 | 0 | 1 |
| 独立求和与平方和归约 | 36 | 36 | 36 | 0 | 0 |
| 索引梯度累加 | 1 | 1 | 0 | 1 | 0 |
| GELU | 34 | 34 | 34 | 0 | 0 |
| 卷积（含逐通道与视觉卷积） | 24 | 24 | 0 | 24 | 0 |
| Embedding lookup | 1 | 1 | 1 | 0 | 0 |
| Top-k / sort selection | 0 | 0 | 0 | 0 | 1 |
| Optimizer parameter and moment update | 0 | 0 | 0 | 0 | 1 |
| Fused causal attention | 0 | 0 | 0 | 0 | 1 |
| 按通道偏置加法 | 0 | 0 | 0 | 0 | 0 |

## 当前应深入的去重问题组

| 问题组 | 实际实现 | 当前闭合程度 |
|---|---|---|
| ADAMW8BIT_BLOCKWISE_MOMENT_QUANTIZATION | TORCH_COMPILE_GENERATED_TRITON | RECURRENCE_SOURCE_UPDATE_AND_TRAINING_IMPROVEMENT_CONFIRMED; SINGLE_DECLARED_PROTOCOL_ONLY |
| FUSED_ROTARY_POSITION_TRANSFORM | INDUCTOR_GENERATED_TRITON | STRONG_FIXED_SUITE_PHENOMENON; SOURCE_AND_TRAINING_CHAIN_OPEN |
| FUSED_LINEAR_CE_WEIGHT_GRADIENT_ACCUMULATION | MIXED_LIGER_TRAINING_COMPUTATION | SOURCE_IDENTITY_AND_TRAJECTORY_SPLIT; PERSISTENT_DEGRADATION_NOT_SUPPORTED |
| FUSED_CAUSAL_ATTENTION_BACKEND_SUBSTITUTION | PYTORCH_FLASH_CUDA_VS_MATH | STRONG_FIXED_SUITE_COMPARATOR; NOT_A_TRITON_MECHANISM_CHAIN |

该表不从已核验位置推断 bias，也不把 update 差异、数学来源、修改成功和训练后果合并成一个 PASS。模型、层号、shape、checkpoint 和训练阶段变化不会自动增加问题组。
