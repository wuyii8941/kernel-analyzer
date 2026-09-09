# Fused RoPE 与 position scaling 固定集合核验

本页汇总一个真实编译生成 Triton 计算的相同输入比较。数字由
`scripts/build_rotary_family_evidence.py` 从原始分析、源码绑定和 optimizer 条件
摘要中核对生成；不是从本文手工录入。

## 比较对象

- 模型：Ministral-3-3B-Base-2512。
- 实现：Inductor 生成的 fused RoPE / query position scaling Triton 计算。
- 参考：对同一局部输入执行已登记的 FP32 计算，再写回原输出 dtype。
- 目标参数：`model.language_model.layers.6.self_attn.o_proj.weight`。
- 统计范围：32 个固定训练状态，前后各 16 个用于方向描述与确认。
- 主要判断：确认集合实际参数写入的原坐标总差异 RMS；工程范围为 1%。

运行时和源码绑定记录了 `ATTENTION_POSITION_SCALING` 参考类型、Triton 实现身份及
完整函数的语义摘要。高、低位置比较绑定到同一个语义摘要和同一个目标参数。

## 结果

| 条件 | 确认集合参数写入 RMS | 相对正常写入的整体缩放 | 固定集合判断 |
|---|---:|---:|---|
| 高位置，position scaling 生效 | 10.8809% | -0.59197% | 超出 1% 范围 |
| 低位置，scaling 因子为 1 | 8.34657% | -0.34834% | 超出 1% 范围 |

低位置仍有明显差异，因此不能把全部效应唯一归因为 position scaling。当前证据支持
的研究对象是 **fused RoPE / position scaling 的中间数值实现**；position scaling
是增强条件，而不是已经证明的唯一根因。

使用同一组 32 个状态进一步比较 optimizer 条件：

| optimizer 条件 | 参数写入 RMS |
|---|---:|
| cold，moments 为零 | 8.67183% |
| warm，保留自然 moments | 0.453140% |
| 相同 warm 参数，清空 moments | 10.0152% |

warm 与清空 moments 两种设置的 gradient 原坐标统计完全相同，但参数写入差异相差
约 22.1 倍；warm 相对 cold 降至约 5.23%。因此，在这组固定状态和目标参数上，
AdamW moments 会明显抑制该实现差异进入实际参数写入。

## 结论边界

这些结果能够支持：

- 新的真实 Triton 计算可复用同一套发现、参考绑定、三阶段采集和原坐标分析；
- 小 local difference 可以在 gradient 和 cold-start 参数写入中显著放大；
- 实际参数写入效应依赖 optimizer state。

这些结果不能支持：

- position scaling 是唯一根因；
- 随机训练状态总体中的效应大小；
- 全模型或其他参数具有相同差异；
- 已经出现 loss 变化、训练恶化或崩溃；
- optimizer 条件比较是事前冻结的机制预测。该项是结果后的诊断比较。

精简机器证据位于：

- `results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json`
- `results/property/numerical_coverage_v1/ministral_fused_rotary_family_evidence_v1.json`
- `results/property/numerical_coverage_v1/operator_family_report_v10.json`

逐状态原始结果保留在 `/data1/tzh/kernel-analyzer/results/property/numerical_coverage_v1/`
的对应运行目录，不因精简摘要进入 Git 而删除或覆盖。
