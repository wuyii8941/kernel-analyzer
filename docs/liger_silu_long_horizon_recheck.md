# Liger 与 SiLU：历史 4096 步复核

本页保留该轮直接/反馈和 loss 数据，不是当前所有实验的总表。
核心案例按[数学成因与 loss 连接](case_evidence_map.md)整理；
“后期仍有直接 bias”是另外一项更强结论。

## Liger fused CE：历史 Qwen 设置

- 直接作用：4096 步 A=14.018；后半程 64 个 32 步窗口均有方向；
  自身随机基线 95% 上界 1.163，单侧 p=0.000999。
- 独立配对后果记录：参数距离 9.2663；第 4096 步 loss 差（被测 − 参考）
  −0.13049；后 512 步平均差 +0.000673。
- 原标签：`PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT`。

直接重放和配对训练分别给证据，不把 loss 差的正负解释成直接偏差的符号。
新小型 GPT-2 全参数训练到 10000 步是[另一组设置](liger_single_boundary_collapse_experiment.md)，
不能用其 loss 数字替换本页，更不能把本页 A 值搬到新实验。

## Qwen3-VL SiLU backward

- 直接作用：4096 步 A=1.017，不能写成持续的局部直接方向。
- 反馈：A=3.100，最终实际分离 0.8884；反馈与最终分离 cosine 0.999997，
  局部方向 cosine 0.00552。
- loss 差：第 4096 步 −7.95e-9；后 512 步平均 +4.93e-8，标准差 1.29e-7。
- 原标签：`FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT`。

这是很小的 loss 非同一与反馈维持证据，不是已证明稳定质量损害。
SiLU 正负响应的非镜像证据另有实验，不能仅凭反馈方向推出自然源 bias 的完整成因。

## 如何使用旧标签

bias 成因已建立且对应配对 loss 分叉的材料应保留，即使后期直接作用不持续。
若只有 loss 分叉、没有独立 bias 证据，则仍是后果对照。
本页旧“持久”标签只说明该测量长度内的规则；不是任意更长时间的定理，
也不要求所有核心案例都通过同一个固定方向门槛。未决不改成阴性。

原始证据：

- [Liger 直接测量](../results/property/declared_persistent_4096/liger_fused_ce.json)
- [Liger 配对 loss](../results/property/paired_loss_4096/liger_fused_ce.json)
- [SiLU 直接/反馈/loss](../results/property/declared_persistent_4096/qwen3vl_silu_4096_with_loss.json)
- [历史全表](../results/property/declared_persistent_4096/all_bias_case_audit.json)
