# 这次补做的四项检查

> 历史记录：本页的 14 行 Oracle 把三个无状态 SGD 案例与十一个
> AdamW 控制项放在同一张表中，因此不能再作为 Oracle 准确率证据。
> 同一 optimizer 的修正版见 `docs/oracle_repair_v3.md`。本页其余随机
> 对照与 loss 结果仍可单独引用。

本轮补做的目标很简单：不要让 Oracle 只靠一个正例，也不要只在一个参数位置上做随机对照，并补上随机臂的 loss 和 AdamW 的更新映射。

## 1. Oracle 评估集

原来的 12 行里有 11 个控制项和 1 个“局部+反馈混合”的历史阳性。它不是三条 source-persistent 主案例之一，所以保留在审计附录、没有混入新的主评估。

新的主评估包含 11 个原控制项和 3 个事前确定的主案例：

- Liger fused CE；
- Phi lm_head dX；
- Qwen lm_head dX。

因此主评估是 14 行、3 个正例、11 个控制项。16 步筛选分数在这组数据上的 AUROC 是 **1.00**。使用原先冻结的 `A>1` 规则时，召回率 **1.00**，误报率 **2/11 = 18.2%**。这仍然只是已声明载体和已测案例上的 retrospective 结果，不是全算子准确率。

数据：[comparison_v2.json](/data1/tzh/kernel-analyzer/results/property/joint_bias_formation_v1/oracle_baselines/frozen_evaluation_v2/comparison_v2.json)

## 2. 12 个参数载体的随机 null

对 Phi 的 12 个冻结参数载体都做了 32 步、每步注入、RMS 和 support 匹配的随机符号 null，每个载体 5 个随机种子，共 **60 条 null 轨迹**。

最终范数载体的自然算子差异 `A=4.488`；5 个随机种子的 `A` 为 `0.868、0.923、0.929、1.028、1.041`。全 12 个载体的随机 null 均值按种子落在 `0.884–1.029`，最大值为 `1.665`，没有接近自然最终范数载体的 `4.488`。

数据：[random_null_v2/distribution.json](/data1/tzh/kernel-analyzer/results/property/joint_bias_formation_v1/carrier_distribution/random_null_v2/distribution.json)

## 3. 随机臂的 loss

在同一 unseen FP32 loss bank 上，最终范数载体的 seed-101 随机 null 与 repair 的绝对 loss 差为 **8.20e-8**。这是下游结果检查，不用于定义 bias。

数据：[random_null_loss.json](/data1/tzh/kernel-analyzer/results/property/joint_bias_formation_v1/four_scale_arms/random_null_loss.json)

## 4. AdamW 更新映射

同一条 Phi 32-step 共同状态轨迹上，梯度误差的 `A=4.665`；用共享的 AdamW moment 把梯度映射到有效更新后，`A=1.031`。原 SGD 映射的有效更新 `A=4.701`。

这说明更新映射会改变方向性表现；它不能被省略，也不能把 SGD 结果直接当成 AdamW 结果。该实验只补齐响应层，不改变已有形成标签。

数据：[phi_three_stage_adamw.json](/data1/tzh/kernel-analyzer/results/property/joint_bias_formation_v1/phi_three_stage_adamw.json)

## 当前边界

这四项修正了正例数量、载体 null 覆盖和 optimizer 映射，但仍不能把项目表述成“通用全算子 Oracle”。当前最稳妥的说法是：在声明的 matched-repair、单参数载体协议内，短前缀的有效更新方向性能够区分三条已知持久案例与控制项；跨实现类和全参数训练仍需单独确认。
