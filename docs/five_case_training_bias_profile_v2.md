# 五案例统一训练偏差检查

本页只汇总同一套 Training Bias Profile v2 下的五个开发案例。所有判断规则、案例、
输入顺序、三个测量分支和整体校正组都在结果揭示前冻结。

## 实验条件

- 每例 32 个冻结且不重叠的输入窗口；前 16 个确定方向，后 16 个检查；
- 每个窗口都从同一 pretrained checkpoint 和零 AdamW moments 开始；
- `AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0)`；
- candidate 重复运行在 32/32 个窗口上逐位一致；
- 大参数向量使用三个预先声明的 4096 维 CountSketch，确认结果要求三者方向一致；
- AdamW update 的 15 项为主要检验组，local/gradient 的 30 项为解释检验组，分别
  使用 Holm 校正。

输入窗口不是独立训练 runs，也不是已证明的随机训练总体。以下区间只描述声明的冻结
窗口集合。

## AdamW update 主结果

| 案例 | 通过的 update 结构 | 效应量 | 95% 区间 | Holm p | 结论 |
|---|---|---:|---:|---:|---|
| Liger fused CE | 去掉正常 update 缩放后仍有共同方向 | `0.0151%` | `[0.00614%, 0.0240%]` | `0.0467` | 确认，三份大向量摘要同向 |
| Phi `lm_head dX` | 相对正常 update 的缩小 | `-0.1506%` | `[-0.2046%, -0.0966%]` | `0.00750` | 确认，完整向量 |
| Qwen `lm_head dX` | 无 | — | — | — | 本 cold-start 协议未确认，不等于安全 |
| Qwen `v_proj` | 相对正常 update 的缩小 | `-6.04%` | `[-8.64%, -3.44%]` | `0.00750` | 确认，三份大向量摘要同向 |
| Mamba `in_proj` | 相对正常 update 的缩小 | `-3.12%` | `[-3.37%, -2.86%]` | `0.00750` | 确认，三份大向量摘要同向 |

负号表示 candidate 沿同一状态的正常 repair update 方向少走了一部分，不表示固定参数
坐标中的方向为负。

## 三阶段定位

| 案例 | 算子输出 | Parameter gradient | AdamW update | 最简单解释 |
|---|---|---|---|---|
| Liger | 共同剩余方向 `0.326%` | 共同剩余方向 `0.300%` | 共同剩余方向 `0.0151%` | 方向在低精度累加处已经存在，backward 保留，AdamW 大幅压低但未完全消除 |
| Phi | 共同剩余方向 `0.0375%` | 放大到 `0.372%` | 固定方向未确认，改为缩小正常 update `0.1506%` | backward 增强剩余方向；AdamW 改变了它的表现形式 |
| Qwen `lm_head` | 极小缩放 | 共同剩余方向 `0.00491%` | 无分支确认 | backward 产生共同方向，cold-start AdamW 抵消 |
| Qwen `v_proj` | 无分支确认 | 无分支确认 | 缩小正常 update `6.04%` | 在当前窗口集合中，系统结构首次明确出现在 optimizer update |
| Mamba `in_proj` | 极小缩放 | 无分支确认 | 缩小正常 update `3.12%` | 非 Transformer 案例同样可出现 optimizer 阶段的稳定缩放 |

每一格只写通过本阶段 30 项整体校正的结构。阶段之间使用不同 repair signal 归一化，
所以百分比不能直接当成传播倍率。

## 与 4096 步结果对照

4096 步使用另一种问题和协议：它检查同状态 direct update 在固定参数坐标中是否长期
同向，并记录 paired loss 是否分开。它没有参与 v2 方法或阈值选择。

| 案例 | 本次 32 窗口 cold-start 结果 | 4096 步固定方向结果 | 最大 paired loss gap | 合法结论 |
|---|---|---:|---:|---|
| Liger | 剩余方向确认 | `A=14.018 > null95 1.163`；64/64 后半程窗口有方向 | `0.555` | 短程结构与长程固定方向均有证据 |
| Phi | update 缩小确认 | `A=46.090 > null95 1.565`；64/64 后半程窗口超过自身基线 | `0.0324` | 两种协议都有系统效应，但不能把短程缩放直接当成长程成因 |
| Qwen `lm_head` | cold-start update 未确认 | `A=6.488 > null95 1.031`；64/64 后半程窗口有方向 | `0.00427` | verdict 随 optimizer state、输入和观测长度变化，不能给算子永久标签 |
| Qwen `v_proj` | update 缩小确认 | `A=0.981 < null95 1.424` | `0.134` | 稳定缩小不等于固定方向持久；有 loss 分叉但不能归因于持续 direct direction |
| Mamba `in_proj` | update 缩小确认 | `A=0.935 < null95 1.167` | `0.0471` | 同上 |

这张对照是方法没有围着旧结果调参的重要检查：新方法没有机械复现旧的三正两负，
而是区分了固定方向和随正常 update 旋转的缩放。

## 当前能说什么

可以说：

> 在同一套 cold-start AdamW、同一三阶段测量和整体误报控制下，五个开发案例出现了
> 固定/剩余方向、随正常 update 缩放、被 optimizer 抵消三种不同结果。

不能说：

- 四个 update 确认项都是 4096 步持久固定方向；
- Qwen `lm_head` 在其他 optimizer state 下也一定抵消；
- 这五例代表所有 LLM training 算子的发生比例；
- 检验未确认等于 candidate 安全；
- loss 分叉证明本次确认的某个短程分支是唯一成因。

机器结果：

- `results/property/training_bias_profile_v2/five_case_raw/*.json`
- `results/property/training_bias_profile_v2/five_case_summary.json`
- `results/property/declared_persistent_4096/all_bias_case_audit.json`
