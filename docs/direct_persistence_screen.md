# Cold-start AdamW Direct Persistence Screen

这是一套短程分诊流程，不是“安全证明”。它只回答一个问题：

> 在声明好的 AdamW 训练设置下，一个算子和 repair 的有效更新差异，是否值得进入更昂贵的长程检查？

## 当前固定规则

每个案例使用相同的参数、输入、状态顺序和 AdamW 设置。moments 在开始时为零，之后正常更新。短筛看前 16 步的方向一致性：

```text
A16 = ||sum(update_difference_t)|| / sqrt(sum(||update_difference_t||^2))
```

当前冻结的短筛规则是 `A16 > 1.0`。输出只有：

```text
ESCALATE
NO_ESCALATION_UNDER_SHORT_SCREEN
ABSTAIN
```

短筛没有升级，不代表长期安全。32 步只允许报告“这个短窗口内是否检测到方向”；当前长期标签必须来自 4096 步及其后半程滚动窗口。

在保留了完整每步更新向量的两个回放中，我们另外计算了每一行自己的随机符号对照：
Phi `lm_head dX` 的 16 步 `A=1.013`，随机对照 95% 上界为 `1.003`；Qwen
`lm_head dX` 的 16 步 `A=0.968`，随机对照 95% 上界为 `1.034`。这只是离线校准，
没有用来改写已经冻结的 `A16>1.0` 规则。其余历史行没有保存足够的 prefix 向量，
所以按规定记为 `ABSTAIN`，不会补造校准值。详见
`results/property/direct_persistence_v4/prefix_null_reanalysis.json`。

## 当前 15 行短程回溯结果

统一 AdamW 后，方向性分数的回溯 AUROC 为 `0.944`，同一批数据上的更新 RMS AUROC 为 `0.528`。这是回溯分诊结果，不是未见实现上的准确率。

把 full-32 分数再按每行自己的 sign-flip null 做归一化后，当前 15 行的
回溯 AUROC 为 `1.0`；这是确认阶段的诊断量，不能替代 16 步短筛，因为旧
v3 没有保存 prefix-16 的 sign-flip Gram。

三个预先声明的案例中：

| 案例 | 32 步 direct A | 结论 |
|---|---:|---|
| Liger fused CE | 1.720 | 升级到长程复核 |
| Phi `lm_head dX` | 1.029 | 升级到长程复核 |
| Qwen `lm_head dX` | 0.957 | 此 cold-start 短窗口内抵消；不能作为长期标签 |

结果盲抽样中的 `0543` 名义 `p≈0.026`，但没有通过 12 行 family 的 Holm 校正。因此它保留为 `UNRESOLVED_CANDIDATE`，不能改标为 negative，也不进入 confirmed-only 性能分母。

## 4096 步长程复核

长程协议先 warm up 128 步，再测量 4096 步同状态直接更新，并检查后半程 64 个不重叠的 32 步窗口：

| 案例 | A4096 | 后半程窗口 | 长程结论 |
|---|---:|---:|---|
| Liger fused CE | 14.018 | 64/64 有方向 | 稳健长程直接方向；配对参数距离 9.266，末步 loss gap -0.1305 |
| Phi `lm_head dX` | 46.090 | 64/64 有方向 | 稳健长程直接方向 |
| Qwen `lm_head dX` | 6.488 | 64/64 有方向 | 稳健长程直接方向；结果依赖训练状态 |
| Mamba `in_proj` | 1.110 | 33/64 有方向 | 未形成稳健长程方向 |
| Qwen saved-P | 1.195 | 28/64 有方向 | 未形成稳健长程直接方向 |
| Qwen seq64 `v_proj` | 1.000 | 不超过自身随机基线 | 未形成稳健长程直接方向 |
| Qwen seq128 `v_proj` | 0.981 | 不超过自身随机基线 | 未形成稳健长程直接方向 |
| Qwen3-VL SiLU | 3.100（反馈） | 长程反馈维持 | 反馈维持型长程案例；配对 loss gap 已记录 |
| layer-23 attention | — | — | 实现身份变化，`ABSTAIN` |
| Gemma4 RMSNorm | 未决 | 未决 | 兼容运行包未能写出完整长程结果；不判阴性 |
| DeepSeek layer-35 `dV` | — | — | 形成阶段证据不足，未升级长程；单独保留为未决 |

Qwen 是短筛边界的直接证据：旧 cold-start 32 步为 `A=0.957`，但 warm-state 长程从 `A32=1.084` 增长到 `A4096=6.488`。因此 `NO_ESCALATION_UNDER_SHORT_SCREEN` 只能表示较低优先级，不能表示“不会长期形成方向”。Liger 的长程直接结果是 `A4096=14.018`；SiLU 则不是直接源方向，而是实际反馈分离在 4096 步仍保持，并记录到配对 loss gap。

## Phi 协议边界

旧的 Phi `A=3.325→0.956` 是无状态 SGD 源干预，不能解释 AdamW 结果。现在同一 cold-start AdamW 协议已经闭合：deterministic BF16 为 `A=1.02959`，四次 stochastic-rounding 重复为 `1.00045、1.00004、1.00005、1.00182`，均未超过各自随机抵消上界。详细结果见 `docs/phi_adamw_source_intervention.md`。

## 优化器不是统一根因

同状态重放显示，优化器会改变误差进入有效更新的方式，但三例的方向不同：

| 案例 | 梯度差异 A32 | captured AdamW A32 | 每步重置 moments A32 |
|---|---:|---:|---:|
| Liger | 2.838 | 1.681 | 1.828 |
| Phi | 4.683 | 1.030 | 1.014 |
| Qwen | 1.343 | 0.961 | 1.000 |

所以可以说“优化器会抑制或保留方向性”，不能说“优化器就是数值误差的来源”。
这些结果仍限定在 moments 从零开始、之后正常演化的 32 步 AdamW 设置。Qwen 的 4096 步结果说明，短程 optimizer 结论还会随训练状态改变。

Gemma 4 的未见实现反馈控制也做了同状态响应对照：梯度差异和无状态 SGD
的 `A32` 都约为 `1.019`，captured AdamW 为 `0.9995`，每步重置 moments
为 `1.0001`。因此它不是 direct-persistence positive；它说明最终轨迹分离
不能替代局部更新检查，反馈和直接作用必须分开报告。

## 未见实现的前瞻检查

Gemma 4 是事前冻结的新实现。它的局部 direct 分数在 16 步为 `0.986`、在
32 步为 `1.0003`，因此没有通过 direct persistence；但 actual trajectory
为 `3.231`，主要由 feedback 维持。这是一个有效的 `NEW_IMPL` 负例，而不是
“整个训练安全”的证明。

另外三个 Gemma 4 目标已经在新进程中完成了 16 步形成检查和 32 步后果检查，
且源判断在轨迹开始前冻结：

| 目标 | 形成判断 | 32 步局部作用 | 32 步实际分离 | 解释 |
|---|---|---:|---:|---|
| softmax backward / `k_norm` | 无直接持续性 | `A=0.000`，无可见载体差异 | `1.5e-8` | 没有可测的参数作用，记为不适用 |
| GELU/loss backward / projection | 无直接持续性 | `A=1.0002` | `A=3.027` | 局部作用近似抵消，分离主要由反馈维持 |
| GELU backward / `backward:1860` | 无直接持续性 | `A=0.000`，无可见载体差异 | `A=0.000` | 没有可测的参数作用，记为不适用 |

这三个结果增加了未见实现的控制项，但没有增加新的 direct-persistence 正例。
当前未见实现池仍没有正例，因此不能计算 recall 或 AUROC；它们也不证明整个模型安全。

详见 [Gemma held-out 记录](direct_persistence_heldout.md)。
本轮三个新目标的紧凑结果见
`results/property/direct_persistence_v4/heldout/new_impl_targets_v2.json`。

## 结果位置

- [v4 protocol](../results/property/direct_persistence_v4/protocol.json)
- [回溯统计](../results/property/direct_persistence_v4/retrospective_metrics.json)
- [多重比较](../results/property/direct_persistence_v4/multiplicity.json)
- [direct/feedback/actual 贡献表](../results/property/direct_persistence_v4/contribution_table.csv)
- [可由原始回放重算的误差指标](../results/property/direct_persistence_v4/tolerance_comparison.json)
- [可由原始回放重算的影响量](../results/property/direct_persistence_v4/severity.json)
- [逐项完成审计](../results/property/direct_persistence_v4/completion_audit.json)

长程结果见 `results/property/declared_persistent_4096/summary.md`。身份更完整的 v4.1 清单仍保留为未来扩展；它不覆盖当前 4096 步结论。
