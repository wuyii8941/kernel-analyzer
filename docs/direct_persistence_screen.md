# Cold-start AdamW Direct Persistence Screen

这是一套短程分诊流程，不是“安全证明”。它只回答一个问题：

> 在声明好的 AdamW 训练设置下，一个算子和 repair 的有效更新差异，是否值得进入完整的 32 步检查？

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

短筛没有升级，不代表安全。只有完成 32 步确认后，才允许报告“没有检测到直接持续性”。

## 当前 15 行回溯结果

统一 AdamW 后，方向性分数的回溯 AUROC 为 `0.944`，同一批数据上的更新 RMS AUROC 为 `0.528`。这是回溯分诊结果，不是未见实现上的准确率。

把 full-32 分数再按每行自己的 sign-flip null 做归一化后，当前 15 行的
回溯 AUROC 为 `1.0`；这是确认阶段的诊断量，不能替代 16 步短筛，因为旧
v3 没有保存 prefix-16 的 sign-flip Gram。

三个预先声明的案例中：

| 案例 | 32 步 direct A | 结论 |
|---|---:|---|
| Liger fused CE | 1.720 | confirmed |
| Phi `lm_head dX` | 1.029 | confirmed，小幅度 |
| Qwen `lm_head dX` | 0.957 | direct update 抵消 |

结果盲抽样中的 `0543` 名义 `p≈0.026`，但没有通过 12 行 family 的 Holm 校正。因此它保留为 `UNRESOLVED_CANDIDATE`，不能改标为 negative，也不进入 confirmed-only 性能分母。

## Phi 协议边界

Phi 的 `A=3.325→0.956` 是 16 个共同状态、无 moments 的 stateless SGD 源干预；Phi 的 `A=1.029` 是 32 步 cold-start AdamW 结果。两者不是同一个测量，前者不能解释后者。AdamW 下的 matched source intervention 仍未完成。

## 优化器不是统一根因

同状态重放显示，优化器会改变误差进入有效更新的方式，但三例的方向不同：

| 案例 | 梯度差异 A32 | captured AdamW A32 | 每步重置 moments A32 |
|---|---:|---:|---:|
| Liger | 2.838 | 1.681 | 1.828 |
| Phi | 4.683 | 1.030 | 1.014 |
| Qwen | 1.343 | 0.961 | 1.000 |

所以可以说“优化器会抑制或保留方向性”，不能说“优化器就是数值误差的来源”。
这些结果仍限定在 moments 从零开始、之后正常演化的 AdamW 设置。

Gemma 4 的未见实现反馈控制也做了同状态响应对照：梯度差异和无状态 SGD
的 `A32` 都约为 `1.019`，captured AdamW 为 `0.9995`，每步重置 moments
为 `1.0001`。因此它不是 direct-persistence positive；它说明最终轨迹分离
不能替代局部更新检查，反馈和直接作用必须分开报告。

## 第一个未见实现的前瞻检查

Gemma 4 是事前冻结的新实现。它的局部 direct 分数在 16 步为 `0.986`、在
32 步为 `1.0003`，因此没有通过 direct persistence；但 actual trajectory
为 `3.231`，主要由 feedback 维持。这是一个有效的 `NEW_IMPL` 负例，而不是
“整个训练安全”的证明。当前只有这一个未见实现，不能计算 recall 或 AUROC。

详见 [Gemma held-out 记录](direct_persistence_heldout.md)。

## 结果位置

- [v4 protocol](../results/property/direct_persistence_v4/protocol.json)
- [回溯统计](../results/property/direct_persistence_v4/retrospective_metrics.json)
- [多重比较](../results/property/direct_persistence_v4/multiplicity.json)
- [direct/feedback/actual 贡献表](../results/property/direct_persistence_v4/contribution_table.csv)
- [逐项完成审计](../results/property/direct_persistence_v4/completion_audit.json)
