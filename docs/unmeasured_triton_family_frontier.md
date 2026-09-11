# 未测 Triton 算子族前沿（2026-09-11）

这是由 `scripts/build_unmeasured_triton_family_frontier.py` 生成的家族级规划快照，按
算子族去重，不按模型位置重复计数。规划和执行选择不读取数值好坏。

## 本轮状态

| 家族 | 状态 | 证据或下一步 |
|---|---|---|
| `DATA_MOVEMENT_LAYOUT` | 已完成有效固定集合测量 | 新 runtime release；32 个状态；confirmation update RMS 约 41.53%，对齐系数约 −8.62% |
| `ELEMENTWISE` | 已完成有效固定集合测量 | 新 runtime release；显式记录 graph-break policy；32 个状态；confirmation update RMS 约 39.78%，对齐系数约 −7.91% |
| `FUSED_MIXED` | 阻塞 | 旧尝试的 AOT 端口身份不一致；先修参考绑定，不重复运行 |
| `MASK_POSITION_CONTROL` | 阻塞 | 当前没有可用参考适配器 |
| `ELEMENTWISE_BIAS` | 外部证据 | 没有目录位置，不作为目录测量结果 |

两个有效测量都属于声明的 zero-moment AdamW、单一目标参数、32 个固定状态协议。它们支持
该协议下的 `NON_EQUIVALENT` 固定集合判断；不支持随机状态总体保证、跨 checkpoint 外推、
训练质量结论或唯一根因归因。原始分析文件仍保存在：

```text
results/property/numerical_coverage_v1/unmeasured_family_campaigns_v4/
results/property/numerical_coverage_v1/unmeasured_family_campaigns_v5/
```

## 自动化规则

规划器会：

1. 合并目录清单、已有主线角色和历史 family campaign；
2. 排除已有有效测量或主线证据的家族；
3. 将历史执行失败标为“先修绑定”，而不是负例；
4. 只为静态未测家族生成最多一个代表任务；
5. 在新 runtime release 或显式编译策略改变时记录原 release、替代 release 和协议因素。

因此，后续重新执行必须先产生新的参考/运行绑定证据，不能仅换一个文件名重跑原失败任务。
