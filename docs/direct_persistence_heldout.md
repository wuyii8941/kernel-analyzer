# 新实现检查说明

Gemma 4 在运行轨迹前被登记为新实现。它与后来的 atlas-derived pool 分开保存，因为后者缺少完整 repair 和 state 身份。

## 第一个 Gemma 目标

| 项目 | 结果 |
|---|---:|
| 16 步直接作用 A | 0.9860 |
| 16 步随机对照 95% 上界 | 1.0125 |
| 32 步直接作用 A | 1.0003 |
| 32 步实际参数变化 A | 3.2312 |
| 32 步状态反馈 A | 3.2340 |

事前判断是“没有直接持续作用”，32 步结果与之相符。最终参数仍明显分开，但主要由训练状态反馈维持，因此不属于 Direct Persistence Screen 的直接正例。

## v4 的三个新目标

| 目标 | 直接作用 | 实际结果 | 分类 |
|---|---:|---:|---|
| softmax backward / `k_norm` | `A32=0`，没有可见参数作用 | `1.5e-8` | 不适用 |
| GELU/loss backward / projection | `A32=1.0002` | `A32=3.0267` | 状态反馈对照 |
| GELU backward `backward:1860` | `A32=0`，没有可见参数作用 | `A32=0` | 不适用 |

三行增加了未见实现控制项，但没有直接持续正例。因此：

- 不能计算跨未见实现的 recall 或 AUROC；
- 不能把“不适用”改写成负例；
- 不能据此证明整个模型安全；
- 也不需要为了当前稿件继续扩大该池。

紧凑审计：`results/property/direct_persistence_v4/heldout/new_impl_targets_v2.json`。
