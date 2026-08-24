# Phi：同一 AdamW 设置下的随机舍入实验

这次实验补上了此前最重要的协议缺口：旧的随机舍入实验使用无状态 SGD，不能解释 Phi 在 cold-start AdamW 下的 `A=1.0296`。新实验使用完全相同的 32 个状态、参数位置、AdamW 超参数和自然轨迹。

## 结果

| 实验臂 | A | 自身随机抵消 95% 上界 | 结论 |
|---|---:|---:|---|
| deterministic BF16 | 1.02959 | 1.00376 | 存在直接持续作用 |
| no-op sham | 1.02959 | 1.00375 | 与原实现完全相同 |
| stochastic rounding seed 0 | 1.00045 | 1.00167 | 未检测到持续作用 |
| stochastic rounding seed 1 | 1.00004 | 1.00147 | 未检测到持续作用 |
| stochastic rounding seed 2 | 1.00005 | 1.00220 | 未检测到持续作用 |
| stochastic rounding seed 3 | 1.00182 | 1.00371 | 未检测到持续作用 |

前面三个 stochastic-rounding 实验臂的逐步误差总能量与 deterministic BF16 接近或略高。因此方向性消失不能简单解释为“误差变小”；改变的是误差是否持续朝同一方向累积。

## AdamW 内部发生了什么

离线分解使用同一批已捕获 gradient 和 AdamW 状态。Phi 的第一部分沿最终方向贡献 `+5.62`，第二部分贡献 `-4.62`，两者相加为 `1`。这说明 AdamW 通过两个较大的相反作用显著压低了最终方向性，而不是把输入差异简单缩小。

Qwen 对照呈相反符号：两部分分别为 `-2.34` 与 `+3.34`，最终 `A=0.9611`。因此 optimizer 的影响依赖当前 gradient 和 moment 状态，不能给所有算子套一个固定结论。

## 可以说什么

在声明的 cold-start AdamW 设置下，Phi 的 deterministic BF16 实现产生了显著但较小的直接持续更新作用；改用真实 stochastic rounding 后，四次重复都回到各自随机抵消范围内。这是在同一目标 optimizer 下闭合的因果 source intervention。

这里测的是同一状态下的直接有效更新，不是完整 candidate/repair 闭环轨迹，也不证明所有 AdamW 状态或所有算子都会得到相同结果。
