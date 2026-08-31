# 三类实现的统一测量结果

这份结果回答一个具体问题：除了 Liger 和 `lm_head dX`，normalization、softmax
backward 和 attention BMM 能否用同一套方法比较？答案是可以，而且三者表现不同。

> **v2 边界：** 本页是 v1 固定-suite/机制比较。逐状态区间不代表独立 training
> runs，固定 4096 维摘要也尚未做多-seed 复核。它保留为开发证据，不自动升级为
> [`Training Bias Profile v2`](training_bias_profile_v2.md) 的总体确认。

## 相同实验设置

三条记录都使用：

- 一个准确绑定的 candidate implementation 和同边界 repair；
- 32 个不同的自然文本状态；
- 前 16 个状态确定平均方向，后 16 个状态只用于确认；
- local output、parameter gradient、cold-start AdamW update 三层；
- 总差异、相对正常 signal 的缩放、不能由缩放解释的新方向；
- 95% 区间，并把 `3 个案例 × 3 个阶段 × 3 类效应` 放入同一个 Holm 校正组。

大载体使用 v1 固定 4096 维摘要，小载体使用完整坐标。v1 摘要会产生周期性坐标碰撞，
因此这些数值只能作为固定摘要下的观察；v2 已替换哈希并要求额外 seeds 或完整 Gram。

## 最重要的结果

| 模型与位置 | local | gradient | AdamW update | 4096 步后果 | 最简单的解释 |
|---|---|---|---|---|---|
| Gemma-4 E2B，normalization reduction | 有稳定但极小的新方向，Holm `p=0.00675` | 确认状态中为零 | 没有稳定方向 | 旧配对轨迹有 loss 分叉，主要由 feedback 维持 | 局部误差有方向，但没有持续传到声明参数；最终轨迹分离不能归因于每步直接推动 |
| Llama-3.2-3B，softmax backward | 有稳定但极小的新方向，Holm `p=0.00675` | 未校正区间排除零，但 Holm 后不过线 | 区间覆盖零 | 旧配对轨迹实际方向很强，但 local 直接作用不强；有 loss 分叉 | backward 实现差异能到达 gradient，但 AdamW 下未确认直接持续；长程主要是 feedback |
| Llama-3.2-3B，attention BMM | 固定方向不稳定；相对 repair signal 的缩放稳定，Holm `p=0.00675` | 没有稳定平均方向 | 相对缩放区间不跨零，但 Holm 后未确认 | 旧配对轨迹有很强 direct 与 feedback，并有 loss 分叉 | 这不是固定低秩方向，而是随训练状态变化的相对缩放型效应 |

这里的 `local` 对 softmax 指被测 backward kernel 的输出，不是 forward softmax
概率。三条 4096 步结果来自此前冻结的 paired trajectory；它们与新的 32-state
profile 使用相同 exact implementation family，但 state bank 不完全相同，所以只做
后果对齐，不把长程标签倒灌进短程统计检验。

## 关键数字

### Gemma normalization

- local held-out effect / repair RMS：`6.01e-8`，95% 区间
  `[6.01e-8, 6.01e-8]`；
- gradient confirmation effect：`0`；
- AdamW update 区间：`[-5.82e-5, 0]`，不支持稳定直接方向；
- 4096 步旧轨迹：direct `A=1.078`，feedback `A=4.514`；loss gap 在
  `4092/4096` 步非零。

### Llama softmax backward

- local held-out effect / repair RMS：`1.20e-7`，Holm `p=0.00675`；
- gradient held-out effect：`1.90e-4`，95% 区间 `[4.95e-5, 3.55e-4]`，但
  在完整 27 项 Holm 组中 `p=0.441`；
- AdamW update 区间：`[-4.06e-5, 3.11e-5]`；
- 4096 步旧轨迹：local `A=1.019`，actual/feedback `A=4.710`；loss gap 在
  `4093/4096` 步非零。

### Llama attention BMM

- local 固定方向区间跨零；
- local 相对缩放为 `-5.22e-7`，95% 区间
  `[-7.48e-7, -3.20e-7]`，Holm `p=0.00675`；
- gradient 固定方向与相对缩放均未确认；
- AdamW 相对缩放区间 `[-3.73e-4, -8.04e-5]`，但完整 Holm 后
  `p=0.223`，保留为候选而非确认；
- 4096 步旧轨迹：direct `A=5.927`，feedback `A=34.341`；loss gap 在
  `4096/4096` 步非零。

## 对主线的影响

这三条补测否定了“所有 bias case 都是固定低秩方向”的简化说法：

1. Gemma 展示 local 有方向、但到 gradient 消失；
2. softmax 展示 backward output 的方向可到达 gradient、但 AdamW 后未确认；
3. BMM 展示固定方向可以不稳定，但相对正常 signal 的缩放可以稳定。

因此主线仍应同时报告“新的平均方向”和“沿正常 signal 的缩放”。4096 步 loss
分叉说明这些实现值得继续检查，但不能单凭 loss 分叉把每条记录称为持续的直接
bias。

机器结果见：

- `results/property/three_mechanism_profiles_v1/summary.json`
- `results/property/three_mechanism_profiles_v1/consequence_alignment.json`
