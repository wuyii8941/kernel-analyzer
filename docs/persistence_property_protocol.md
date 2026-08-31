# Reduction orbit predictor protocol

本页只定义 reduction / summation / reassociation 类实现的候选预测量。当前统一
主方法见 [`method.md`](method.md)。Orbit mean 不是通用 property，也不负责判断
backward、optimizer 或长程训练后果。

## 1. 定义

对输入 `a`，冻结一组在实数数学上等价的 reduction schedules `pi ~ nu`，并用
高精度或精确累加结果 `y*(a)` 作为数学目标：

\[
m_{\mathrm{orb}}(a;\nu)
=
\mathbb E_{\pi\sim\nu}
\left[\operatorname{fl}_{\pi}(a)-y^\star(a)\right].
\]

`m_orb` 衡量：改变合法 reduction schedule 后，source residual 的平均是否仍为
非零。它可以是完整 tensor/vector，不能只用一个挑选出来的坐标代替。

## 2. 明确边界

- 只用于 reduction、summation 和 reassociation；
- 量化网格、截断、saturation、underflow、saved-state mismatch 和 optimizer
  response 需要各自的 matched contrast；
- `m_orb != 0` 不表示所有 schedules 同号，也不表示不存在更好的 schedule；
- 需要执行多个合法 schedules，因此不是一次 forward 即得的免费静态 Oracle；
- 它只提出 source-stage 预测，最终仍要经过 local、gradient、update 和长程检查。

## 3. 采样与防止自我相关

每个 state 使用一个冻结的非默认 schedule roster：

1. default schedule 不参与 orbit mean；
2. 非默认 schedules 分成互不重叠的 A/B 两半；
3. 使用 cross-half inner products，避免有限 schedule 数量把 orbit variance 当成
   mean energy；
4. permutation 只作用于 exact kernel replay boundary 的 reduction operands；
5. reduction 两侧必须共同置换，保持实数运算不变；
6. tile/chunk geometry 若不在 roster 中，结论只能叫该 geometry 下的 orbit mean。

## 4. Liger 的预注册预测

冻结一组合法 chunk/reduction schedules，并在 untouched states 上执行：

1. 计算 BF16 accumulator 下的 `m_orb`；
2. 事前冻结其预测方向；
3. 检查该方向是否与 local-stage candidate-repair mean effect 对齐；
4. 将 accumulator 改为 FP32；
5. 预测 `m_orb` 与 local-stage mean effect 同时下降，同时保留 total residual
   energy 作为单独报告项。

只有上述方向和干预预测都在 confirmation states 成立时，才能说 orbit mean 对
Liger 这一 reduction family 有预测价值。失败则保留为无效候选预测量，不改换
roster 或方向重试。

## 5. 输出

每条 certificate 至少保存：

```text
case_id
state_ids
schedule_distribution
schedule_digests
default_schedule_digest
mathematical_target
accumulator_dtype
orbit_mean
cross_half_mean_energy
local_stage_mean
direction_alignment
total_residual_energy
confidence_interval
decision
```

合法输出只有：

- `SOURCE_PREDICTOR_CONFIRMED_FOR_DECLARED_ORBIT`；
- `SOURCE_PREDICTOR_NOT_CONFIRMED`；
- `ABSTAIN`。

任何一个结果都不能直接改写成 `SAFE` 或通用全算子结论。
