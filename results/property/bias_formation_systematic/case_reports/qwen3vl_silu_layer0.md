# qwen3vl_silu_layer0

## 数学单位

模型：Qwen3-VL-2B

F+B：y=x*sigmoid(x); dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))

闭合范围：same forward and one exact layer-0 SiLU backward invocation（CLOSED）。

## 统一 bias 分解

使用 `E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。本例的物理差异是：AOT graph-dtype elementary backward arithmetic differs from native aten.silu_backward arithmetic。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`CAUSAL_IMPLEMENTATION_DIFFERENCE_FORMATION_UNRESOLVED`。

原因：the backward implementation causes real update differences, but no sign-symmetric epsilon intervention or conditional formation trace identifies rectification。

干预：swap only the target backward between decomposed and native implementations; forward is identical。

边界：complete causal F+B difference and trajectory; P3/P4 formation mechanism unresolved。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.0020657628774642944` → `0.08343788981437683`。轨迹不提供 formation 标签。

## 下一项决定性实验

capture repeated within-condition traces and run a norm/support-matched +/-epsilon nonlinear control。

## 证据

- `results/round2/vl_silu_cause.json`
- `results/round2/vl_silu_cause_fp32.json`
- `results/coverage/cases/qwen3vl_layer0_silu_trajectory.json`
