# qwen64_vproj_mm

## 数学单位

模型：Qwen3-1.7B

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj

闭合范围：one exact forward MM with actual AOT backward edges（CLOSED）。

## 统一 bias 分解

使用 `E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。本例的物理差异是：precision contrast is directional; full split into kernel, output rounding, and inherited operands is incomplete。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`PARTIAL_SOURCE_MECHANISM`。

原因：the isolated accumulation residual changes the real dW path, but its relation to the complete precision residual is not closed。

干预：same-input FP32 MM accumulation followed by the original BF16 ABI。

边界：trajectory-local partial source; no complete P1 attribution。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.003998710308223963` → `0.010890079662203789`。轨迹不提供 formation 标签。

## 下一项决定性实验

complete the three-way local source decomposition, then capture within-condition formation。

## 证据

- `results/coverage/cases/qwen64_vproj.json`
- `results/coverage/cases/qwen64_vproj_repair_pilot.json`
- `results/coverage/cases/qwen64_vproj_trajectory.json`
