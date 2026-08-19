# mamba_seq64_input_proj

## 数学单位

模型：Mamba-130M

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 in_proj

闭合范围：one exact recurrent input-projection MM and actual VJP edges（CLOSED）。

## 统一 bias 分解

使用 `E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。本例的物理差异是：both same-operand MM kernel arithmetic and deterministic output rounding are directional。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`PARTIAL_SOURCE_MECHANISM`。

原因：at least two additive local source terms are directional; the trajectory closes the kernel-accumulation arm but not the output-rounding arm。

干预：promote only MM accumulation to FP32, then restore the BF16 ABI。

边界：cross-architecture partial source mechanism; total observed error is not single-source。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.004031004849821329` → `0.008289474993944168`。轨迹不提供 formation 标签。

## 下一项决定性实验

run separate kernel-only and output-rounding-only conditional interventions。

## 证据

- `results/coverage/cases/mamba_seq64_input_proj.json`
- `results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json`
- `results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json`
- `results/coverage/cases/mamba_seq64_input_proj_trajectory.json`
