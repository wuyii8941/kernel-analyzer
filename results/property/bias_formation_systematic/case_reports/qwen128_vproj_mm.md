# qwen128_vproj_mm

## 数学单位

模型：Qwen3-1.7B

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj

闭合范围：one exact forward MM with actual AOT backward edges（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`SUPPORTING_SOURCE_OBSERVATION_CONTRAST_MISMATCH`）。output rounding is directional, but the existing trajectory repairs accumulation instead。

本例的物理差异是：global local decomposition identifies deterministic FP32-to-BF16 output rounding, not MM kernel arithmetic。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`UNRESOLVED_CONTRAST_MISMATCH`。

原因：the source decomposition and trajectory manipulate different contrasts; they cannot yet be composed into one causal explanation。

干预：existing trajectory promotes MM accumulation but retains BF16 output rounding。

边界：do not attribute the trajectory to output rounding until an output-rounding repair is run。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.004516110755503178` → `0.010251143015921116`。轨迹不提供 formation 标签。

## 下一项决定性实验

run an exact output-rounding intervention with sham at the same F+B boundary。

## 证据

- `results/coverage/cases/qwen128_vproj.json`
- `results/coverage/cases/qwen128_vproj_precision_decomposition.json`
- `results/coverage/cases/qwen128_vproj_repair_pilot.json`
- `results/coverage/cases/qwen128_vproj_trajectory.json`
