# qwen128_vproj_mm

## 数学单位

模型：Qwen3-1.7B

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj

闭合范围：one exact forward MM with actual AOT backward edges（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`MATCHED_CONDITIONAL_SOURCE_SUPPORT`）。repair local residual is centered in 16/16 fixed conditions, while candidate-minus-repair local, gradient, SGD-update, and zero-moment AdamW-update effects are biased in 16/16。

本例的物理差异是：global local decomposition identifies deterministic FP32-to-BF16 output rounding, not MM kernel arithmetic。

条件化 formation（local / gradient / update）：`BIASED / BIASED / BIASED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM`。

原因：at each of 16 fixed states, deterministic nearest rounding has a nonzero candidate-minus-debiased-ensemble mean that remains directional after the actual backward and both declared optimizer mappings。

干预：replace deterministic nearest BF16 output rounding by coordinate-wise unbiased BF16 materialization while retaining the noncoherent kernel residual。

边界：this closes conditional source formation relative to the stochastic source-debiased ensemble; absolute downstream repair bias remains unidentified without an exact downstream reference, and the historical trajectory used a different repair contrast。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.004516110755503178` → `0.010251143015921116`。轨迹不提供 formation 标签。

## 下一项决定性实验

add an exact downstream reference only if claiming the repaired F+B/update itself is absolutely unbiased; use a new ROUNDING_ONLY trajectory for persistence。

## 证据

- `results/coverage/cases/qwen128_vproj.json`
- `results/coverage/cases/qwen128_vproj_precision_decomposition.json`
- `results/coverage/cases/qwen128_vproj_source_aligned_repair.json.gz`
- `results/coverage/cases/qwen128_vproj_conditional_debias.json.gz`
- `results/property/conditional_debias/qwen128_vproj.json`
- `results/coverage/cases/qwen128_vproj_repair_pilot.json`
- `results/coverage/cases/qwen128_vproj_trajectory.json`
