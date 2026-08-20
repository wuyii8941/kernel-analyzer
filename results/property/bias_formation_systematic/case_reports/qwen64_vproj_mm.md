# qwen64_vproj_mm

## 数学单位

模型：Qwen3-1.7B

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj

闭合范围：one exact forward MM with actual AOT backward edges（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`MATCHED_CONDITIONAL_SOURCE_SUPPORT`）。an independent 16-repeat confirmation centers the repair local residual in 16/16 fixed conditions, while candidate-minus-repair local, gradient, SGD-update, and zero-moment AdamW-update effects are biased in 16/16。

本例的物理差异是：same-operand MM kernel arithmetic and deterministic output rounding are both directional。

条件化 formation（local / gradient / update）：`BIASED / BIASED / BIASED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM`。

原因：at each of 16 fixed states, the deterministic joint MM-kernel plus output-rounding residual has a nonzero candidate-minus-debiased-ensemble mean that remains directional after the actual backward and both declared optimizer mappings。

干预：joint FP32 MM plus coordinate-wise unbiased BF16 materialization; kernel-only and rounding-only factorial controls。

边界：this closes conditional source formation relative to the stochastic joint-source-debiased ensemble; it does not certify absolute downstream repair bias without an exact downstream reference。

## 轨迹后果

separation：`TRAJECTORY_SEPARATION`；directional persistence：`CONFIRMED`。共 32 steps，drift norm `0.003998710308223963` → `0.010890079662203789`。

formation contrast：`JOINT_KERNEL_PLUS_UNBIASED_ROUNDING`；trajectory contrast：`KERNEL_ONLY_FP32_MM_WITH_BF16_ABI`；alignment：`MISMATCH`；same-contrast full chain：`False`。

参数距离增长只证明 causal separation，不单独证明方向性 persistence，也不提供 formation 标签。

## 下一项决定性实验

use a new JOINT-repair trajectory only if persistence of this exact identified source is required。

## 证据

- `results/coverage/cases/qwen64_vproj.json`
- `results/coverage/cases/qwen64_vproj_precision_decomposition.json`
- `results/coverage/cases/qwen64_vproj_source_aligned_repair.json.gz`
- `results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz`
- `results/property/conditional_debias/qwen64_vproj.json`
- `results/coverage/cases/qwen64_vproj_repair_pilot.json`
- `results/coverage/cases/qwen64_vproj_trajectory.json`
