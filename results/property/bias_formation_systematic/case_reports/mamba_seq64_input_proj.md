# mamba_seq64_input_proj

## 数学单位

模型：Mamba-130M

F+B：Y=XW^T; dX=QW; dW=Q^T X at layer-0 in_proj

闭合范围：one exact recurrent input-projection MM and actual VJP edges（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`MATCHED_LOCAL_SOURCE_MIXED_F_B_ADAM_DIRECTIONAL`）。repair local residual is centered 16/16 and candidate local plus zero-moment AdamW effects are biased 16/16; gradient/SGD are biased 13/16 and unresolved 3/16, so the complete conditional chain fails closed rather than being promoted。

本例的物理差异是：both same-operand MM kernel arithmetic and deterministic output rounding are directional。

条件化 formation（local / gradient / update）：`BIASED / UNRESOLVED / UNRESOLVED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`PARTIAL_SOURCE_MECHANISM`。

原因：the joint repair centers the declared local source in all 16 fixed conditions, while the natural local effect and zero-moment AdamW effect are biased in all 16; the actual backward/SGD effect is biased in 13 conditions and unresolved in three。

干预：kernel-only, rounding-only, and joint factorial arms; joint uses FP32 MM plus coordinate-wise unbiased BF16 materialization。

边界：cross-architecture conditional local-source and bounded optimizer evidence; the all-layer mechanism gate remains unresolved because three real-backward conditions do not obtain a directional verdict, and the historical trajectory closes only the KERNEL_ONLY arm。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.004031004849821329` → `0.008289474993944168`。轨迹不提供 formation 标签。

## 下一项决定性实验

only if this partial case is revisited, use an exact gradient antithetic control in the three unresolved conditions; do not add repeats or relax the gate。

## 证据

- `results/coverage/cases/mamba_seq64_input_proj.json`
- `results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json`
- `results/coverage/cases/mamba_seq64_input_proj_source_aligned_repair.json.gz`
- `results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz`
- `results/property/conditional_debias/mamba_seq64_input_proj.json`
- `results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json`
- `results/coverage/cases/mamba_seq64_input_proj_trajectory.json`
