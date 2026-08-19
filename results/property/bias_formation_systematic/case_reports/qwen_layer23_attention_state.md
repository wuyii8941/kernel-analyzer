# qwen_layer23_attention_state

## 数学单位

模型：Qwen3-1.7B

F+B：S_bwd=alpha*J_softmax(P)^T(DV^T); Gq=S_bwd*K; dWq=Gq^T H

闭合范围：layer-23 q_proj attention-state semantic region and exact tile carrier（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`CONSISTENT_NOT_MARGINAL_PRESERVING`）。S_bwd carries the direction through Gq=S_bwd*K, but its repair removes rather than antithetically pairs the residual。

本例的物理差异是：attention-backward state S_bwd is causal; upstream contributors overlap and include delayed key materialization。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_SEMANTIC_REGION_TRANSPORT_CONTRACT_MECHANISM`。

原因：the changed attention state is transported through Gq=S_bwd*K into a fixed q_proj tile; S_bwd restoration removes that carrier。

干预：restore S_bwd at bmm_76; K-only repair is insufficient; joint S/K repair closes the direction。

边界：validated semantic-region mechanism, not a uniquely identified kernel instruction。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.00032629986526444554` → `0.0006082479958422482`。轨迹不提供 formation 标签。

## 下一项决定性实验

capture conditional layer traces if a first-bias-stage claim is required; do not force single-kernel attribution。

## 证据

- `results/coverage/cases/l23_qproj_attention_state_region.json`
- `results/final/l23_attention_live_weight.json`
- `results/property/bias_formation_final/qwen_l23_attention_mechanism.json`
