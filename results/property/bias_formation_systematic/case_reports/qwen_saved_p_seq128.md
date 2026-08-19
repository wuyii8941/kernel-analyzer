# qwen_saved_p_seq128

## 数学单位

模型：Qwen3-1.7B

F+B：p=softmax(a); da=p*(q-<p,q>) at layer-27 attention

闭合范围：softmax forward, saved/reconstructed P, dS, and actual q/k VJPs（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`RESPONSE_RECTIFICATION`（`MATCHED_SUPPORT`）。equal and opposite gradient residuals at identical weights and Adam moments produce a nonzero response-even update component。

本例的物理差异是：backward reconstructs P from BF16 logits plus FP32 max/sum instead of consuming true-forward FP32 P。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`CENTERED / CENTERED / CENTERED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_CONTRACT_MECHANISM`。

原因：the implementation violates a forward/saved/backward representation contract, and Adam maps an exact +delta_g/-delta_g pair to non-antithetic updates。

干预：replace reconstructed P by the exact true-forward P only at dS, retain BF16 dS ABI。

边界：the head-specific transport-pairing hypothesis is rejected; the contract source and optimizer response are supported, while unrelated-state global centering remains compatible with trajectory-conditioned bias。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.0044684866443276405` → `0.008657907135784626`。轨迹不提供 formation 标签。

对称四反事实 recurrence 已测：local accumulation L2 `0.004676968354223248`，feedback accumulation L2 `0.004862931525779159`，最大相对闭合残差 `3.3190557446489453e-08`。

## 下一项决定性实验

derive a coordinate/state susceptibility predictor for the measured Adam even response。

## 证据

- `results/coverage/cases/qwen128_softmax_fb.json`
- `results/coverage/cases/qwen128_softmax_fb_formal.json`
- `results/property/bias_formation/formation/qwen_saved_p_seq128.json`
- `results/coverage/cases/qwen128_softmax_saved_p_trajectory.json`
- `results/property/bias_property_search/saved_p_pairing_work_v2.json`
