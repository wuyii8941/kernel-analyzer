# phi4_seq64_lmhead_dx

## 数学单位

模型：Phi-4-mini

F+B：Y=XW^T; dX=QW; dW=Q^T X at lm_head input VJP

闭合范围：one exact backward MM invocation and both VJP edges（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`MATCHED_SUPPORT`）。the natural residual/transport joint pairing is directional, while a residual-marginal-preserving row permutation restores centering。

本例的物理差异是：same-BF16-operand MM kernel arithmetic; final output rounding is noncoherent。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`CENTERED / BIASED / BIASED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_TRANSPORT_MECHANISM`。

原因：the local mean is small, but the real backward pairing makes Cov(T,epsilon|c) nonzero。

干预：permute residual/row transport pairing while preserving the local residual multiset and norm。

边界：empirical composite transport mechanism; analytic transport reconstruction remains incomplete。

## 轨迹后果

separation：`TRAJECTORY_SEPARATION`；directional persistence：`CONFIRMED`。共 32 steps，drift norm `2.127923607986304e-06` → `9.185909584630281e-05`。

formation contrast：`ANALYTIC_DX_MM_REPAIR`；trajectory contrast：`ANALYTIC_DX_MM_REPAIR`；alignment：`ALIGNED`；same-contrast full chain：`True`。

参数距离增长只证明 causal separation，不单独证明方向性 persistence，也不提供 formation 标签。

对称四反事实 recurrence 已测：local accumulation L2 `4.751236706890506e-05`，feedback accumulation L2 `2.3945317827997007e-06`，最大相对闭合残差 `7.037889645672179e-09`。

## 下一项决定性实验

close the remaining analytic VJP factors before naming one physical transport factor。

## 证据

- `results/coverage/cases/phi4_seq64_lmhead_dx.json`
- `results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json`
- `results/property/bias_formation/interventions/phi4_mm_transport_pairing.json`
- `results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json`
