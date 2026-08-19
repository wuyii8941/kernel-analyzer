# liger_fused_ce

## 数学单位

模型：Qwen3-1.7B + Liger fused linear CE

F+B：Z=HW^T; G=(softmax(Z)-onehot)/N; dH=GW; dW=G^T H

闭合范围：full fused loss F+B region（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`EVENT_PAIRING_ASYMMETRY`（`MATCHED_SUPPORT`）。the BF16 chunk/schedule orbit is 24/24 one-signed, while the same semantic orbit with FP32 accumulation is 13/11 and centered。

本例的物理差异是：64 two-token chunk contributions are added sequentially to a BF16 dW accumulator; chunk geometry changes the rounding schedule。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`UNRESOLVED / UNRESOLVED / UNRESOLVED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM`。

原因：finite-precision sequential accumulation is conditionally asymmetric under the declared chunk schedule, so E[epsilon|chunk geometry] need not vanish。

干预：promote only dW accumulation to FP32 while preserving loss, dH, and all untied gradients。

边界：case-specific source mechanism; no universal P1 property and no M7。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `8.586806677537658e-06` → `0.0022393549008995974`。轨迹不提供 formation 标签。

对称四反事实 recurrence 已测：local accumulation L2 `7.63580291049743e-05`，feedback accumulation L2 `0.0002732251622248212`，最大相对闭合残差 `3.4057641385143195e-08`。

## 下一项决定性实验

variance-matched stratum-mean removal if P1 is to become a general property。

## 证据

- `archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json`
- `archive/nonprecision_v1/runs/liger.fused_ce.certificate.json`
- `archive/nonprecision_v1/runs/liger.fused_ce.chunk.certificate.json`
- `results/trajectory/liger_trajectory.json`
- `results/property/seup_mainline/liger_seup.json`
