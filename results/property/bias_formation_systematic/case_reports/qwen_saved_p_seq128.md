# qwen_saved_p_seq128

## 数学单位

模型：Qwen3-1.7B

F+B：p=softmax(a); da=p*(q-<p,q>) at layer-27 attention

闭合范围：softmax forward, saved/reconstructed P, dS, and actual q/k VJPs（CLOSED）。

## 统一 bias 分解

使用 `E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。本例的物理差异是：backward reconstructs P from BF16 logits plus FP32 max/sum instead of consuming true-forward FP32 P。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`CENTERED / CENTERED / CENTERED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_CONTRACT_MECHANISM`。

原因：the implementation violates a forward/saved/backward representation contract; its effect is trajectory-conditioned even though unrelated-state directions cancel。

干预：replace reconstructed P by the exact true-forward P only at dS, retain BF16 dS ABI。

边界：validated case-specific contract difference; conditional formation stage is unresolved, while symmetric recurrence shows local and feedback accumulation of comparable norm without a stable fixed carrier。

## 轨迹后果

`TRAJECTORY_BIAS`，32 steps，drift norm `0.0044684866443276405` → `0.008657907135784626`。轨迹不提供 formation 标签。

对称四反事实 recurrence 已测：local accumulation L2 `0.004676968354223248`，feedback accumulation L2 `0.004862931525779159`，最大相对闭合残差 `3.3190557446489453e-08`。

## 下一项决定性实验

measure conditional local/gradient/update traces; symmetric recurrence is already closed and shows comparable local and feedback accumulation。

## 证据

- `results/coverage/cases/qwen128_softmax_fb.json`
- `results/coverage/cases/qwen128_softmax_fb_formal.json`
- `results/property/bias_formation/formation/qwen_saved_p_seq128.json`
- `results/coverage/cases/qwen128_softmax_saved_p_trajectory.json`
