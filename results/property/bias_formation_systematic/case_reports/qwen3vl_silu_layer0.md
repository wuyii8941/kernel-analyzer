# qwen3vl_silu_layer0

## 数学单位

模型：Qwen3-VL-2B

F+B：y=x*sigmoid(x); dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))

闭合范围：same forward and one exact layer-0 SiLU backward invocation（CLOSED）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`RESPONSE_RECTIFICATION`（`MATCHED_INDEPENDENT_REPLICATION`）。an exact antithetic gradient pair produces a nonzero Adam response-even component。

本例的物理差异是：AOT graph-dtype elementary backward arithmetic differs from native aten.silu_backward arithmetic。

条件化 formation（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。

旧跨无关状态结果（local / gradient / update）：`NOT_MEASURED / NOT_MEASURED / NOT_MEASURED`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`SUPPORTED_CASE_SPECIFIC_OPTIMIZER_RESPONSE_MECHANISM`。

原因：the backward implementation supplies delta_g, and Adam maps the exact +delta_g/-delta_g pair to almost orthogonal rather than opposite update resultants。

干预：use the exact natural delta_g and its negation around the native-SiLU repair gradient at identical Adam state。

边界：optimizer response rectification is closed; the arithmetic origin inside the decomposed backward remains case-specific。

## 轨迹后果

separation：`TRAJECTORY_SEPARATION`；directional persistence：`CONFIRMED`。共 32 steps，drift norm `0.0020657628774642944` → `0.0813409760594368`。

formation contrast：`NATIVE_SILU_BACKWARD_PLUS_ANTITHETIC_ADAM_RESPONSE`；trajectory contrast：`NATIVE_SILU_BACKWARD`；alignment：`ALIGNED_BASE_CONTRAST`；same-contrast full chain：`False`。

参数距离增长只证明 causal separation，不单独证明方向性 persistence，也不提供 formation 标签。

有序四反事实 recurrence：verdict `FEEDBACK_SUSTAINED_SEPARATION`；local / feedback / actual coherence amplification = `1.0013570186547331` / `3.967687298174451` / `3.948838652969838`；最大相对闭合残差 `5.671221865053544e-10`。

## 下一项决定性实验

derive a predictor that distinguishes source-persistent from feedback-sustained cases。

## 证据

- `results/round2/vl_silu_cause.json`
- `results/round2/vl_silu_cause_fp32.json`
- `results/coverage/cases/qwen3vl_layer0_silu_trajectory.json`
- `results/coverage/cases/qwen3vl_layer0_silu_persistence_recurrence.json`
- `results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json`
