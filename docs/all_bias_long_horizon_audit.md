# 所有历史偏差候选的长程复核

仓库中有 **23 个唯一主矩阵 case ID**；本审计逐行复核 **26 个 extended candidate rows**（其中包含历史候选、Llama/Ministral 同族复现、Gemma4、12 个结果盲抽的长程 consequence 候选，以及单独追踪的 Gemma GELU consequence 候选）。合并后表共有 **34 行**。这些数字分别表示覆盖分母、候选分母和逐行审计行数，不能混用。

按当前口径，最终计入 **6 个持久性 bias 案例**：其中直接长程方向案例 **5 个**，反馈维持型案例 **1 个**。另有 **3 个**虽没有持久 bias 组件、但确实出现长程配对 loss 分叉；因此当前共有 **9 个训练结果相关记录**，但不能把后果对照改称为持久性 bias。未决/不可安全重放共 **22 个**，不作阴性判断。

Gemma/Llama 的追加首轮扫描另有 **242 行**，其中冻结升级门通过 **0 行**；本轮按冻结规则选出 **6 个**新的 exact F+B 目标，目前完成长程重放 **0 个**。没有完成合法重放的行不计入 bias 案例数，也不改判为阴性。

持久性 bias 的必要条件是 bias 本身在 4096 步仍然存在：直接源方向或反馈有效更新方向至少有一个通过长程检验。配对训练中的参数或 loss 分叉是后果证据，不足以单独把一个没有持久 bias 组件的记录升级为案例。这里不要求两条训练轨迹收敛到不同的最终 loss，也不作这种声称。

| 模型 | 算子或位置 | 形成路径 | 4096 步直接结果 | 参数/loss 分叉 | 最终分类 |
|---|---|---|---|---|---|
| Qwen3-1.7B | fused CE dW accumulation | event/pairing imbalance | A4096=14.018，后半程 64/64 | 是；参数距离 9.27，末步 loss gap -0.13 | 最终持久性 bias 案例 |
| Phi-4-mini | lm_head backward dX | event/pairing imbalance + backward transport | A4096=46.090，后半程 64/64 | 是；参数距离 0.0191，末步 loss gap +0.00158 | 最终持久性 bias 案例 |
| Qwen3-1.7B | lm_head backward dX | event/pairing imbalance + backward transport | A4096=6.488，后半程 64/64 | 是；参数距离 0.00163，末步 loss gap +1.21e-05 | 最终持久性 bias 案例 |
| Qwen3-1.7B | seq64 v_proj MM + output rounding | conditional source asymmetry | A4096=0.993，p=0.511 | 是；参数距离 0.22，末步 loss gap -0.0048 | 后果对照：有长程 loss 分叉，但没有持久 bias 组件 |
| Qwen3-1.7B | v_proj MM/output rounding | local arithmetic/pairing | A4096=0.981，p=0.509 | 是；参数距离 0.302，末步 loss gap +0.000371 | 后果对照：有长程 loss 分叉，但没有持久 bias 组件 |
| Mamba-130M | in_proj matrix multiply | local arithmetic/pairing | A4096=0.935，p=0.405 | 配对长程阶段未能安全完成，未决 | 长程未保持 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | response/state-contract imbalance | A4096=1.195，p=0.084 | 是；参数距离 0.123，末步 loss gap -0.00228 | 后果对照：有长程 loss 分叉，但没有持久 bias 组件 |
| Qwen3-VL-Reranker-2B | SiLU backward | response asymmetry | COMPLETE_4096 | 是；参数距离 0.888，末步 loss gap -7.95e-09 | 反馈维持型 bias，且有 loss 分叉 |
| Qwen3-1.7B | attention S_bwd/K to q_proj | event/pairing imbalance | ABSTAIN | 未测 | 不可安全重放 |
| DeepSeek-R1-Qwen3-8B | attention dV BMM | formation unresolved; event/pairing candidate | NOT_RUN | 未测 | 形成阶段未确认，不升级长程 |
| Llama-3.2-3B | lm_head backward dX | event/pairing family replication | A4096=5.881，超过自身随机基线（窗口统计未导出） | 是；参数距离 0.000376，末步 loss gap +4.24e-05 | 最终持久性 bias 案例 |
| Ministral-3-3B | lm_head backward dX | event/pairing family replication | A4096=5.050，超过自身随机基线（窗口统计未导出） | 是；参数距离 0.00042，末步 loss gap +0 | 最终持久性 bias 案例 |
| Gemma-4 E2B | RMSNorm / projection feedback region | response asymmetry / feedback candidate | UNRESOLVED_LONG_REPLAY_PENDING | 未测 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0057; post-attention LayerNorm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0103; input LayerNorm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0153; attention k-norm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0190; attention q-norm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0191; attention q-norm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_RESOURCE | 配对长程阶段未能安全完成，未决 | 长程运行环境不再可重放，未决 |
| Mamba-130M | Mamba backward cell 0450; dt-projection bias carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Phi-4-mini | Phi backward cell 0501; post-attention LayerNorm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Phi-4-mini | Phi backward cell 0508; post-attention LayerNorm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Phi-4-mini | Phi backward cell 0543; final norm carrier | small mixed candidate | UNRESOLVED_LONG_REPLAY_RESOURCE | 配对长程阶段未能安全完成，未决 | 长程运行环境不再可重放，未决 |
| Qwen3-1.7B | Qwen backward cell 0654; input LayerNorm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Qwen3-1.7B | Qwen backward cell 0745; attention q-norm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Qwen3-1.7B | Qwen backward cell 0747; attention k-norm carrier | feedback-sustained candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Gemma-4 E2B | GELU/loss backward region 1401; projection carrier | response asymmetry / feedback candidate | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Gemma-4 E2B | backward:671 triton_red_fused__to_copy_mul_pow_sum_view_1 [out_ptr1] | new operator scan; exact generated target replay | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Gemma-4 E2B | forward:17 triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_10 [out_ptr0] | new operator scan; exact generated target replay | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Gemma-4 E2B | forward:22 triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_12 [out_ptr0] | new operator scan; exact generated target replay | UNRESOLVED_LONG_REPLAY_PENDING | 配对长程阶段未能安全完成，未决 | 已通过短程 consequence 筛查，4096 步仍未完成 |
| Llama-3.2-3B | forward:20 triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11 [in_out_ptr0] | new operator scan; exact generated target replay | UNRESOLVED_PARAMETER_BINDING | 配对长程阶段未能安全完成，未决 | 长程运行环境不再可重放，未决 |
| Llama-3.2-3B | forward:449 triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13 [out_ptr0] | new operator scan; exact generated target replay | UNRESOLVED_PARAMETER_BINDING | 配对长程阶段未能安全完成，未决 | 长程运行环境不再可重放，未决 |
| Llama-3.2-3B | forward:32 triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12 [in_out_ptr0] | new operator scan; exact generated target replay | UNRESOLVED_PARAMETER_BINDING | 配对长程阶段未能安全完成，未决 | 长程运行环境不再可重放，未决 |
| google/gemma-4-E2B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 没有可达载体，不适用 |
| google/gemma-4-E2B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 没有可达载体，不适用 |

## 口径

- 32 步只能说明短程方向性，不能单独称为持久性 bias。
- 4096 步是同一训练状态下的直接更新审计，不等于完整全参数训练收敛。
- 配对 loss gap 是功能后果信号；这里不要求、也不声称两条轨迹收敛到不同的最终 loss。
- 反馈造成的轨迹分离不能替代直接 bias 证据。
