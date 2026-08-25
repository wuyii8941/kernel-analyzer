# 所有偏差候选的 4096 步复核摘要

审计表包含 28 行：23 个主矩阵案例、11 个历史候选，以及新增模型的复核行。
最终持久性 bias 标签要求 bias-bearing 的直接作用或反馈作用本身在长程仍有方向；只有 loss 分叉但没有持久 bias 组件的记录保留为后果对照，不计入持久性 bias。两条轨迹不需要收敛到不同的最终 loss。

当前已确认 **6 个持久性 bias 案例**（5 个直接方向、1 个反馈维持），另有 **3 个只有长程 loss 分叉的后果记录**。后者说明训练结果受到影响，但不能据此把没有持久 bias 组件的案例改称为持久性 bias。

| 模型 | 算子或位置 | 长程状态 | 结果 |
|---|---|---|---|
| Qwen3-1.7B | fused CE dW accumulation | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Phi-4-mini | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | seq64 v_proj MM + output rounding | COMPLETE_4096 | 后果对照：loss 分叉但没有持久 bias 组件 |
| Qwen3-1.7B | v_proj MM/output rounding | COMPLETE_4096 | 后果对照：loss 分叉但没有持久 bias 组件 |
| Mamba-130M | in_proj matrix multiply | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | COMPLETE_4096 | 后果对照：loss 分叉但没有持久 bias 组件 |
| Qwen3-VL-Reranker-2B | SiLU backward | COMPLETE_4096 | 反馈长程维持 + loss 分叉 |
| Qwen3-1.7B | attention S_bwd/K to q_proj | ABSTAIN | 无法安全重放 |
| DeepSeek-R1-Qwen3-8B | attention dV BMM | NOT_RUN | UNRESOLVED_FORMATION |
| Llama-3.2-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Ministral-3-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Gemma-4 E2B | RMSNorm / projection feedback region | INVALID_OR_INCOMPLETE | UNRESOLVED |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0057; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0103; input LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0153; attention k-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0190; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| DeepSeek-R1-Qwen3-8B | DeepSeek backward cell 0191; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Mamba-130M | Mamba backward cell 0450; dt-projection bias carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0501; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0508; post-attention LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Phi-4-mini | Phi backward cell 0543; final norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | Qwen backward cell 0654; input LayerNorm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | Qwen backward cell 0745; attention q-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| Qwen3-1.7B | Qwen backward cell 0747; attention k-norm carrier | UNRESOLVED_LONG_REPLAY_PENDING | UNRESOLVED_LONG_REPLAY_PENDING |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | FEEDBACK_CONTROL_NO_DIRECT_GATE |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |

32 步只叫短程方向性；4096 步是长程复核，不等于完整全参数训练收敛。
