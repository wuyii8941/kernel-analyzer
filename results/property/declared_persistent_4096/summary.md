# 所有偏差候选的 4096 步复核摘要

审计表包含 28 行：23 个主矩阵案例、11 个历史候选，以及新增模型的复核行。
最终标签允许两条路径：直接更新在长程仍有方向，或虽然直接作用不持续但在某个测量窗口已经出现配对 loss 分叉。两条轨迹不需要收敛到不同的最终 loss。

| 模型 | 算子或位置 | 长程状态 | 结果 |
|---|---|---|---|
| Qwen3-1.7B | fused CE dW accumulation | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Phi-4-mini | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | lm_head backward dX | COMPLETE_4096 | 直接长程方向 + loss 分叉 |
| Qwen3-1.7B | seq64 v_proj MM + output rounding | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Qwen3-1.7B | v_proj MM/output rounding | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Mamba-130M | in_proj matrix multiply | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | COMPLETE_4096 | 未发现稳健长程直接方向 |
| Qwen3-VL-Reranker-2B | SiLU backward | COMPLETE_4096 | 反馈长程维持 + loss 分叉 |
| Qwen3-1.7B | attention S_bwd/K to q_proj | ABSTAIN | 无法安全重放 |
| DeepSeek-R1-Qwen3-8B | attention dV BMM | NOT_RUN | UNRESOLVED_FORMATION |
| Llama-3.2-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Ministral-3-3B | lm_head backward dX | COMPLETE_LONG_HORIZON | 直接长程方向 + loss 分叉 |
| Gemma-4 E2B | RMSNorm / projection feedback region | UNRESOLVED_LONG_REPLAY_RESOURCE | 运行环境或资源未决 |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | FEEDBACK_CONTROL_NO_DIRECT_GATE |
| google/gemma-4-E2B | case-stage matrix row | NOT_ESCALATED | NOT_APPLICABLE_NO_CARRIER_EFFECT |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/state-spaces/mamba-130m-hf | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | NOT_ESCALATED | UNRESOLVED_FORMATION |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | NOT_ESCALATED | SCREENED_NO_CONFIRMED_DIRECT_BIAS |

32 步只叫短程方向性；4096 步是长程复核，不等于完整全参数训练收敛。
