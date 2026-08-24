# 所有历史偏差候选的长程复核

仓库中有 **23 个唯一主矩阵 case ID**；本审计逐行复核 **13 个 extended candidate rows**（其中 11 个来自历史候选，另含 Llama、Ministral 的同族复现行）。合并后表共有 **28 行**。这些数字分别表示覆盖分母、候选分母和逐行审计行数，不能混用。

按当前口径，最终计入 **9 个**：其中直接长程方向案例 **5 个**，直接方向未持续但有长程配对 loss 分叉 **3 个**；反馈维持型案例单独列出。未决/不可安全重放共 **4 个**，不作阴性判断。

直接源案例只要 4096 步直接更新差异仍有稳定方向，就已经是持久性 bias；如果配对训练还观察到参数或 loss 分叉，就作为后果一并报告。反馈维持型案例只有在 4096 步反馈分离并观察到配对参数/loss gap 时才单独计入，不冒充直接源 bias。这里不要求两条训练轨迹收敛到不同的最终 loss，也不作这种声称。

| 模型 | 算子或位置 | 形成路径 | 4096 步直接结果 | 参数/loss 分叉 | 最终分类 |
|---|---|---|---|---|---|
| Qwen3-1.7B | fused CE dW accumulation | event/pairing imbalance | A4096=14.018，后半程 64/64 | 是；参数距离 9.27，末步 loss gap -0.13 | 最终持久性 bias 案例 |
| Phi-4-mini | lm_head backward dX | event/pairing imbalance + backward transport | A4096=46.090，后半程 64/64 | 是；参数距离 0.0191，末步 loss gap +0.00158 | 最终持久性 bias 案例 |
| Qwen3-1.7B | lm_head backward dX | event/pairing imbalance + backward transport | A4096=6.488，后半程 64/64 | 是；参数距离 0.00163，末步 loss gap +1.21e-05 | 最终持久性 bias 案例 |
| Qwen3-1.7B | seq64 v_proj MM + output rounding | conditional source asymmetry | A4096=0.993，p=0.511 | 是；参数距离 0.22，末步 loss gap -0.0048 | 有长程 loss 分叉，但直接方向未持续 |
| Qwen3-1.7B | v_proj MM/output rounding | local arithmetic/pairing | A4096=0.981，p=0.509 | 是；参数距离 0.302，末步 loss gap +0.000371 | 有长程 loss 分叉，但直接方向未持续 |
| Mamba-130M | in_proj matrix multiply | local arithmetic/pairing | A4096=1.110，p=0.108 | 未测 | 长程未保持 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | response/state-contract imbalance | A4096=1.195，p=0.084 | 是；参数距离 0.123，末步 loss gap -0.00228 | 有长程 loss 分叉，但直接方向未持续 |
| Qwen3-VL-Reranker-2B | SiLU backward | response asymmetry | COMPLETE_4096 | 是；参数距离 0.888，末步 loss gap -7.95e-09 | 反馈维持型 bias，且有 loss 分叉 |
| Qwen3-1.7B | attention S_bwd/K to q_proj | event/pairing imbalance | ABSTAIN | 未测 | 不可安全重放 |
| DeepSeek-R1-Qwen3-8B | attention dV BMM | formation unresolved; event/pairing candidate | NOT_RUN | 未测 | 形成阶段未确认，不升级长程 |
| Llama-3.2-3B | lm_head backward dX | event/pairing family replication | A4096=5.881，超过自身随机基线（窗口统计未导出） | 是；参数距离 0.000376，末步 loss gap +4.24e-05 | 最终持久性 bias 案例 |
| Ministral-3-3B | lm_head backward dX | event/pairing family replication | A4096=5.050，超过自身随机基线（窗口统计未导出） | 是；参数距离 0.00042，末步 loss gap +0 | 最终持久性 bias 案例 |
| Gemma-4 E2B | RMSNorm / projection feedback region | response asymmetry / feedback candidate | UNRESOLVED_LONG_REPLAY_RESOURCE | 未测 | 长程运行环境不再可重放，未决 |
| google/gemma-4-E2B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 没有可达载体，不适用 |
| google/gemma-4-E2B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 反馈对照，没有直接 bias 门 |
| google/gemma-4-E2B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 没有可达载体，不适用 |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/state-spaces/mamba-130m-hf | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/microsoft/Phi-4-mini-instruct | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 形成阶段未确认，不升级长程 |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |
| /data1/tzh/models/Qwen/Qwen3-1.7B | case-stage matrix row | complete roster row; no confirmed long-source gate | NOT_ESCALATED | 未测 | 短程筛查未确认直接 bias |

## 口径

- 32 步只能说明短程方向性，不能单独称为持久性 bias。
- 4096 步是同一训练状态下的直接更新审计，不等于完整全参数训练收敛。
- 配对 loss gap 是功能后果信号；这里不要求、也不声称两条轨迹收敛到不同的最终 loss。
- 反馈造成的轨迹分离不能替代直接 bias 证据。
