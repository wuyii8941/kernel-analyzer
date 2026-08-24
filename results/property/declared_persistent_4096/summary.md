# 历史有效案例的 4096 步复核

这里复核的是同一训练状态下，candidate 相对 repair 的直接更新差异。它不是完整训练收敛，也不包含两条轨迹分开后的反馈。

| 模型 | 算子或位置 | 执行情况 | 4096 步方向分数 | 后半程窗口 | 结果 |
|---|---|---|---:|---:|---|
| Qwen3-1.7B | lm_head backward dX | COMPLETE_4096 | 6.488 | 64/64 A>1 | 4096 步仍有稳定直接方向 |
| Qwen3-1.7B + Liger | fused cross-entropy dW accumulation | PENDING | — | — | 仍在运行 |
| Phi-4-mini | lm_head backward dX | COMPLETE_4096 | 46.090 | 64/64 过自身随机上界 | 4096 步仍有稳定直接方向 |
| Mamba-130M | in_proj matrix multiply | COMPLETE_4096 | 1.110 | 33/64 A>1 | 未形成稳健的 4096 步直接方向 |
| Qwen3-1.7B | layer-27 saved-P softmax backward | COMPLETE_4096 | 1.195 | 28/64 A>1 | 未形成稳健的 4096 步直接方向 |
| Qwen3-1.7B | layer-23 attention S_bwd/K to q_proj | ABSTAIN_REPAIR_IDENTITY_DRIFT_BEFORE_4096 | — | — | 历史实现不可重放，暂不判断 |

## 解释边界

- 32 步结果只叫短程方向性；本表单独报告 4096 步直接作用。
- `ROBUST` 是便于阅读的事后描述：整体 sign-flip p≤0.05，且后半程至少 75% 的窗口保持方向。原始分数、随机上界和窗口计数仍是主要证据。
- 4096 步直接作用仍不等于 loss 收敛后的功能差异。
- layer-23 的历史编译文件已不可用；当前重编译后 repair 为零差异，因此 fail-closed 地 abstain。
