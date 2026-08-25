# Liger 与 SiLU：长程口径复核

本页只引用 4096 步长程结果。早期 16/32 步、不同优化器或不同修复边界的数字仍保留在历史文档中，但不覆盖下面的长程标签。

## Liger fused CE

- **直接作用**：4096 步 `A=14.018`；后半程 64 个 32 步窗口全部保持方向；自身符号翻转随机基线的 95% 上界为 `1.163`，整体单侧 `p=0.000999`。
- **配对训练后果**：参数距离 `9.2663`；第 4096 步 loss gap（candidate − repair）为 `-0.13049`，后 512 步平均 gap 为 `+0.000673`。
- **统一标签**：`PERSISTENT_BIAS_WITH_PAIRED_LOSS_SPLIT`。
- **含义**：Liger 的方向在 fused CE 的低精度分块累加阶段已经出现，并在长程直接更新审计中保持；配对 loss 分叉作为后果单独记录。

## Qwen3-VL SiLU backward

- **局部直接作用**：4096 步 `A=1.017`，接近扩散，不能把它写成持续的局部 source bias。
- **反馈作用**：4096 步 `A=3.100`，最终实际分离 `0.8884`；反馈方向与最终分离 cosine 为 `0.999997`，局部方向 cosine 仅 `0.00552`。
- **配对训练后果**：第 4096 步 loss gap 为 `-7.95e-9`；后 512 步平均 gap 为 `+4.93e-8`，标准差 `1.29e-7`。
- **统一标签**：`FEEDBACK_SUSTAINED_BIAS_WITH_PAIRED_LOSS_SPLIT`。
- **含义**：SiLU 不是直接源方向案例；它是一个长程反馈维持案例。它仍计入最终 bias 案例，因为有效分离在长程保持，并且观察到了配对 loss 分叉。

## 统一规则

1. 32 步只能叫短程方向性，不能单独叫持久性 bias。
2. 直接作用在 4096 步保持方向，标为直接持久 bias。
3. 直接作用不保持、但反馈在 4096 步保持并且配对参数或 loss 出现分叉，标为反馈维持的持久 bias；它不冒充直接 source bias。
4. 无法安全重放的实验标为未决，不改写成阴性。

原始证据：

- `results/property/declared_persistent_4096/liger_fused_ce.json`
- `results/property/paired_loss_4096/liger_fused_ce.json`
- `results/property/declared_persistent_4096/qwen3vl_silu_4096_with_loss.json`
- `results/property/declared_persistent_4096/all_bias_case_audit.json`
