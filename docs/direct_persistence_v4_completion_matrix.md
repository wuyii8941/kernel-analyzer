# v4 完成度矩阵

这张表按原始计划逐项记录当前证据。`PARTIAL` 和 `ABSTAIN` 是有意保留的结果，不表示把缺失数据当成通过。

当前 16/32 步 v4 计划已经按有界结论收口；长期标签由单独的 4096 步复核覆盖。表中的“仍缺什么”表示升级更强主张所需的未来证据，不表示短程 v4 没有完成。

| 计划部分 | 当前状态 | 现有证据 | 仍缺什么 |
|---|---|---|---|
| 15 行回溯统计 | `COMPLETE` | 贡献表、三组 Holm 校正、BH 辅助结果、区间、逐对排序、短筛三态输出 | 0543 仍需独立复现 |
| v4 协议冻结 | `COMPLETE` | `protocol.json` 已冻结；规则、身份字段、严重度字段和误差基线均写入协议 | 历史运行无法补回缺失的原始字段 |
| 实验身份 | `PARTIAL_FAIL_CLOSED` | `identity_audit.json` 检查 18 行 | 历史行缺 moment-state 或坐标 digest；不使用相近运行补填 |
| 优化器同状态对照 | `COMPLETE`（4 个案例） | Liger、Phi、Qwen、Gemma feedback；另有 Qwen 早/中/晚真实阶段 | 0543 因运行包身份变化保持 `ABSTAIN` |
| 未见实现前瞻检查 | `COMPLETE`（3 行） | Gemma 新进程、预测先冻结、16 步筛查和 32 步确认均完成 | 3 行均无 direct positive，因此不能计算 recall/AUROC |
| 16/32 步方向筛查 | `PARTIAL` | 新 Gemma 行有保存向量上的 16/32 步重算 | 未覆盖历史冻结池的所有原始向量；不能给长期标签 |
| 4096 步长程复核 | `COMPLETE_WITH_FAIL_CLOSED_ABSTENTION` | Liger、Phi、Qwen、Mamba、saved-P 已完成；SiLU 为反馈维持型长程结果；layer-23 fail-closed abstain | Gemma4/DeepSeek 等仍按逐行审计表保持未决；完成项见 `results/property/declared_persistent_4096/summary.md` 与 `docs/all_bias_long_horizon_audit.md` |
| 影响大小 | `PARTIAL` | 新 Gemma 行有 direct resultant / candidate-update-path 比例 | 参数总范数、loss 投影、真实 loss 仍为 `ABSTAIN` |
| 传统误差标准对比 | `PARTIAL` | 3 个新 Gemma 行有输出/梯度/更新对、ULP 和 `rtol/atol`；历史两行有部分 raw 指标 | 历史冻结池缺原始 operand 和位级数据，不能生成全池表 |
| catch-and-fix | `NOT_APPLICABLE` | 前瞻池没有 direct positive | 没有合法的“先发现再修复”对象，不能人为制造 |
| 通用 Oracle | `NOT_SUPPORTED` | 审计明确限制为 cold-start AdamW Direct Persistence Screen | 不能把当前结果称为全算子安全证明 |

## 当前可以说什么

当前版本支持一个有明确适用范围的两级流程：在指定的 cold-start AdamW 设置下，用 16/32 步给实验排优先级；再用 warm-state 4096 步和后半程窗口给当前长程方向标签。最终轨迹分离仍要拆成直接作用和反馈作用。

## 当前不能说什么

- 不能说 18 行都有完整可重放身份；
- 不能说全冻结样本已经完成 ULP 和 `rtol/atol`；
- 不能说 3 个 Gemma 控制足以估计未见实现上的 recall 或 AUROC；
- 不能说优化器是数值偏差的根因；
- 不能说已经得到通用全算子 Oracle。
- 不能说 4096 步直接方向已经证明 loss 收敛到不同终点。

验证入口：

- `results/property/direct_persistence_v4/completion_audit.json`
- `results/property/direct_persistence_v4/identity_audit.json`
- `results/property/direct_persistence_v4/execution_status.json`

## v4.1 的下一轮冻结入口

`results/property/direct_persistence_v4_1/` 是一个新的、未开始测量的冻结清单。它只允许实验身份完整的行进入下一轮运行；旧 v4 中缺少原始字段的行被列在 `feasibility_report.json` 中，不会被补填或重新解释。

当前 v4.1 只有 1 行可运行模板，另外 3 行因缺少参数坐标、moment、repair 或 runner 身份而明确排除。v4.1 的文件是实验入口，不是新的科学结果。

当前决定是不启动这一轮运行。它保留为未来扩展入口，不影响 v4 有界结论。

已有的 Mamba 32 步结果另存为 `v4_1/development/mamba_0450_summary.json`，只作为跨架构的开发证据；它没有被冒充成 v4.1 的前瞻 held-out 结果。
