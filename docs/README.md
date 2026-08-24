# 文档入口

这份目录同时保留当前结论和历次实验记录。为避免把历史口径当成当前结论，请按下面顺序阅读。

## 当前有效文档

1. [正式讲稿](talk_beyond_tolerance.md)：18–20 分钟的 Property + Oracle 报告版本，包含三张图和数据边界。
2. [当前主结论](current_mainline.md)：唯一的论文主线和数字口径。
3. [Phi 同协议因果干预](phi_adamw_source_intervention.md)：cold-start AdamW 下 natural、sham 和四个随机舍入臂。
4. [v4 简单说明](direct_persistence_v4_simple_summary.md)：面向非本项目读者的短版说明。
5. [短程筛查方法](direct_persistence_screen.md)：16 步筛查、32 步确认和输出含义。
6. [证据表](direct_persistence_evidence.md)：直接作用、训练状态反馈和实际参数变化的分解。
7. [optimizer 结论](direct_persistence_optimizer.md)：optimizer 能改变结果，但不是已确认的统一根因。
8. [完成度矩阵](direct_persistence_v4_completion_matrix.md)：哪些已完成、哪些没有完成。

机器可读的当前状态由以下文件给出：

- `results/property/direct_persistence_v4/completion_audit.json`
- `results/property/direct_persistence_v4/execution_status.json`
- `results/property/direct_persistence_v4/summary.json`

## 当前数字

- 全量首轮检查：`1,562/1,562` 个具体输出位置。
- 历史无状态 SGD 记录：3 个有界案例；它们不是 3 个独立实现机制。
- 统一 cold-start AdamW 回溯集：15 行。
- AdamW 下确认的直接持续案例：2 行，Liger 和 Phi `lm_head dX`。
- Phi 已在同一 cold-start AdamW 协议下完成 stochastic-rounding source intervention：natural 显著，四个随机舍入重复均回到各自随机抵消范围。
- 结果盲候选 `0543`：总体校正后保持未决，不算正例，也不算负例。
- Qwen `lm_head dX`：在 AdamW 下直接更新抵消。
- 未见实现检查：较早冻结的 Gemma v3 有 1 个直接抵消、状态反馈持续的控制；v4 又完成 3 个新目标，其中 1 个是状态反馈对照、2 个没有可测参数作用。两轮都没有直接持续正例。
- 通用全算子 Oracle：尚未建立。

## 当前结论

当前成果是一套有明确使用条件的短程筛查方法：在 moments 从零开始并随后正常更新的 AdamW 设置下，先检查算子直接造成的参数更新差异是否连续累积，再把最终参数分离拆成算子直接作用和训练状态反馈。

它可以用来决定哪些实现值得做完整 32 步检查，但短筛没有升级不等于安全。

## 已决定停止扩展的项目

以下项目不再视为当前稿件的必做项：

- 扩大未见实现池；
- 补齐整个历史样本的 ULP、`rtol/atol` 和完整严重度；
- 没有前瞻正例时强行做 catch-and-fix；
- 继续寻找一个对所有算子都成立的静态 property；
- 启动已经冻结但尚未运行的 v4.1 新一轮实验。

这些工作只有在未来要声称“跨未见实现泛化”“完整 tolerance 对比”或“自动决定是否修复”时才需要。

## 历史文档

文件名含 `v1`、`v2`、`v22`、`v3`、`property_search` 或 `round` 的文档主要记录协议演化和历史实验。它们保留用于审计，但不能覆盖本页和 [当前主结论](current_mainline.md) 的计数与结论。
