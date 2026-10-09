# 符号对照表（理论包 v1.0-rc2 ↔ 仓库字段；2026-10-08）

理论包原文不在仓库中；本表按任务书第 0 节给出的符号（K₀ 对照实现、G 分析语义目标、ℛ 包围）编写，**待理论包核对**。历史文件中的
字段名不改，新报告按本表读。

| 理论包 | 含义 | 仓库字段 / 位置 |
| --- | --- | --- |
| K | 被测实现在设备上的输出 | `k`（`check.run` 的 keep）、各报告的 `K` |
| K₀ | 对照实现（E 组的 eager 或另一实现） | E 组 `eager_*` 候选；`classify.combine` 的 `eref_class` |
| G | 分析语义目标：被测 kernel 的 TTIR 在实数算术下的值（舍入、浮点转换按恒等；求和与点积取精确值） | `K_R` 的实数语义；`ttir_eval.evaluate_sequence` |
| ℛ = [L, U] | G 的包围 | `r_lo` / `r_hi`（keep）、`K_R` 区间；`reference_classes` 中 `complete_composed` 的元素 |
| κ | 组合包含性的状态：complete / conditional / unestablished | `reference_classes`（`finite_complete_fraction`）、`not_established_reasons_seed0`；条件元素即 `buf.cond` |
| e_num = K − G | 数值差异 | `e_num`、`n_lo` / `n_hi`（`residual_interval`） |
| f | 任务规格（模式 B） | `1_experiments/specs/phase1`、`1_experiments/specs/phase2` 的规格函数；`spec(inputs)` |
| e_sem = G − f | 语义差异 | `e_sem`（`s_lo` / `s_hi`）；只在不读非 Triton 中间值的输出上作语义判据 |
| a_i = ⟨w, e_num⟩ | 第 i 个单位的投影 | 规则记录的 `per_unit_bounds`、`mean_projection` |
| R1, R5 | 固定均值类方向 | `negative_ones`、`fixed_direction` |
| R2, R3 | 对齐类方向 | `toward_zero`、`scale_down` |
| M | 有界路线的逐单位幅度界 | `measure.bounded_mean_test` 的 `M`；声明 `error_budget` |
| D_m | 下游映射（如 8 bit 实际写入） | `analysis` 的 `measurement`（`adamw_write` 等） |
