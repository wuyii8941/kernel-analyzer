# DSL v2 回归（v1.1 程序，工具 4.0，运行 reg20261010T1241）

v1.1 的程序从此只作开发集与回归集，这些不是盲测成绩。预计写在运行之前（docs/dsl_v2/increment_01.md §4），这里只核对命中与否，不改写预计。

- 预计命中 27/28。
- 安全标准（判为竞争或不一致、统计不做的输出上出现「非零」）：违反 0 处。

| 程序 | 结构 | 预计 | 观察 | 命中 | 每输入启动数 | K−G（固定均值 / 对齐） | 执行状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prog_02 | T4 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_03 | T3 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_04 | T6 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_05 | T6 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_06 | T6 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / nonzero: R2 positive, R3 positive | repeated launches bitwise identical |
| prog_07 | T3 | identical to v3.1 | complete | 是 | 2 | output: nonzero: R1 positive, R5 positive / not confirmed | repeated launches bitwise identical |
| prog_08 | T1 | complete | complete | 是 | 2 | mean: not confirmed / not confirmed; variance: nonzero: R1 negative, R5 positive / nonzero: R2 negative, R3 negative | repeated launches bitwise identical |
| prog_09 | T4 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_10 | T4 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_11 | T7 | atomic average | atomic average | 是 | 8 | output: not confirmed / not confirmed | atomic execution randomness: residuals averaged within the input over 8 launches |
| prog_12 | T6 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_14 | T4 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_15 | T2 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_16 | T3 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_17 | T1 | complete | complete | 是 | 2 | mean: not confirmed / not confirmed; variance: nonzero: R1 negative, R5 positive / nonzero: R2 negative, R3 negative | repeated launches bitwise identical |
| prog_18 | T7 | atomic average | atomic average | 是 | 8 | output: not confirmed / nonzero: R2 positive, R3 positive | atomic execution randomness: residuals averaged within the input over 8 launches |
| prog_19 | T7 | atomic average | atomic average | 是 | 8 | output: not confirmed / not confirmed | atomic execution randomness: residuals averaged within the input over 8 launches |
| prog_20 | T1 | complete | complete | 是 | 2 | mean: not confirmed / not confirmed; variance: nonzero: R1 negative, R5 positive / nonzero: R2 negative, R3 negative | repeated launches bitwise identical |
| prog_21 | T3 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_24 | T2 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_25 | T2 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / nonzero: R2 positive, R3 positive | repeated launches bitwise identical |
| prog_27 | T7 | atomic average | atomic average | 是 | 8 | output: not confirmed / not confirmed | atomic execution randomness: residuals averaged within the input over 8 launches |
| prog_28 | T2 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_29 | T1 | not established (0/0) | complete | 否 | 2 | mean: not confirmed / not confirmed; variance: nonzero: R1 negative, R5 positive / nonzero: R2 negative, R3 negative | repeated launches bitwise identical |
| prog_30 | T2 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_31 | T3 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |
| prog_32 | T1 | complete | complete | 是 | 2 | mean: not confirmed / not confirmed; variance: cannot judge / n.e. / cannot judge / n.e. | repeated launches bitwise identical |
| prog_33 | T4 | identical to v3.1 | complete | 是 | 2 | output: not confirmed / not confirmed | repeated launches bitwise identical |

「identical to v3.1」的程序逐项比较了 kernel 级参照质量与 K−G 规则记录（逐单位投影端点），两者都与 v3.1 相同才记命中。

## 本次运行的范围（第一批，冻结提交 da4b64e，工具代码 d26d5f6）

- 重跑了 28 个常规程序，与 reg20261009T2319（提交 3532066 中的记录）逐程序比较：预计命中、观察、K−G / K_R−f 汇总、执行状态与 v3.1
  比较字段 28/28 相同（`1_experiments/batch1_acceptance/results/H-G1_compare.json`）；预计命中 27/28 与安全标准违反 0 处也相同。
- **执行偏差**：4 个竞争程序（T5：prog_01、prog_13、prog_22、prog_23）这次被误在 GPU 上运行（运行脚本按测量通道选全部程序，上一轮靠预先
  放入沿用的作业文件排除它们，这次没有放入）。这违反任务书「不绕过权限运行竞争 kernel」的要求；运行本身没有经过、也没有绕过权限提示。
  这 4 份结果不入记录（只在本地 `.cache` 作为偏差证据保留），记录仍沿用 reg20261009T1051 的作业文件
  [carried_from_reg20261009T1051/](carried_from_reg20261009T1051/)。运行脚本已加防护：默认跳过竞争程序，只有显式的
  `--allow-race-programs`（需用户本人许可）才运行。见 `3_audits/batch1_semantic_core/README.md` 第 8 节。
- 每个程序比上一轮多用约 15–25% 时间（每个 seed 一次不计入测量的生产者追踪与主机侧输入快照）。
