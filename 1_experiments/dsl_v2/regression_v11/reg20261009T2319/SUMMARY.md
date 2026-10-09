# DSL v2 回归（v1.1 程序，工具 4.0，运行 reg20261009T2319）

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

## 本次运行的范围（外部审计 1aee15e 修复版之后，提交 5fe8e56）

- 重跑了 28 个常规程序；结果与增量 1 的运行 reg20261009T1051 逐程序比较：观察、命中、K−G 判定与执行状态 28/28 相同。
- 4 个竞争程序（T5：prog_01、prog_13、prog_22、prog_23）这次没有在 GPU 上运行（审计任务书：不绕过权限运行竞争 kernel）。
  它们在 reg20261009T1051 的作业文件原样放在 [carried_from_reg20261009T1051/](carried_from_reg20261009T1051/)（工具为增量 1 的版本），当时统计已
  withheld；修复只会让准入更严，不会放宽。
- 与 v3.1 的比较（v31_comparison）在运行后由作业自身的输出重算：只比较 v3.1 写过的字段（递归），之后新增的字段（增量 2 的设计灵敏度、
  修复版的证明状态、resolution_met、未声明 δ 的说明）不算差异；v3.1 写过而 4.0 缺的字段仍算差异。重算脚本见
  3_audits/fix_1aee15e/evidence/scripts/recompare_v31.py。
