# 范围修订：原始执行表与有效测量表（运行 r20261008T2030）

依据 2026-10-08 出题方与审阅方的处理。原报告 `REPORT.md`、`summary.json`、p 值与各检验族的判断都不改。本轮登记的是
608 个互相独立的检验族，每个族内做 Holm，不是一次全局 Holm。标签评分表要等原出题方交付答案文件，现在不做。

- 原报告的程序级「27/33 建立」属于原始执行表，含 T5 的四个常规项，不能当作修订后有效任务的成功率。
- 有限均值比较前提成立的程序 28/33；其中全部输出参照已建立的 23/28（未建立的是 5 个 T1 程序，它们的 F 与 baseline 仍可用）。
- 排除：竞争裁决 ['prog_01', 'prog_13', 'prog_22', 'prog_23']，移出有限均值标签评分但保留记录；边界项 ['prog_26']，本轮未执行。
- 这个集合给不出均值检测的误报率：一致性对检查的是程序之间的差，不提供各程序相对 G 的零均值标签。

## 原始执行表（全部 33 项）

| 程序 | 结构 | 输出 | 通道 | 作业状态 | 参照（kernel 级） | 主轮 K 非有限元素 |
| --- | --- | --- | --- | --- | --- | --- |
| prog_01 | T5 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_02 | T4 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_03 | T3 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_04 | T6 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_05 | T6 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_06 | T6 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_07 | T3 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_08 | T1 | mean | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_08 | T1 | variance | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_09 | T4 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_10 | T4 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_11 | T7 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_12 | T6 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_13 | T5 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_14 | T4 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_15 | T2 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_16 | T3 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_17 | T1 | mean | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_17 | T1 | variance | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_18 | T7 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_19 | T7 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_20 | T1 | mean | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_20 | T1 | variance | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_21 | T3 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_22 | T5 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_23 | T5 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_24 | T2 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_25 | T2 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_26 | T5 | output | isolated_boundary | main_B not run, main_A not run, replication_B not run | not run | – |
| prog_27 | T7 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_28 | T2 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_29 | T1 | mean | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_29 | T1 | variance | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_30 | T2 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_31 | T3 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |
| prog_32 | T1 | mean | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_32 | T1 | variance | measurement | main_B ok, main_A ok, replication_B ok, D1_main ok, D1_replication ok | not established (semantics missing) | 0 |
| prog_33 | T4 | output | measurement | main_B ok, main_A ok, replication_B ok | established | 0 |

调用级完整一律未建立（协议 §1）。

## 有效测量表

| 程序 | 输出 | 有限均值比较前提 | 启动可复现性 | 可用比较对象 |
| --- | --- | --- | --- | --- |
| prog_01 | output | race boundary (issuer ruling 2026-10-08: buf written then read in the same program without a barrier); removed from finite-mean label scoring; raw records and execution validity kept | 0/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_02 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_03 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_04 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_05 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_06 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_07 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_08 | mean | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_08 | variance | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_09 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_10 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_11 | output | valid under the atomic protocol (8 executions per input, within-input mean); execution randomness by design | order-random by design (atomic): not required bitwise | FR_e_num, FR_e_sem, F_total, baseline |
| prog_12 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_13 | output | race boundary (issuer ruling 2026-10-08: buf written then read in the same program without a barrier); removed from finite-mean label scoring; raw records and execution validity kept | 0/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_14 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_15 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_16 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_17 | mean | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_17 | variance | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_18 | output | valid under the atomic protocol (8 executions per input, within-input mean); execution randomness by design | order-random by design (atomic): not required bitwise | FR_e_num, FR_e_sem, F_total, baseline |
| prog_19 | output | valid under the atomic protocol (8 executions per input, within-input mean); execution randomness by design | order-random by design (atomic): not required bitwise | FR_e_num, FR_e_sem, F_total, baseline |
| prog_20 | mean | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_20 | variance | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_21 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_22 | output | race boundary (issuer ruling 2026-10-08: buf written then read in the same program without a barrier); removed from finite-mean label scoring; raw records and execution validity kept | 0/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_23 | output | race boundary (issuer ruling 2026-10-08: buf written then read in the same program without a barrier); removed from finite-mean label scoring; raw records and execution validity kept | 0/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_24 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_25 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_26 | output | boundary item: not run this round (environment permission check); not a correct rejection, not a numerical negative; does not block other scoring | – | – |
| prog_27 | output | valid under the atomic protocol (8 executions per input, within-input mean); execution randomness by design | order-random by design (atomic): not required bitwise | FR_e_num, FR_e_sem, F_total, baseline |
| prog_28 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_29 | mean | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_29 | variance | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_30 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_31 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |
| prog_32 | mean | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_32 | variance | valid | not checked (one launch per input with a kept K) | F_total, baseline |
| prog_33 | output | valid | 96/96 units bitwise equal between the two main-round launches (A, B) | FR_e_num, FR_e_sem, F_total, baseline |

## 标签评分表

未做：等原出题方交付两份原始答案文件、各自完整的 64 位 SHA-256、对应的包版本与计分协议，以及此前的承诺记录。
本轮定位预算为 0，答案里若有定位标签，记为「未执行定位」。
