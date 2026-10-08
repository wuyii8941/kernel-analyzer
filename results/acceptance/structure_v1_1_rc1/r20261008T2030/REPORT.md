# 结构验收集 v1.1-rc1 运行 r20261008T2030（冻结工具 general-v3.1）

> 本报告只汇总冻结规则下的测量，不是正式盲评分数：包状态 `ISSUER_RESEAL_REQUIRED`，新答案未签署，敏感度 / 特异度不计算（seal_status.json）。结论只按三种状态写：建立 / 未建立 / 无法判断。

- 工具提交 `52412958f8ed`（src 树与 general-v3.1 相同），冻结记录 `RUN_FREEZE.json` 在测量前提交；包 SHA256SUMS 哈希 `92fdb0fe046bb6b5…`，协议哈希 `f1f61238b35c9e7e…`。
- 作业：96/96 有结果，状态 {'ok': 96}；缺失 0。
- 汇总脚本 `scripts/acceptance/structure_v11_report.py` 写于冻结之后，只读作业文件，不做测量。

## 1. 完整率（主轮、模式 B）

- 程序级「全部必要输出已建立」：27/33（全部 33 项）；27/32（32 个常规项条件比例）。
- 输出级：27/38 个输出参照完整（kernel 级）。
- TTIR 覆盖完整的程序：27/32。
- 调用级完整：一律未建立（协议 §1：仅凭内容相同或全零不升级为调用级）；工具自身的调用级分类另列在 summary.json。
- 边界项：{'prog_26': 'not run (isolated opt-in not approved)'}，不计入有限均值检出率，也不算作正确拒绝。

| 程序 | 结构 | 输出 | 状态 | 完整率 | 分辨率达标 | 宽度/ulp 中位 / p90 / max | 失败类别或原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prog_01 | T5 | output | 建立 | 1 | 1 | 7.45e-09 / 2.98e-08 / 0.0625 |  |
| prog_02 | T4 | output | 建立 | 1 | 1 | 0 / 0 / 0 |  |
| prog_03 | T3 | output | 建立 | 1 | 1 | 5.59e-08 / 3.43e-07 / 9.63e-05 |  |
| prog_04 | T6 | output | 建立 | 1 | 1 | 9.31e-09 / 1.3e-08 / 1.68e-08 |  |
| prog_05 | T6 | output | 建立 | 1 | 1 | 9.31e-09 / 1.3e-08 / 1.68e-08 |  |
| prog_06 | T6 | output | 建立 | 1 | 1 | 9.31e-09 / 1.3e-08 / 1.68e-08 |  |
| prog_07 | T3 | output | 建立 | 1 | 1 | 5.59e-08 / 3.43e-07 / 9.63e-05 |  |
| prog_08 | T1 | mean | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_08 | T1 | variance | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_09 | T4 | output | 建立 | 1 | 1 | 0 / 0 / 0 |  |
| prog_10 | T4 | output | 建立 | 1 | 1 | 0 / 0 / 0 |  |
| prog_11 | T7 | output | 建立 | 1 | 1 | 3.17e-08 / 1.56e-07 / 4.96e-05 |  |
| prog_12 | T6 | output | 建立 | 1 | 1 | 9.31e-09 / 1.3e-08 / 1.68e-08 |  |
| prog_13 | T5 | output | 建立 | 1 | 1 | 7.45e-09 / 2.98e-08 / 0.0625 |  |
| prog_14 | T4 | output | 建立 | 1 | 1 | 0 / 0 / 0 |  |
| prog_15 | T2 | output | 建立 | 1 | 1 | 6.55e-05 / 0.000373 / 668 | enclosure too wide |
| prog_16 | T3 | output | 建立 | 1 | 1 | 5.03e-08 / 3.13e-07 / 4.57e-05 |  |
| prog_17 | T1 | mean | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_17 | T1 | variance | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_18 | T7 | output | 建立 | 1 | 1 | 3.17e-08 / 1.56e-07 / 4.96e-05 |  |
| prog_19 | T7 | output | 建立 | 1 | 1 | 3.17e-08 / 1.56e-07 / 4.96e-05 |  |
| prog_20 | T1 | mean | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_20 | T1 | variance | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_21 | T3 | output | 建立 | 1 | 1 | 5.59e-08 / 3.43e-07 / 9.63e-05 |  |
| prog_22 | T5 | output | 建立 | 1 | 1 | 7.45e-09 / 2.98e-08 / 0.0625 |  |
| prog_23 | T5 | output | 建立 | 1 | 1 | 7.45e-09 / 2.98e-08 / 0.0625 |  |
| prog_24 | T2 | output | 建立 | 1 | 1 | 6.55e-05 / 0.000373 / 668 | enclosure too wide |
| prog_25 | T2 | output | 建立 | 1 | 1 | 6.55e-05 / 0.000373 / 668 | enclosure too wide |
| prog_26 | T5 | output | not run | – | – | – / – / – |  |
| prog_27 | T7 | output | 建立 | 1 | 1 | 3.17e-08 / 1.56e-07 / 4.96e-05 |  |
| prog_28 | T2 | output | 建立 | 1 | 1 | 6.55e-05 / 0.000373 / 668 | enclosure too wide |
| prog_29 | T1 | mean | not established | – | – | – / – / – | semantics missing: tt.reduce@16: unrecognized reduction combiner |
| prog_29 | T1 | variance | not established | – | – | – / – / – | semantics missing: tt.reduce@16: unrecognized reduction combiner |
| prog_30 | T2 | output | 建立 | 1 | 0.999 | 6.57e-05 / 0.000378 / 4.55e+32 | enclosure too wide |
| prog_31 | T3 | output | 建立 | 1 | 1 | 5.59e-08 / 3.43e-07 / 9.63e-05 |  |
| prog_32 | T1 | mean | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_32 | T1 | variance | not established | – | – | – / – / – | semantics missing: tt.reduce@18: unrecognized reduction combiner |
| prog_33 | T4 | output | 建立 | 1 | 1 | 0 / 0 / 0 |  |

## 2. 统计（Holm 按登记 family，缺格按 p = 1）

登记 family 608 个（两轮 × 33 程序 × 输出 × 4 个比较对象 × 2 个规则类）。各轮各对象的状态计数：

| 轮 / 对象 / 规则类 | 非零 | 未确认 | 无法判断 / 未建立 |
| --- | --- | --- | --- |
| main/FR_e_num/aligned | 3 | 24 | 11 |
| main/FR_e_num/fixed_mean | 5 | 22 | 11 |
| main/FR_e_sem/aligned | 3 | 20 | 15 |
| main/FR_e_sem/fixed_mean | 1 | 22 | 15 |
| main/F_total/aligned | 12 | 25 | 1 |
| main/F_total/fixed_mean | 11 | 26 | 1 |
| main/baseline/aligned | 12 | 25 | 1 |
| main/baseline/fixed_mean | 11 | 26 | 1 |
| replication/FR_e_num/aligned | 3 | 24 | 11 |
| replication/FR_e_num/fixed_mean | 5 | 22 | 11 |
| replication/FR_e_sem/aligned | 2 | 21 | 15 |
| replication/FR_e_sem/fixed_mean | 1 | 22 | 15 |
| replication/F_total/aligned | 12 | 25 | 1 |
| replication/F_total/fixed_mean | 11 | 26 | 1 |
| replication/baseline/aligned | 12 | 25 | 1 |
| replication/baseline/fixed_mean | 11 | 26 | 1 |

### 固定向量均值（R1, R5），主轮模式 B

| 程序 | 输出 | K−G | G−f | K−f | 普通参照 K−f64 |
| --- | --- | --- | --- | --- | --- |
| prog_01 | output | nonzero: R1 negative, R5 positive | not confirmed | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_02 | output | not confirmed | nonzero: R1 positive, R5 positive | nonzero: R1 positive, R5 positive | nonzero: R1 positive, R5 positive |
| prog_03 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_04 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_05 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_06 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_07 | output | nonzero: R1 positive, R5 positive | not confirmed | nonzero: R1 positive, R5 positive | nonzero: R1 positive, R5 positive |
| prog_08 | mean | – | – | not confirmed | not confirmed |
| prog_08 | variance | – | – | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_09 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_10 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_11 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_12 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_13 | output | nonzero: R1 negative, R5 positive | not confirmed | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_14 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_15 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_16 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_17 | mean | – | – | not confirmed | not confirmed |
| prog_17 | variance | – | – | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_18 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_19 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_20 | mean | – | – | not confirmed | not confirmed |
| prog_20 | variance | – | – | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_21 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_22 | output | nonzero: R1 negative, R5 positive | not confirmed | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_23 | output | nonzero: R1 negative, R5 positive | not confirmed | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_24 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_25 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_27 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_28 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_29 | mean | – | – | not confirmed | not confirmed |
| prog_29 | variance | – | – | nonzero: R1 negative, R5 positive | nonzero: R1 negative, R5 positive |
| prog_30 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_31 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_32 | mean | – | – | not confirmed | not confirmed |
| prog_32 | variance | – | – | nonzero: R1 positive, R5 positive | nonzero: R1 positive, R5 positive |
| prog_33 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |

### 对齐作用（R2, R3），主轮模式 B

| 程序 | 输出 | K−G | G−f | K−f | 普通参照 K−f64 |
| --- | --- | --- | --- | --- | --- |
| prog_01 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_02 | output | not confirmed | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_03 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_04 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_05 | output | not confirmed | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_06 | output | nonzero: R2 positive, R3 positive | not confirmed | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_07 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_08 | mean | – | – | not confirmed | not confirmed |
| prog_08 | variance | – | – | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_09 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_10 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_11 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_12 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_13 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_14 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |
| prog_15 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_16 | output | not confirmed | not confirmed | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_17 | mean | – | – | not confirmed | not confirmed |
| prog_17 | variance | – | – | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_18 | output | nonzero: R2 positive, R3 positive | not confirmed | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_19 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_20 | mean | – | – | not confirmed | not confirmed |
| prog_20 | variance | – | – | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_21 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_22 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_23 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_24 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_25 | output | nonzero: R2 positive, R3 positive | not confirmed | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_27 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_28 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_29 | mean | – | – | not confirmed | not confirmed |
| prog_29 | variance | – | – | nonzero: R2 negative, R3 negative | nonzero: R2 negative, R3 negative |
| prog_30 | output | not confirmed | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_31 | output | not confirmed | not confirmed | not confirmed | not confirmed |
| prog_32 | mean | – | – | not confirmed | not confirmed |
| prog_32 | variance | – | – | nonzero: R2 positive, R3 positive | nonzero: R2 positive, R3 positive |
| prog_33 | output | not confirmed | cannot judge / n.e. | not confirmed | not confirmed |

规则级效应、原始与校正 p 值、逐单位投影端点在 `jobs/*.json`（`outputs.<name>.comparisons.<对象>.record`）。「未确认」不等于没有 bias；「非零」是对该投影平均作用的判断，不是逐元素契约违反。

### 非零判断的规则级明细（模式 B，两轮）

| 轮 | 程序 | 输出 | 对象 | 规则类 | 规则 | 判断 | 平均投影 | n | 原始 p | Holm p（登记 family） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | prog_01 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -62.1 | 64 | 1e-80 | 2e-80 |
| main | prog_01 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 92.8 | 64 | 1.24e-80 | 2e-80 |
| main | prog_01 | output | F_total | fixed_mean | R1 | nonzero (negative) | -62.1 | 64 | 1e-80 | 2e-80 |
| main | prog_01 | output | F_total | fixed_mean | R5 | nonzero (positive) | 92.8 | 64 | 1.24e-80 | 2e-80 |
| main | prog_01 | output | baseline | fixed_mean | R1 | nonzero (negative) | -62.1 | 64 | 1e-80 | 2e-80 |
| main | prog_01 | output | baseline | fixed_mean | R5 | nonzero (positive) | 92.8 | 64 | 1.24e-80 | 2e-80 |
| main | prog_02 | output | FR_e_sem | fixed_mean | R1 | nonzero (positive) | 0.732 | 64 | 6.21e-58 | 1.24e-57 |
| main | prog_02 | output | FR_e_sem | fixed_mean | R5 | nonzero (positive) | 4.13 | 64 | 4.23e-47 | 4.23e-47 |
| main | prog_02 | output | FR_e_sem | aligned | R2 | nonzero (negative) | -3.39 | 64 | 6.86e-110 | 1.37e-109 |
| main | prog_02 | output | FR_e_sem | aligned | R3 | nonzero (negative) | -5.22 | 64 | 2.4e-92 | 2.4e-92 |
| main | prog_02 | output | F_total | fixed_mean | R1 | nonzero (positive) | 0.732 | 64 | 6.21e-58 | 1.24e-57 |
| main | prog_02 | output | F_total | fixed_mean | R5 | nonzero (positive) | 4.13 | 64 | 4.23e-47 | 4.23e-47 |
| main | prog_02 | output | F_total | aligned | R2 | nonzero (negative) | -3.14 | 64 | 1.41e-105 | 2.83e-105 |
| main | prog_02 | output | F_total | aligned | R3 | nonzero (negative) | -2.02 | 64 | 3.66e-91 | 3.66e-91 |
| main | prog_02 | output | baseline | fixed_mean | R1 | nonzero (positive) | 0.732 | 64 | 6.21e-58 | 1.24e-57 |
| main | prog_02 | output | baseline | fixed_mean | R5 | nonzero (positive) | 4.13 | 64 | 4.23e-47 | 4.23e-47 |
| main | prog_02 | output | baseline | aligned | R2 | nonzero (negative) | -3.14 | 64 | 1.41e-105 | 2.83e-105 |
| main | prog_02 | output | baseline | aligned | R3 | nonzero (negative) | -2.02 | 64 | 3.66e-91 | 3.66e-91 |
| main | prog_05 | output | FR_e_sem | aligned | R2 | nonzero (negative) | -0.0705 | 64 | 9.15e-43 | 9.15e-43 |
| main | prog_05 | output | FR_e_sem | aligned | R3 | nonzero (negative) | -0.267 | 64 | 4.44e-44 | 8.88e-44 |
| main | prog_05 | output | F_total | aligned | R2 | nonzero (negative) | -0.0705 | 64 | 9.14e-43 | 9.14e-43 |
| main | prog_05 | output | F_total | aligned | R3 | nonzero (negative) | -0.0889 | 64 | 4.04e-44 | 8.08e-44 |
| main | prog_05 | output | baseline | aligned | R2 | nonzero (negative) | -0.0705 | 64 | 9.14e-43 | 9.14e-43 |
| main | prog_05 | output | baseline | aligned | R3 | nonzero (negative) | -0.0889 | 64 | 4.04e-44 | 8.08e-44 |
| main | prog_06 | output | FR_e_num | aligned | R2 | nonzero (positive) | 1.23e-05 | 64 | 2.32e-58 | 4.37e-58 |
| main | prog_06 | output | FR_e_num | aligned | R3 | nonzero (positive) | 1.55e-05 | 64 | 2.19e-58 | 4.37e-58 |
| main | prog_06 | output | F_total | aligned | R2 | nonzero (positive) | 1.23e-05 | 64 | 2.32e-58 | 4.37e-58 |
| main | prog_06 | output | F_total | aligned | R3 | nonzero (positive) | 1.55e-05 | 64 | 2.19e-58 | 4.37e-58 |
| main | prog_06 | output | baseline | aligned | R2 | nonzero (positive) | 1.23e-05 | 64 | 2.32e-58 | 4.37e-58 |
| main | prog_06 | output | baseline | aligned | R3 | nonzero (positive) | 1.55e-05 | 64 | 2.19e-58 | 4.37e-58 |
| main | prog_07 | output | FR_e_num | fixed_mean | R1 | nonzero (positive) | 0.00302 | 64 | 6.39e-120 | 1.23e-119 |
| main | prog_07 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 0.00302 | 64 | 6.17e-120 | 1.23e-119 |
| main | prog_07 | output | F_total | fixed_mean | R1 | nonzero (positive) | 0.00302 | 64 | 6.39e-120 | 1.23e-119 |
| main | prog_07 | output | F_total | fixed_mean | R5 | nonzero (positive) | 0.00302 | 64 | 6.17e-120 | 1.23e-119 |
| main | prog_07 | output | baseline | fixed_mean | R1 | nonzero (positive) | 0.00302 | 64 | 6.39e-120 | 1.23e-119 |
| main | prog_07 | output | baseline | fixed_mean | R5 | nonzero (positive) | 0.00302 | 64 | 6.17e-120 | 1.23e-119 |
| main | prog_13 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -59.1 | 64 | 2.62e-85 | 5.25e-85 |
| main | prog_13 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 89.5 | 64 | 1.77e-80 | 1.77e-80 |
| main | prog_13 | output | F_total | fixed_mean | R1 | nonzero (negative) | -59.1 | 64 | 2.62e-85 | 5.25e-85 |
| main | prog_13 | output | F_total | fixed_mean | R5 | nonzero (positive) | 89.5 | 64 | 1.77e-80 | 1.77e-80 |
| main | prog_13 | output | baseline | fixed_mean | R1 | nonzero (negative) | -59.1 | 64 | 2.62e-85 | 5.25e-85 |
| main | prog_13 | output | baseline | fixed_mean | R5 | nonzero (positive) | 89.5 | 64 | 1.77e-80 | 1.77e-80 |
| main | prog_16 | output | F_total | aligned | R2 | nonzero (positive) | 24.2 | 64 | 2.87e-19 | 2.87e-19 |
| main | prog_16 | output | F_total | aligned | R3 | nonzero (positive) | 28.8 | 64 | 5.74e-21 | 1.15e-20 |
| main | prog_16 | output | baseline | aligned | R2 | nonzero (positive) | 24.2 | 64 | 2.87e-19 | 2.87e-19 |
| main | prog_16 | output | baseline | aligned | R3 | nonzero (positive) | 28.8 | 64 | 5.74e-21 | 1.15e-20 |
| main | prog_18 | output | FR_e_num | aligned | R2 | nonzero (positive) | 1.77e-05 | 64 | 4.22e-34 | 8.32e-34 |
| main | prog_18 | output | FR_e_num | aligned | R3 | nonzero (positive) | 2.27e-05 | 64 | 4.16e-34 | 8.32e-34 |
| main | prog_18 | output | F_total | aligned | R2 | nonzero (positive) | 1.77e-05 | 64 | 4.22e-34 | 8.32e-34 |
| main | prog_18 | output | F_total | aligned | R3 | nonzero (positive) | 2.27e-05 | 64 | 4.16e-34 | 8.32e-34 |
| main | prog_18 | output | baseline | aligned | R2 | nonzero (positive) | 1.77e-05 | 64 | 4.22e-34 | 8.32e-34 |
| main | prog_18 | output | baseline | aligned | R3 | nonzero (positive) | 2.27e-05 | 64 | 4.16e-34 | 8.32e-34 |
| main | prog_22 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -60.6 | 64 | 2.06e-89 | 4.12e-89 |
| main | prog_22 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 93 | 64 | 2.77e-84 | 2.77e-84 |
| main | prog_22 | output | F_total | fixed_mean | R1 | nonzero (negative) | -60.6 | 64 | 2.06e-89 | 4.12e-89 |
| main | prog_22 | output | F_total | fixed_mean | R5 | nonzero (positive) | 93 | 64 | 2.77e-84 | 2.77e-84 |
| main | prog_22 | output | baseline | fixed_mean | R1 | nonzero (negative) | -60.6 | 64 | 2.06e-89 | 4.12e-89 |
| main | prog_22 | output | baseline | fixed_mean | R5 | nonzero (positive) | 93 | 64 | 2.77e-84 | 2.77e-84 |
| main | prog_23 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -59.8 | 64 | 3.76e-75 | 7.53e-75 |
| main | prog_23 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 87 | 64 | 9.38e-72 | 9.38e-72 |
| main | prog_23 | output | F_total | fixed_mean | R1 | nonzero (negative) | -59.8 | 64 | 3.76e-75 | 7.53e-75 |
| main | prog_23 | output | F_total | fixed_mean | R5 | nonzero (positive) | 87 | 64 | 9.38e-72 | 9.38e-72 |
| main | prog_23 | output | baseline | fixed_mean | R1 | nonzero (negative) | -59.8 | 64 | 3.76e-75 | 7.53e-75 |
| main | prog_23 | output | baseline | fixed_mean | R5 | nonzero (positive) | 87 | 64 | 9.38e-72 | 9.38e-72 |
| main | prog_25 | output | FR_e_num | aligned | R2 | nonzero (positive) | 0.000104 | 64 | 2.51e-33 | 5.03e-33 |
| main | prog_25 | output | FR_e_num | aligned | R3 | nonzero (positive) | 0.00014 | 64 | 4.01e-30 | 4.01e-30 |
| main | prog_25 | output | F_total | aligned | R2 | nonzero (positive) | 0.000104 | 64 | 2.5e-33 | 5e-33 |
| main | prog_25 | output | F_total | aligned | R3 | nonzero (positive) | 0.00014 | 64 | 3.99e-30 | 3.99e-30 |
| main | prog_25 | output | baseline | aligned | R2 | nonzero (positive) | 0.000104 | 64 | 2.5e-33 | 5e-33 |
| main | prog_25 | output | baseline | aligned | R3 | nonzero (positive) | 0.00014 | 64 | 3.99e-30 | 3.99e-30 |
| main | prog_30 | output | FR_e_sem | aligned | R2 | nonzero (positive) | 0.222 | 64 | 0.0118 | 0.0236 |
| main | prog_30 | output | FR_e_sem | aligned | R3 | nonzero (positive) | 0.188 | 64 | 0.0455 | 0.0455 |
| main | prog_30 | output | F_total | aligned | R2 | nonzero (positive) | 9.38 | 64 | 1.3e-78 | 1.3e-78 |
| main | prog_30 | output | F_total | aligned | R3 | nonzero (positive) | 8.5 | 64 | 5.88e-82 | 1.18e-81 |
| main | prog_30 | output | baseline | aligned | R2 | nonzero (positive) | 9.38 | 64 | 1.3e-78 | 1.3e-78 |
| main | prog_30 | output | baseline | aligned | R3 | nonzero (positive) | 8.5 | 64 | 5.88e-82 | 1.18e-81 |
| replication | prog_01 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -59 | 96 | 2.65e-121 | 5.29e-121 |
| replication | prog_01 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 85.7 | 96 | 1.2e-119 | 1.2e-119 |
| replication | prog_01 | output | F_total | fixed_mean | R1 | nonzero (negative) | -59 | 96 | 2.65e-121 | 5.29e-121 |
| replication | prog_01 | output | F_total | fixed_mean | R5 | nonzero (positive) | 85.7 | 96 | 1.2e-119 | 1.2e-119 |
| replication | prog_01 | output | baseline | fixed_mean | R1 | nonzero (negative) | -59 | 96 | 2.65e-121 | 5.29e-121 |
| replication | prog_01 | output | baseline | fixed_mean | R5 | nonzero (positive) | 85.7 | 96 | 1.2e-119 | 1.2e-119 |
| replication | prog_02 | output | FR_e_sem | fixed_mean | R1 | nonzero (positive) | 0.709 | 96 | 9.44e-87 | 1.89e-86 |
| replication | prog_02 | output | FR_e_sem | fixed_mean | R5 | nonzero (positive) | 3.97 | 96 | 7.04e-71 | 7.04e-71 |
| replication | prog_02 | output | FR_e_sem | aligned | R2 | nonzero (negative) | -3.37 | 96 | 2.04e-168 | 4.07e-168 |
| replication | prog_02 | output | FR_e_sem | aligned | R3 | nonzero (negative) | -5.18 | 96 | 2.04e-144 | 2.04e-144 |
| replication | prog_02 | output | F_total | fixed_mean | R1 | nonzero (positive) | 0.709 | 96 | 9.44e-87 | 1.89e-86 |
| replication | prog_02 | output | F_total | fixed_mean | R5 | nonzero (positive) | 3.97 | 96 | 7.04e-71 | 7.04e-71 |
| replication | prog_02 | output | F_total | aligned | R2 | nonzero (negative) | -3.13 | 96 | 8.61e-163 | 1.72e-162 |
| replication | prog_02 | output | F_total | aligned | R3 | nonzero (negative) | -2.01 | 96 | 2.05e-142 | 2.05e-142 |
| replication | prog_02 | output | baseline | fixed_mean | R1 | nonzero (positive) | 0.709 | 96 | 9.44e-87 | 1.89e-86 |
| replication | prog_02 | output | baseline | fixed_mean | R5 | nonzero (positive) | 3.97 | 96 | 7.04e-71 | 7.04e-71 |
| replication | prog_02 | output | baseline | aligned | R2 | nonzero (negative) | -3.13 | 96 | 8.61e-163 | 1.72e-162 |
| replication | prog_02 | output | baseline | aligned | R3 | nonzero (negative) | -2.01 | 96 | 2.05e-142 | 2.05e-142 |
| replication | prog_05 | output | FR_e_sem | aligned | R2 | nonzero (negative) | -0.0721 | 96 | 6.58e-71 | 6.58e-71 |
| replication | prog_05 | output | FR_e_sem | aligned | R3 | nonzero (negative) | -0.268 | 96 | 9.54e-72 | 1.91e-71 |
| replication | prog_05 | output | F_total | aligned | R2 | nonzero (negative) | -0.0721 | 96 | 6.58e-71 | 6.58e-71 |
| replication | prog_05 | output | F_total | aligned | R3 | nonzero (negative) | -0.0892 | 96 | 8.97e-72 | 1.79e-71 |
| replication | prog_05 | output | baseline | aligned | R2 | nonzero (negative) | -0.0721 | 96 | 6.58e-71 | 6.58e-71 |
| replication | prog_05 | output | baseline | aligned | R3 | nonzero (negative) | -0.0892 | 96 | 8.97e-72 | 1.79e-71 |
| replication | prog_06 | output | FR_e_num | aligned | R2 | nonzero (positive) | 1.26e-05 | 96 | 7.75e-88 | 1.55e-87 |
| replication | prog_06 | output | FR_e_num | aligned | R3 | nonzero (positive) | 1.58e-05 | 96 | 7.92e-88 | 1.55e-87 |
| replication | prog_06 | output | F_total | aligned | R2 | nonzero (positive) | 1.26e-05 | 96 | 7.75e-88 | 1.55e-87 |
| replication | prog_06 | output | F_total | aligned | R3 | nonzero (positive) | 1.58e-05 | 96 | 7.92e-88 | 1.55e-87 |
| replication | prog_06 | output | baseline | aligned | R2 | nonzero (positive) | 1.26e-05 | 96 | 7.75e-88 | 1.55e-87 |
| replication | prog_06 | output | baseline | aligned | R3 | nonzero (positive) | 1.58e-05 | 96 | 7.92e-88 | 1.55e-87 |
| replication | prog_07 | output | FR_e_num | fixed_mean | R1 | nonzero (positive) | 0.00301 | 96 | 5.19e-176 | 1.03e-175 |
| replication | prog_07 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 0.00301 | 96 | 5.17e-176 | 1.03e-175 |
| replication | prog_07 | output | F_total | fixed_mean | R1 | nonzero (positive) | 0.00301 | 96 | 5.19e-176 | 1.03e-175 |
| replication | prog_07 | output | F_total | fixed_mean | R5 | nonzero (positive) | 0.00301 | 96 | 5.17e-176 | 1.03e-175 |
| replication | prog_07 | output | baseline | fixed_mean | R1 | nonzero (positive) | 0.00301 | 96 | 5.19e-176 | 1.03e-175 |
| replication | prog_07 | output | baseline | fixed_mean | R5 | nonzero (positive) | 0.00301 | 96 | 5.17e-176 | 1.03e-175 |
| replication | prog_13 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -62.2 | 96 | 3.36e-137 | 6.72e-137 |
| replication | prog_13 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 91.2 | 96 | 2.05e-112 | 2.05e-112 |
| replication | prog_13 | output | F_total | fixed_mean | R1 | nonzero (negative) | -62.2 | 96 | 3.36e-137 | 6.72e-137 |
| replication | prog_13 | output | F_total | fixed_mean | R5 | nonzero (positive) | 91.2 | 96 | 2.05e-112 | 2.05e-112 |
| replication | prog_13 | output | baseline | fixed_mean | R1 | nonzero (negative) | -62.2 | 96 | 3.36e-137 | 6.72e-137 |
| replication | prog_13 | output | baseline | fixed_mean | R5 | nonzero (positive) | 91.2 | 96 | 2.05e-112 | 2.05e-112 |
| replication | prog_16 | output | F_total | aligned | R2 | nonzero (positive) | 27.3 | 96 | 9.17e-32 | 9.17e-32 |
| replication | prog_16 | output | F_total | aligned | R3 | nonzero (positive) | 34.6 | 96 | 1.16e-40 | 2.33e-40 |
| replication | prog_16 | output | baseline | aligned | R2 | nonzero (positive) | 27.3 | 96 | 9.17e-32 | 9.17e-32 |
| replication | prog_16 | output | baseline | aligned | R3 | nonzero (positive) | 34.6 | 96 | 1.16e-40 | 2.33e-40 |
| replication | prog_18 | output | FR_e_num | aligned | R2 | nonzero (positive) | 1.72e-05 | 96 | 5.24e-50 | 5.24e-50 |
| replication | prog_18 | output | FR_e_num | aligned | R3 | nonzero (positive) | 2.14e-05 | 96 | 1.88e-51 | 3.75e-51 |
| replication | prog_18 | output | F_total | aligned | R2 | nonzero (positive) | 1.72e-05 | 96 | 5.24e-50 | 5.24e-50 |
| replication | prog_18 | output | F_total | aligned | R3 | nonzero (positive) | 2.14e-05 | 96 | 1.88e-51 | 3.75e-51 |
| replication | prog_18 | output | baseline | aligned | R2 | nonzero (positive) | 1.72e-05 | 96 | 5.24e-50 | 5.24e-50 |
| replication | prog_18 | output | baseline | aligned | R3 | nonzero (positive) | 2.14e-05 | 96 | 1.88e-51 | 3.75e-51 |
| replication | prog_22 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -60 | 96 | 2.47e-127 | 2.47e-127 |
| replication | prog_22 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 90.2 | 96 | 7.36e-128 | 1.47e-127 |
| replication | prog_22 | output | F_total | fixed_mean | R1 | nonzero (negative) | -60 | 96 | 2.47e-127 | 2.47e-127 |
| replication | prog_22 | output | F_total | fixed_mean | R5 | nonzero (positive) | 90.2 | 96 | 7.36e-128 | 1.47e-127 |
| replication | prog_22 | output | baseline | fixed_mean | R1 | nonzero (negative) | -60 | 96 | 2.47e-127 | 2.47e-127 |
| replication | prog_22 | output | baseline | fixed_mean | R5 | nonzero (positive) | 90.2 | 96 | 7.36e-128 | 1.47e-127 |
| replication | prog_23 | output | FR_e_num | fixed_mean | R1 | nonzero (negative) | -61.5 | 96 | 4.79e-117 | 9.58e-117 |
| replication | prog_23 | output | FR_e_num | fixed_mean | R5 | nonzero (positive) | 90.3 | 96 | 3.46e-103 | 3.46e-103 |
| replication | prog_23 | output | F_total | fixed_mean | R1 | nonzero (negative) | -61.5 | 96 | 4.79e-117 | 9.58e-117 |
| replication | prog_23 | output | F_total | fixed_mean | R5 | nonzero (positive) | 90.3 | 96 | 3.46e-103 | 3.46e-103 |
| replication | prog_23 | output | baseline | fixed_mean | R1 | nonzero (negative) | -61.5 | 96 | 4.79e-117 | 9.58e-117 |
| replication | prog_23 | output | baseline | fixed_mean | R5 | nonzero (positive) | 90.3 | 96 | 3.46e-103 | 3.46e-103 |
| replication | prog_25 | output | FR_e_num | aligned | R2 | nonzero (positive) | 8.97e-05 | 96 | 1.98e-52 | 3.97e-52 |
| replication | prog_25 | output | FR_e_num | aligned | R3 | nonzero (positive) | 0.00012 | 96 | 5.09e-51 | 5.09e-51 |
| replication | prog_25 | output | F_total | aligned | R2 | nonzero (positive) | 8.97e-05 | 96 | 1.97e-52 | 3.93e-52 |
| replication | prog_25 | output | F_total | aligned | R3 | nonzero (positive) | 0.00012 | 96 | 5.06e-51 | 5.06e-51 |
| replication | prog_25 | output | baseline | aligned | R2 | nonzero (positive) | 8.97e-05 | 96 | 1.97e-52 | 3.93e-52 |
| replication | prog_25 | output | baseline | aligned | R3 | nonzero (positive) | 0.00012 | 96 | 5.06e-51 | 5.06e-51 |
| replication | prog_30 | output | F_total | aligned | R2 | nonzero (positive) | 9.37 | 96 | 1.35e-119 | 1.35e-119 |
| replication | prog_30 | output | F_total | aligned | R3 | nonzero (positive) | 8.54 | 96 | 1.5e-124 | 3e-124 |
| replication | prog_30 | output | baseline | aligned | R2 | nonzero (positive) | 9.37 | 96 | 1.35e-119 | 1.35e-119 |
| replication | prog_30 | output | baseline | aligned | R3 | nonzero (positive) | 8.54 | 96 | 1.5e-124 | 3e-124 |
| main | prog_08 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -1.79e-07 | 64 | 3.48e-16 | 6.97e-16 |
| main | prog_08 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 1.21e-07 | 64 | 3.36e-09 | 3.36e-09 |
| main | prog_08 | variance | F_total | aligned | R2 | nonzero (negative) | -1.79e-07 | 64 | 3.48e-16 | 6.97e-16 |
| main | prog_08 | variance | F_total | aligned | R3 | nonzero (negative) | -1.78e-07 | 64 | 4.58e-16 | 6.97e-16 |
| main | prog_08 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -1.79e-07 | 64 | 3.48e-16 | 6.97e-16 |
| main | prog_08 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 1.21e-07 | 64 | 3.36e-09 | 3.36e-09 |
| main | prog_08 | variance | baseline | aligned | R2 | nonzero (negative) | -1.79e-07 | 64 | 3.48e-16 | 6.97e-16 |
| main | prog_08 | variance | baseline | aligned | R3 | nonzero (negative) | -1.78e-07 | 64 | 4.58e-16 | 6.97e-16 |
| main | prog_17 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_17 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_17 | variance | F_total | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_17 | variance | F_total | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_17 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_17 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_17 | variance | baseline | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_17 | variance | baseline | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_20 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_20 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_20 | variance | F_total | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_20 | variance | F_total | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_20 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_20 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_20 | variance | baseline | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_20 | variance | baseline | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_29 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_29 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_29 | variance | F_total | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_29 | variance | F_total | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_29 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_29 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.18e-07 | 64 | 1.86e-59 | 2.22e-59 |
| main | prog_29 | variance | baseline | aligned | R2 | nonzero (negative) | -9.31e-07 | 64 | 1.11e-59 | 2.22e-59 |
| main | prog_29 | variance | baseline | aligned | R3 | nonzero (negative) | -9.3e-07 | 64 | 1.16e-59 | 2.22e-59 |
| main | prog_32 | variance | F_total | fixed_mean | R1 | nonzero (positive) | 18 | 64 | 1.13e-148 | 2.23e-148 |
| main | prog_32 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 18 | 64 | 1.12e-148 | 2.23e-148 |
| main | prog_32 | variance | F_total | aligned | R2 | nonzero (positive) | 18 | 64 | 1.13e-148 | 2.18e-148 |
| main | prog_32 | variance | F_total | aligned | R3 | nonzero (positive) | 18 | 64 | 1.09e-148 | 2.18e-148 |
| main | prog_32 | variance | baseline | fixed_mean | R1 | nonzero (positive) | 18 | 64 | 1.13e-148 | 2.23e-148 |
| main | prog_32 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 18 | 64 | 1.12e-148 | 2.23e-148 |
| main | prog_32 | variance | baseline | aligned | R2 | nonzero (positive) | 18 | 64 | 1.13e-148 | 2.18e-148 |
| main | prog_32 | variance | baseline | aligned | R3 | nonzero (positive) | 18 | 64 | 1.09e-148 | 2.18e-148 |
| replication | prog_08 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -1.49e-07 | 96 | 1e-18 | 2e-18 |
| replication | prog_08 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 7.29e-08 | 96 | 1.35e-07 | 1.35e-07 |
| replication | prog_08 | variance | F_total | aligned | R2 | nonzero (negative) | -1.49e-07 | 96 | 1e-18 | 1.74e-18 |
| replication | prog_08 | variance | F_total | aligned | R3 | nonzero (negative) | -1.49e-07 | 96 | 8.69e-19 | 1.74e-18 |
| replication | prog_08 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -1.49e-07 | 96 | 1e-18 | 2e-18 |
| replication | prog_08 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 7.29e-08 | 96 | 1.35e-07 | 1.35e-07 |
| replication | prog_08 | variance | baseline | aligned | R2 | nonzero (negative) | -1.49e-07 | 96 | 1e-18 | 1.74e-18 |
| replication | prog_08 | variance | baseline | aligned | R3 | nonzero (negative) | -1.49e-07 | 96 | 8.69e-19 | 1.74e-18 |
| replication | prog_17 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_17 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_17 | variance | F_total | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_17 | variance | F_total | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_17 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_17 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_17 | variance | baseline | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_17 | variance | baseline | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_20 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_20 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_20 | variance | F_total | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_20 | variance | F_total | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_20 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_20 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_20 | variance | baseline | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_20 | variance | baseline | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_29 | variance | F_total | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_29 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_29 | variance | F_total | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_29 | variance | F_total | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_29 | variance | baseline | fixed_mean | R1 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2.71e-83 |
| replication | prog_29 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 9.2e-07 | 96 | 2.88e-83 | 2.88e-83 |
| replication | prog_29 | variance | baseline | aligned | R2 | nonzero (negative) | -9.38e-07 | 96 | 1.35e-83 | 2e-83 |
| replication | prog_29 | variance | baseline | aligned | R3 | nonzero (negative) | -9.39e-07 | 96 | 1e-83 | 2e-83 |
| replication | prog_32 | variance | F_total | fixed_mean | R1 | nonzero (positive) | 18 | 96 | 7.08e-224 | 1.39e-223 |
| replication | prog_32 | variance | F_total | fixed_mean | R5 | nonzero (positive) | 18 | 96 | 6.96e-224 | 1.39e-223 |
| replication | prog_32 | variance | F_total | aligned | R2 | nonzero (positive) | 18 | 96 | 7.08e-224 | 1.42e-223 |
| replication | prog_32 | variance | F_total | aligned | R3 | nonzero (positive) | 18 | 96 | 9.19e-224 | 1.42e-223 |
| replication | prog_32 | variance | baseline | fixed_mean | R1 | nonzero (positive) | 18 | 96 | 7.08e-224 | 1.39e-223 |
| replication | prog_32 | variance | baseline | fixed_mean | R5 | nonzero (positive) | 18 | 96 | 6.96e-224 | 1.39e-223 |
| replication | prog_32 | variance | baseline | aligned | R2 | nonzero (positive) | 18 | 96 | 7.08e-224 | 1.42e-223 |
| replication | prog_32 | variance | baseline | aligned | R3 | nonzero (positive) | 18 | 96 | 9.19e-224 | 1.42e-223 |

G 与 f 的包围不相交（e_sem 逐元素确证）的输出：[('prog_02', 'output', 196608), ('prog_05', 'output', 12582911), ('prog_16', 'output', 6144), ('prog_30', 'output', 6291456)]。这类元素要么是程序语义与任务规格的真实差异，要么是某一侧包围的反例，本轮不判定、不计分。

## 3. 复现轮（种子 96–191，开发块 0–31 固定坐标与学习方向）

两轮分别校正、不合并。主轮与复现轮摘要相同 255 个，不同 1 个：

| 程序 | 输出 | 对象 | 规则类 | 主轮 | 复现轮 |
| --- | --- | --- | --- | --- | --- |
| prog_30 | output | FR_e_sem | aligned | nonzero: R2 positive, R3 positive | not confirmed |

普通参照 + 相同统计（K − f64）与精确规格 K − f 的摘要相同 148/148 个格。T1 的 K − f 与 K − f64 来自补跑 D1（DEVIATIONS.md）。

## 4. 一致性检查

- 模式 A 与 B（主轮，非原子程序）：K 逐位、参照质量与 K−G 摘要不一致的输出 4 个：[('prog_01', 'output'), ('prog_13', 'output'), ('prog_22', 'output'), ('prog_23', 'output')]。原子程序 K 随执行次序变化，单列。
- 冻结入口（`acceptance_run.run_case` → `measure.run` + 基线）对照 prog_01：{"output": {"entry_status": "evaluated", "adapter_status": "evaluated", "reference_equal": true, "e_num_class_statistics_equal": false, "baseline_class_statistics_equal": false, "semantic_equal": true}}
- 冻结入口（`acceptance_run.run_case` → `measure.run` + 基线）对照 prog_03：{"output": {"entry_status": "evaluated", "adapter_status": "evaluated", "reference_equal": true, "e_num_class_statistics_equal": true, "baseline_class_statistics_equal": true, "semantic_equal": true}}

## 5. E / F / P

| 轮 | 程序 | 输出 | E：K vs f64 相对 RMS | E：max/ulp | F：区间排除 0 的比例 | F：均值/ulp |
| --- | --- | --- | --- | --- | --- | --- |
| main | prog_01 | output | 0.294 | 8.62e+12 | 1 | 2.34e+07 |
| main | prog_02 | output | 0.0945 | 1.37e+12 | 0.549 | -1.11e+06 |
| main | prog_03 | output | 1.06e-07 | 4.76e+03 | 1 | -0.0873 |
| main | prog_04 | output | 5.07e-08 | 2.86 | 1 | 4.65e-05 |
| main | prog_05 | output | 0.0221 | 1.45e+06 | 1 | 17.6 |
| main | prog_06 | output | 6.68e-08 | 3.22 | 1 | -0.000203 |
| main | prog_07 | output | 5.89e-06 | 1.98e+05 | 1 | -521 |
| main | prog_09 | output | 2.54e-08 | 0.5 | 0.536 | 0.000115 |
| main | prog_10 | output | 2.54e-08 | 0.5 | 0.536 | 0.000115 |
| main | prog_11 | output | 1.01e-07 | 3.46e+03 | 1 | -0.557 |
| main | prog_12 | output | 5.07e-08 | 2.86 | 1 | 4.65e-05 |
| main | prog_13 | output | 0.285 | 8.55e+12 | 1 | 2.19e+07 |
| main | prog_14 | output | 2.54e-08 | 0.5 | 0.536 | 0.000115 |
| main | prog_15 | output | 9.09e-08 | 1.63e+07 | 0.999 | -2.4 |
| main | prog_16 | output | 0.249 | 1.23e+10 | 1 | -4.18e+06 |
| main | prog_18 | output | 1.11e-07 | 3.14e+03 | 1 | -1.18 |
| main | prog_19 | output | 1.01e-07 | 3.71e+03 | 1 | -0.801 |
| main | prog_21 | output | 1.06e-07 | 4.76e+03 | 1 | -0.0873 |
| main | prog_22 | output | 0.289 | 8.55e+12 | 1 | 2.19e+07 |
| main | prog_23 | output | 0.288 | 8.62e+12 | 1 | 2.28e+07 |
| main | prog_24 | output | 9.09e-08 | 1.63e+07 | 0.999 | -2.4 |
| main | prog_25 | output | 9.87e-08 | 1.91e+07 | 0.999 | -5.21 |
| main | prog_27 | output | 1.01e-07 | 3.71e+03 | 1 | -0.713 |
| main | prog_28 | output | 9.09e-08 | 1.63e+07 | 0.999 | -2.4 |
| main | prog_30 | output | 0.0438 | 5.64e+12 | 1 | -1.2e+06 |
| main | prog_31 | output | 1.06e-07 | 4.76e+03 | 1 | -0.0873 |
| main | prog_33 | output | 2.54e-08 | 0.5 | 0.536 | 0.000115 |
| main | prog_08 | mean (D1) | 4.5e-08 | 1.87 | 1 | -0.00659 |
| main | prog_08 | variance (D1) | 5.69e-08 | 1.85 | 1 | 0.0914 |
| main | prog_17 | mean (D1) | 4.5e-08 | 1.87 | 1 | -0.00659 |
| main | prog_17 | variance (D1) | 7.63e-08 | 2.66 | 1 | 0.491 |
| main | prog_20 | mean (D1) | 4.5e-08 | 1.87 | 1 | -0.00659 |
| main | prog_20 | variance (D1) | 7.63e-08 | 2.66 | 1 | 0.491 |
| main | prog_29 | mean (D1) | 4.5e-08 | 1.87 | 1 | -0.00659 |
| main | prog_29 | variance (D1) | 7.63e-08 | 2.66 | 1 | 0.491 |
| main | prog_32 | mean (D1) | 4.5e-08 | 1.87 | 1 | -0.00659 |
| main | prog_32 | variance (D1) | 1 | 1.67e+07 | 1 | -9.44e+06 |

F 的区间排除 0 是样本差异证据，不自动是契约违反或均值非零（协议 §2）。P（性质关系）一律 `not_scored`：

| 程序 | 结构 | 关系 | max | / ulp(尺度) | 说明 |
| --- | --- | --- | --- | --- | --- |
| prog_01 | T5 | row sum of y (exact target 0; float centering does | 706 | 5.78e+06 | 原始残差 |
| prog_04 | T6 | sum(y^2) - D * m2 / (m2 + eps), m2 = mean(x^2) (ze | 0.000595 | 4.87 | 原始残差 |
| prog_05 | T6 | sum(y^2) - D * m2 / (m2 + eps), m2 = mean(x^2) (ze | 368 | 3.02e+06 | 原始残差 |
| prog_06 | T6 | sum(y^2) - D * m2 / (m2 + eps), m2 = mean(x^2) (ze | 0.00073 | 5.98 | 原始残差 |
| prog_11 | T7 | 8 次执行间逐位不同的元素比例 0.831 | spread 2.05e+03 ulp | 均值 5.39 | 次序依赖证据 |
| prog_12 | T6 | sum(y^2) - D * m2 / (m2 + eps), m2 = mean(x^2) (ze | 0.000595 | 4.87 | 原始残差 |
| prog_13 | T5 | row sum of y (exact target 0; float centering does | 713 | 5.84e+06 | 原始残差 |
| prog_15 | T2 | y[j] - y[j-1] - x[j] * s (exact target 0; float di | 1.28e-05 | 53.9 | 原始残差 |
| prog_18 | T7 | 8 次执行间逐位不同的元素比例 0.834 | spread 2.56e+03 ulp | 均值 5.32 | 次序依赖证据 |
| prog_19 | T7 | 8 次执行间逐位不同的元素比例 0.829 | spread 3.07e+03 ulp | 均值 5.69 | 次序依赖证据 |
| prog_22 | T5 | row sum of y (exact target 0; float centering does | 575 | 4.71e+06 | 原始残差 |
| prog_23 | T5 | row sum of y (exact target 0; float centering does | 702 | 5.75e+06 | 原始残差 |
| prog_24 | T2 | y[j] - y[j-1] - x[j] * s (exact target 0; float di | 1.28e-05 | 53.9 | 原始残差 |
| prog_25 | T2 | y[j] - y[j-1] - x[j] * s (exact target 0; float di | 1.18e-05 | 49.3 | 原始残差 |
| prog_27 | T7 | 8 次执行间逐位不同的元素比例 0.807 | spread 2.05e+03 ulp | 均值 5.12 | 次序依赖证据 |
| prog_28 | T2 | y[j] - y[j-1] - x[j] * s (exact target 0; float di | 1.28e-05 | 53.9 | 原始残差 |
| prog_30 | T2 | y[j] - y[j-1] - x[j] * s (exact target 0; float di | 5.42 | 3.11e+07 | 原始残差 |

重命名关系（确定性程序，主轮 96 个单位 K 全部逐位相同的同族程序组）：[('T3', ['prog_03', 'prog_21', 'prog_31']), ('T6', ['prog_04', 'prog_12']), ('T4', ['prog_09', 'prog_10', 'prog_14', 'prog_33']), ('T2', ['prog_15', 'prog_24', 'prog_28'])]。

## 6. 失败类别与成本

- 主轮模式 B 各输出的失败类别计数：{'semantics missing': 10, 'enclosure too wide': 5}。
- 全部作业墙钟秒数之和 126426 s；逐作业的分阶段时间在 summary.json `cost`。

## 7. 没有做的事

- 正式盲评分数（等出题方重新封存）；等价判断（未事先声明 δ）；定位（预算 0，未尝试）；边界项 prog_26（未批准隔离运行）；调用级完整（协议 §1）。
