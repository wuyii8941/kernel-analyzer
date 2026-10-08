# 收束包 v2.0 任务书：计划项状态表（2026-10-08）

状态只取协议第二版第 153 行的七种：已执行且可裁决、仅部分方法可执行、不适用、未建立、环境不可用、超预算、未运行。「结果」一栏写通过标准
是否达到，不用状态代替结果。

| # | 计划项（任务书节） | 状态 | 结果 | 依据 |
|---|---|---|---|---|
| 1 | A1 绑定修正与三个回归场景（§2.1） | 已执行且可裁决 | 回归通过；之后才开始新的捕获 | `tests/test_output_binding.py`；`docs/detector_changelog.md` 2.2 |
| 2 | A2、A3（§2.1） | 已执行且可裁决 | 协议 v3 第 1 节：28 项回归通过 | `docs/protocol_closure_v3_20261008.md` §1 |
| 3 | 契约 v2 生效，未覆盖者标「契约外，待审阅」（§2.2） | 已执行且可裁决 | 16 个条件契约外（归约 5、gather 6、Adafactor 2、OneCycle 2、clip_value 1） | `results/closure/contract_classification.json` |
| 4 | 协议 v3：精度不变性判据、S₀ / N₀ / n_min（§2.3） | 已执行且可裁决 | S₀ = 2、N₀ = 64、n_min = 16，校准依据写入；偏离 10 条 | `docs/protocol_closure_v3_20261008.md` |
| 5 | 第 1 行：chain_table.json，校验器 0 问题（§3） | 已执行且可裁决 | 0 problems | `results/closure/chain_table.json`、`chain_table_validation.txt` |
| 6 | 第 1 行：每条降级路径有实际触发记录（§3） | 已执行且可裁决 | 12 / 12 有记录 | `results/closure/degrade/`；D_m 用盲测 v2 第三阶段记录 |
| 7 | 第 1 行：规则 → 函数表（§3） | 已执行且可裁决 | 引用测试 48 项（含参数化）通过 | `docs/rules_to_code_20261008.md`、`chain_table_tests_run.txt` |
| 8 | 第 2 行：九个模式的复审（§4） | 已执行且可裁决 | 33 行，缺漏 0 | `docs/readjudication_20261008.md` |
| 9 | 第 2 行：精度不变性比值代替条件数说法；均值作用由统计层给出（§4） | 已执行且可裁决 | 全部家族「无语义元素」，例外 2 处已归属（EMB-A1 第 2 栏；O-D2 待审阅） | `results/closure/f_eval/*.json` |
| 10 | 第 2 行：G7 GELU 保留为真实数值作用；历史文件不改、出新版（§4） | 已执行且可裁决 | 独立 seed 上复现；契约外，待审阅 | `docs/essential_bugs_phase2b_record_v2_20261008.md` |
| 11 | 第 3 行：契约 v2 逐条件五类标注（§5） | 已执行且可裁决 | 见第 3 项；待审阅条款 4 处 | `results/closure/contract_classification.json` |
| 12 | 第 3 行：七项文档核对（§5） | 已执行且可裁决 | 3 个条款停止裁决（SCH-A1、BASE-A1、E-D2），2 处契约标注请审阅 | `docs/doc_check_seven_items_20261008.md` |
| 13 | 组合 C1 纯计算（§6） | 已执行且可裁决 | 见清单 | `docs/composition_checklists_20261008.md` |
| 14 | 组合 C2 保存值 / 分支（§6） | 已执行且可裁决 | 见清单 | 同上 |
| 15 | 组合 C3 状态更新（§6） | 已执行且可裁决 | 见清单 | 同上 |
| 16 | 三份清单合并覆盖全部降级路径（§6） | 已执行且可裁决 | **未完全达到：9 / 12**；κ = conditional、D_m 无区间扩展、无可比配对未在组合中出现 | 同上「合并后的降级覆盖」 |
| 17 | E（§7） | 已执行且可裁决 | 15 个家族（2b 数据） | `docs/closure_family_tables_20261008.md` |
| 18 | F（§7） | 已执行且可裁决 | 有规格的 12 个家族 + training_program（2b）；checkpoint 不适用（无整程序 f） | `results/closure/f_eval/` |
| 19 | 精度不变性的 float64 补跑（§7） | 已执行且可裁决 | attention、optimizers、packing、rope、moe 的同设备配对 | `docs/protocol_closure_v3_20261008.md` §10 第 3 条 |
| 20 | P：规格 prop_*、gradcheck、jvp 对 VJP、D1–D3（§7） | 仅部分方法可执行 | prop_* 的候选级对应项多数已在 2b 预注册中；gradcheck / jvp 在 9 个家族运行；前向 AD 不支持的候选记「不支持」；D2、D3 不适用 | `results/closure/p_extra/` |
| 21 | FR 模式 B（§7） | 仅部分方法可执行 | 运行 10 个家族的 Triton 候选；matmul 与 Inductor 注意力的主输出由 cuBLAS 写出（未建立）；rope 的 Triton 候选只有库候选（liger 环境，未运行）；schedulers、clip_amp、training_program 无 Triton kernel（不适用）；checkpoint 无 f | `results/closure/fr_modeB/` |
| 22 | 声明约定登记（§7） | 已执行且可裁决 | 15 个家族 | `results/closure/candidate_conventions.json` |
| 23 | G3 覆盖（计划 / 已执行 / 未覆盖）、G5 序列与 A → B → A（§7） | 已执行且可裁决 | 每个家族列出；A → B → A：2b 60 条序列 0 违反，C3 重载逐位相同 | `docs/closure_family_tables_20261008.md`；`results/closure/compositions/C3_main.json` |
| 24 | 数值作用流：20 个程序，独立确认输入（§7） | 已执行且可裁决 | 20 / 20 建立参照；9 / 20 至少一条规则平均作用非零 | 协议 v3 附录 A（84fe39e）；`results/closure/numerical_stream.json` |
| 25 | 新发现的处理（§8） | 已执行且可裁决 | 规格问题 1 个（交审阅方）；工具问题 1 个（检测器 2.3 已修正）；无新的上游问题、无新草稿 | `docs/spec_issues_phase2.md`；`docs/detector_changelog.md` |
| 26 | 成本表（§10） | 已执行且可裁决 | 各家族 F / P 补充 / FR 秒数 | `docs/closure_family_tables_20261008.md` |

## 交审阅方决定的事项（§12）

1. **契约外的条件**：16 个（第 3 项）；组合 C2（契约 v2 未列出组合）。
2. **编译期常数造成的 K_R ≠ f_r**（Inductor GELU 常数、flex 的 RCP_LN2、BatchNorm 折叠的 0.9 与 n/(n−1)、按 float32 持有的超参数）：差异已确证，
   契约未规定常数精度——是否属于「契约允许的数值行为」。
3. **normalization float32 huge_offset 的超出**（数值，精度不变性）：契约未规定精度要求。
4. **文档差异**：SCH-A1（T_cur ≥ T_max）、BASE-A1（var / std 自由度 ≤ 0）、E-D2（max_norm 作用范围）；契约标注：HF 调度 total > warmup、matmul
   批维广播（「规格未覆盖」而不是「不合法」）。
5. **规格问题**：SPEC-ISSUE-1（O-D2 的 maximize 位置）。
6. **组合覆盖缺口**：3 条降级路径未在组合中出现，是否接受第 1 行的记录或指定组合。
7. **上游提交**：既有草稿（B020–B023、B015 的评论、文档问题）由用户决定。
