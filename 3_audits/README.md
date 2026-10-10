# 3 完整审计

## 1. 最新外部审计：DSL v2 复核包（基线 1aee15e，2026-10-09）

收到的材料原样保存在 [received/dsl_v2_audit_recheck_1aee15e_20261009/](received/dsl_v2_audit_recheck_1aee15e_20261009/START_HERE.md)
（复核报告 RECHECK_REPORT.md、任务书 CODEX_NEXT_TASK.md、工作包账 WORKPACKAGES.md、材料账 MATERIALS_LEDGER.md、发现账 review_ledger.json、
上一版审计包 previous_package/ 与复现脚本；未收入包内的 rc3 原件副本，见 1_experiments/dsl_v2/design_rc3，以及与上一版结果相同的重跑输出）。
审计范围是关键路径抽查，不是全面认证；它的复现多数在摘录函数上，原生仓库复现与修复由执行方负责。

审计要求先交修复版，再继续增量 14。该包到达前，增量 14、15 已经完成（基线 1aee15e 是增量 13 的结果提交）。修复版按任务书交付在
[fix_1aee15e/](fix_1aee15e/README.md)：每项的修复前失败测试、修复后测试、普通对照、调用路径、原始日志、SHA 与环境，以及三张账（七项处理表、
测试汇总、提交逐项表与影响、W0–W7 差距表）。修复提交 `5fe8e56`、`6f0ef38`，欠账补齐 `c3dd596`–`503a19e`（含重算捕获时发现的 D6–D11）。

| 编号 | 发现（优先级） | 涉及代码 | 当前状态 |
| --- | --- | --- | --- |
| F01 | 整数精确域错误（P0）：64 位无符号比较 / 除法 / 余数 / 向上取整除 / 逻辑右移 / 最值，有符号极值的宿主溢出；CAS | `_int_op`、`_cmpi`、`_op_atomic_cas` | **已修复**：审计原生复核 1,748 例 150 → 0 错；CAS 在增量 15 已修 |
| F02 | scaled dot 只取输入区间下端，并丢掉 scale 的条件标记（P0） | `_scaled_dot`、`_scaled_upcast` | **已修复**（规则级 + 完整解释器级；sm_100 / gfx9xx 设备未验证） |
| F03 | CAS 正序 / 逆序一致不能认证全部调度；scan 两种括号同理（P0，对无条件结论） | `_cas_reverse_order`、`_check_scan_bracketing`、完整率汇总 | **已修复**：证明状态轴「complete_under_premise」不进无条件完整率与统计；z3 证书（scan 结合性、CAS 自旋锁临界区两两可交换）成立时无条件 |
| F04 | 「值在输入里出现过」「全零」被当作复制来源证明（P0，对调用级结论） | `torch_intermediates`、measure 的范围 | **已修复**：只认声明输入自身的未改写存储或生产者记录（不计入测量的轨迹运行，逐启动对齐）中的复制 / 精确常数；值相等不算；原子读取目标改记为读取 |
| F05 | Gluon 统一入口：`check.run` 硬读 `asm["ttir"]`（P1） | `check.run` | **已修复**：一处 IR 选择，GPU 端到端（Gluon 经 `measure.run`）通过 |
| F06 | 声明摘要只覆盖 call / inputs / compare / budget（P1） | `measure.expand` | **已修复**：摘要覆盖整个展开声明与源码哈希；alpha、δ、重复次数、M 进入统计层；未达分辨率时按需提高工作精度并报告结局 |
| F07 | 执行有效性未建立时统计仍被放行（P0，对受影响的统计结论） | `_execution_status` | **已修复**：未知有效性停统计；输入内平均只用于浮点原子写入的元素；竞争原因码按生产者结构化 |

修复不等于 W0–W7 完成；仍开放的事项与交审阅方的决定见 [fix_1aee15e/README.md](fix_1aee15e/README.md) 第 4、5 节。

**执行方自查发现（整理仓库时，2026-10-09）**：增量 10、12、13 的测试 fixture（`2_tool/tests/data/gluon_ttgir`、`amd_ttgir/`、`nvidia_ttgir/`，
共 32 个 .ttir / .ttgir 文件）因 `.gitignore` 忽略这两种扩展名而从未提交；推送的这些提交在干净克隆上相应测试会失败。整理时已补入
`2_tool/tests/data/` 并修改 `.gitignore`。

## 1A. 第一批共同语义内核（基线 dsl-v2@3532066，2026-10-10）

按任务书第一批：来源 / 内存效果的三类问题（跨输入摘要误配、原地算术后字节未变、重叠 view 部分写入当成整块常数）、精度控制器（不再由
「达标比例没变」推断「加精度无效」，三级求和反例从统一入口走到第 3 级）、安装包的仓库路径依赖，以及 bias 查询的固定、有界路线的族保证与
两个判定轴；可复用的组合规则 M1–M5、T1、P1–P3、Q1–Q3、K1。执行方自查另发现 D12（生产者追踪使编译函数此后按 eager 执行）。
逐项账、修复前失败测试与验收见 [batch1_semantic_core/](batch1_semantic_core/README.md)：运行之前冻结的验收 24 项全部成立，公开 v1.1 回归 28/28 不变；记有一处执行偏差（4 个竞争程序被误在 GPU 上运行，结果不入记录，运行脚本已加默认跳过）。

## 2. 此前审计中仍待审阅方决定的事项

来自 [audit_two_rounds_20261008.md](audit_two_rounds_20261008.md)（收束轮与通用能力轮的联合审计）C 节，与
[spec_issues_phase2.md](spec_issues_phase2.md)：

1. 保证链表中 κ = conditional 的适用范围（check.run 中结构上不出现，只来自 pin_loads）。
2. 编译期常数造成的 K_R ≠ f_r 是否属于契约允许的数值行为。
3. 规格问题 SPEC-ISSUE-1；文档差异 SCH-A1、BASE-A1、E-D2；契约标注（HF total > warmup、matmul 广播）；契约外 16 个条件与组合 C1、C2。
4. 注册表是否要求逐条专属测试；scan 通用折叠的结合性前提。
5. 第 5 项验收集与上游提交（草稿在 4_bugs_cases/bugs/upstream_drafts，由用户本人提交）。

复核包撤回了上一版对 SPEC-ISSUE-1 的裁决用法，恢复「逐项对照后决定」。

## 3. 外部材料账（复核包 MATERIALS_LEDGER）

| 编号 | 材料 | 责任方 | 状态 |
| --- | --- | --- | --- |
| M01 | 29 个提交（截至 1aee15e）逐项清单与影响表 | 执行方导出，审阅方审核 | 执行方已交：dsl-v2 相对 main 的 32 个提交（[fix_1aee15e/commit_table.md](fix_1aee15e/commit_table.md)）；待审阅方审核 |
| M02 | 七项发现的原生复现与修复前后包 | 执行方 | 执行方已交（[fix_1aee15e/](fix_1aee15e/README.md)）；待审阅方复核 |
| M03 | W0 人工核账底稿 | 执行方提取，审阅方签意见 | W0 对账为 READY_FOR_MANUAL_REVIEW，未签 |
| M04 | W1 条目的独立核验表 | 审阅方独立核验 | 未做（执行方侧 430/430 有三类测试与触发证据） |
| M05 | 统一入口与全量回归包 | 执行方 | 部分：sm_86 上 classic / Gluon / 浮点原子经 `measure.run` 端到端，全量回归计数与日志见 fix_1aee15e 第 2 节；其余目标的设备运行属 M11 |
| M06 | 独立现实语料 B、保留结构集 | 审阅方或独立提供方 | 未交 |
| M07 | 独立生成器与严格答案程序 | 独立方 | 未交 |
| M08 | DSL v2 新盲测包 | 审阅方 / 出题方 | 未交 |
| M09 | δ、M 与统计族的事前声明 | 使用 / 审阅方 | 未交 |
| M10 | 结构验收 v1.1 的两份原始答案及承诺链 | 原出题方直接交付 | 未交 |
| M11 | sm_90 / sm_100 / gfx942 / gfx950 设备报告 | 硬件拥有方 | 未交 |
| M12 | 第 2 节各争议的逐项裁决 | 审阅方，必要时用户 | 未交 |
