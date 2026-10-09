# 文档导航

**当前版本、入口、能力范围、状态与待决事项只在仓库根目录的 [`CURRENT.json`](../CURRENT.json) 维护。** 本页只做导航，不复制结论；
文档之间有矛盾时，以 CURRENT.json 指向的机器结果与有范围的台账为准，不按文档新旧自动采用更强的说法。

## 当前（由 CURRENT.json 指向）

- 验收：[结构验收集 v1.1-rc1 冻结规则测量（未计分）](acceptance/structure_v1_1_rc1_status_20261008.md)。
- 下一轮：[参照 DSL v2 任务书（执行方修订稿，待审阅方核对）](taskbook_dsl_v2_executor_revision_20261009.md)；设计规范 reference_dsl_v2_rc2_20261009（SHA-256 7c7fa0c1…6fb4）。
- DSL v2（dsl-v2 分支，工具 4.0，未发布）：依据 rc3 正式版（SHA-256 8050903f…6faf）。增量登记
  [1](dsl_v2/increment_01.md)、[2](dsl_v2/increment_02.md)、[3](dsl_v2/increment_03.md)、[4](dsl_v2/increment_04.md)、
  [5](dsl_v2/increment_05.md)、[6](dsl_v2/increment_06.md)、[7](dsl_v2/increment_07.md)、[8](dsl_v2/increment_08.md)、
  [9](dsl_v2/increment_09.md)、[10](dsl_v2/increment_10.md)、[11](dsl_v2/increment_11.md)、[12](dsl_v2/increment_12.md)、[13](dsl_v2/increment_13.md)、[14](dsl_v2/increment_14.md)；
  结果 [1](dsl_v2/increment_01_status.md)、[2–3](dsl_v2/increment_02_03_status.md)、[4–5](dsl_v2/increment_04_05_status.md)、
  [6](dsl_v2/increment_06_status.md)、[7–8](dsl_v2/increment_07_08_status.md)、[9](dsl_v2/increment_09_status.md)、[10](dsl_v2/increment_10_status.md)、[11](dsl_v2/increment_11_status.md)、[12](dsl_v2/increment_12_status.md)、[13](dsl_v2/increment_13_status.md)、[14](dsl_v2/increment_14_status.md)；[规则注册表（3.6.0 回归 profile）](dsl_v2/rule_registry_report.md)。
  都是开发证据，不是盲测成绩。
- 状态：[通用能力轮](general/status_20261008.md)、[收束轮](closure_status_20261008.md)、[两轮联合审计](audit_two_rounds_20261008.md)
  （C 节是待审阅方决定的事项）、[成绩核账（三种状态）](status_ledger_20261006.md)。
- 协议：[通用能力轮](protocol_general_capability_v1_20261008.md)、[收束轮 v3](protocol_closure_v3_20261008.md)、
  [本质错误一轮第一阶段](protocol_essential_bugs_20261007.md)、[第二阶段](protocol_essential_bugs_phase2_20261007.md)。
- 工具：[检测器 / 工具版本变更](detector_changelog.md)、[规则注册表报告](general/rule_registry_report.md)、
  [参照精度与成本家族表](general/quality_table.md)、[检测校准](general/calibration_table.md)、[失败分类](general/failure_table.md)、
  [符号对照](general/symbol_table.md)、[证据 P05](general/evidence_P05.md)。
- 接入：[接入指南](binding_guide.md)、示例 `examples/general/`（统一入口）与 `examples/bindings/`（`check` 绑定文件）、
  [环境](environments.md)。
- 规格与结论：独立规格在 `specs/`（第一阶段 v0.4、第二阶段包 v0.2）；[规格问题](spec_issues_phase2.md)；
  [2b 记录第二版](essential_bugs_phase2b_record_v2_20261008.md)；[家族四组与四栏汇总](closure_family_tables_20261008.md)；
  [三类组合清单](composition_checklists_20261008.md)；[问题登记](../bugs/README.md)。
- 案例与主张：[主张账本](claims.md)、[全部案例结论](root_cause_closure_current.md)（由脚本生成）。

## 历史

- [阶段性总结（2026-10-02）](stage_summary_20261002.md)：历史基线；后续协议与审计已更正其中部分说法，以 CURRENT.json 为准。
- 整理前的入口：[旧 README](history/README_before_20261008.md)、[旧文档索引](history/docs_README_before_20261008.md)
  （列出当时的专题页、职责边界与历次删除记录）。
- 早期评价与专题（benchmark、外部受控集合、陌生组合子集、逐层表、根因推导等）仍在 `docs/` 原路径，冻结声明中的路径不改；
  从旧文档索引进入。
- `archive/`：已失效的旧研究实现与材料。
