# Reference DSL v2-rc3：官方范围优先的复审与扩展设计

**这是设计修订包，不是生产工具新版。** 基线为用户上传的`reference_dsl_v2_rc2_20261009.zip`；原包原样保存在`history/input_rc2.zip`。

本版不把DSL缩到kernel-analyzer已经实现的229项旧名字表、165项旧TTIR子集或一台GPU。官方Triton、Gluon、NVIDIA/AMD及相关低层/运行时能力进入目标面；点值、集合和环境参数化语义各自规定可兑现的保证。

## 建议阅读顺序

1. [复审结论与更正](REVIEW_REPORT.md)
2. [00 官方目标范围](design/00_official_scope.md)
3. [01 已有仓库怎样迁移](design/01_repository_basis.md)
4. [02 语言、效果与保证](design/02_language_and_guarantees.md)
5. [03 官方清单与覆盖验收](design/03_coverage_and_evaluation.md)
6. [04 下一轮实施任务](design/04_implementation_plan.md)

[Codex执行摘要](CODEX_TASK.md)可用于安排下一轮工作；[全部旧检查点迁移](COVERAGE_CHECKLIST.md)防止在扩张时丢失原研究的统计/训练/评价纪律。

## 本包真正完成了什么

- 上传rc2的11项测试重跑通过，原18个哈希条目核对通过。
- rc2四份主文档、规则schema、例子与125项检查点复审。
- 固定官方主线提交`e50b186e8bd2d16ae3f564311ef868aa3bd45a2d`的入口来源核对，结合官方文档扩展61项能力族。
- 148项classic模块公共导出快照、旧229项历史路由、125项保留或修正的追踪记录。
- 四个拟议规则实例（reduce、dot、CAS、descriptor），按模式与条件给出不同保证。
- 两个只读辅助工具：规则形状检查、官方标准化清单差集核对。
- 35项新增CPU设计/边界/核账测试通过。详细记录见[验证范围](checks/validation_scope.json)。

**没有完成**官方所有TableGen实例化、本地原生注册表与全部外部函数枚举，未运行GPU、未修改生产仓库、未认证新的全域定理、未读取封存答案。
61项是需求族，148项是一个模块的导出名字，125项是文档追踪点。这三个数都不是工具的新支持率。

## 复跑

Python 3.10+；规则验证需要`jsonschema`。无Triton/GPU/网络依赖。

```bash
sha256sum -c SHA256SUMS
python -m unittest discover -s checks -v
python tools/validate_contracts.py examples/*_rule_contract.json
```

清单核账工具需要真实的官方源码/构建/观察输入；本包没有用伪造的“全量清单”跑出通过。用法见[tools/README.md](tools/README.md)。

## 发布纪律

不要把这个包覆盖到生产源码后宣称已支持；先冻结官方profile和目标契约，再按工作包实现与独立验收。
历史结果不改、不重新选择阈值和答案；新增官方功能作为独立版本的实现/验证进度报告。

“官方能表达什么”“参照模型定义了什么”“已证明什么”“代码做到什么”“设备验证到哪里”五列独立。
