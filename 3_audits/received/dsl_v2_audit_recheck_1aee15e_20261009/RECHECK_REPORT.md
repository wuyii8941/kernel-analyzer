# 审计包复核与补充（2026-10-09）

基线仍为 `1aee15e9df7a45f434699b7016056e6650fdae99`。本轮通过GitHub连接器重新读取分支头，结果未变化（turn849file0）。未访问或改写旧封存答案，未修改生产仓库。

## 1. 旧包是不是漏了内容？

不是漏装。原ZIP含17个文件，解压内容合计76,756字节；两份Markdown之外还有4个Python脚本、JSON证据、日志、来源清单和校验表。ZIP CRC无错误，SHA256SUMS中的16条全部相符（校验表本身不自哈希）。

两套旧脚本复制到独立重跑目录运行，退出码均为0，两份结果JSON与旧包完全一致。这里0表示“反例按登记复现”，不是“工具通过正确性验收”。原始包未被覆盖。

**但材料确实不够完整验收。** 它是关键路径抽查包，不是29提交逐项评审，不是421规则独立逐项审核，也没有真实checkout上的原生运行、完整生产解释器或GPU入口结果。目录有17份文件不改变这一证据范围。

## 2. 这次新补的实际检查

### F01补测：整数问题不是仅两个特例

重新读取 `_int_op` 的当前源代码（turn854file0），把该函数按原文转录，以独立Python大整数计算预期答案。对8/16/32/64位的除法、余数、向上整除、无符号最值和逻辑右移，以及signed最小值附近的合法除法做参数化核对。

1,748个规则级边界用例中，150个与独立答案不符，全部发生在64位路径，且均返回ST_OK。涉及9个操作名。这些数是定向边界测试的计数，**不是官方语料故障率，也不是150个独立bug**。

| 操作 | 错例数 |
|---|---:|
| i64/divui | 24 |
| i64/remui | 20 |
| i64/ceildivui | 22 |
| i64/shrui | 12 |
| i64/maxui | 24 |
| i64/minui | 24 |
| i64/divsi | 10 |
| i64/remsi | 8 |
| i64/ceildivsi | 6 |

例如unsigned `max(0,2^63)`应返回2^63，却返回0；unsigned `1 // 2^63`应返回0，实际位模式为2^64−1。signed极值还有宿主abs/取负溢出问题。因此修复应覆盖整数通用路径，不只补两个CAS输入。

### F06新增：声明摘要没有绑定完整测量条件

`measure.expand`只用call/inputs/compare/budget四项生成declaration_sha256（turn855file0）。
实际运行摘录函数：分别改变alpha、units、resolution、magnitude_bound，展开后的字段都变了，摘要却保持相同，4/4对照复现。

这证明**这个字段不能单独证明完整测量方案未变**。本次没有核查全部外层冻结流程，所以不说整个历史承诺链已经被绕过。修复是给完整规范化的有效声明生成摘要，另锁源码和环境；若外层已有完整摘要，须提供对应证据并修正文档角色。

### F07新增：执行有效性未知仍可能放行统计

重新读 `_execution_status`（turn857file0）。函数收集unknown_validity_findings，但选择statistics时没有用它阻断。

四个构造记录对照中：

- 有unknown-validity且重复相同，返回per launch，而不是withheld。
- 有unknown-validity、重复不同且调用含浮点原子，返回within-input mean。
- 已知race对照会withheld。
- 普通稳定对照返回per launch。

前两项属于统计准入缺口。修复时需要核对所有真实生产者的原因码，使用结构化有效性状态；不要只针对本包的字符串打补丁。本次是生产判定函数的摘录运行与构造元数据，不是GPU中已经观察到这类路径的声明。

## 3. 对上一版需要纠正或补全的地方

**证据等级。** 摘录函数、独立数学反例、静态检查、完整解释器、GPU端到端是不同等级。旧包提供前几类；必须由真实仓库回归补强，不能默认全部已达到产品级复现。AST核对选项存在，但本环境没有完整checkout，本轮也未完成与完整源文件的可执行AST对账。

**材料身份。** 问题审计包不是独立语料B、独立程序生成器、新盲测、设备证书或原出题方交付；这些不会因增加几个JSON就齐备。

**CAS报告。** 当前规则表已经披露声明前提，不能指责它完全隐瞒。问题是两序一致没有证明全序性质，且产品complete/point汇总没有独立proof-status轴（turn858file0）。修复要求准确区分保证等级；不要求此修复批次马上证明所有合法并发程序。

**精度与统计不是只差外部输入。** rc3 W2/W6还要求严格包围、按预算提高精度、请求分辨率、M的总体证据、δ及真实排除范围等。旧报告没有把这些逐项兑现状态交清楚，应保留在工作包账中，不把“参照完整”写成“已满足全部分析要求”。

**外部集的时间顺序。** 应先约定接口/总体/切分及冻结清单，再冻结工具并执行保留集；不能等看过工具表现再挑容易项。已公开反例永远是公开回归；本补充的定向整数测试也不是盲测。

**SPEC-ISSUE-1。** 旧AUDIT §7给出了SGD修改裁决，但该包没有附该争议的完整源代码、规格和目标版本对照。本补充撤回把它当成已证实裁决的用法，恢复“逐项对照后决定”。不指令Codex凭旧报告改写该规格，更不回写旧成绩。

## 4. 仍未完成的审计

29提交逐项差分与影响范围；W0全部路线/属性核验；W1全部421条实际契约/测试/触发的独立审核；原生源码全量回归；真实measure.run的classic/Gluon结果；设备验证；独立保留集；新盲测；自适应精度的完整兑现；族级有界统计验证。

这些分别见WORKPACKAGES和MATERIALS_LEDGER。本补充不把它们填成通过。

## 5. 可重复运行

先验包完整性：`python verify_package.py`。

本环境执行过：

```sh
python rerun/repro/run_probes.py
python rerun/repro/run_binding_probes.py
python recheck/repro/run_recheck.py --expect baseline
```

真实基线checkout上补AST核对和原生规则运行：

```sh
python rerun/repro/run_probes.py --repo /path/to/baseline --native
python rerun/repro/run_binding_probes.py --repo /path/to/baseline
python recheck/repro/run_recheck.py --repo /path/to/baseline --expect baseline --out /tmp/audit-baseline-ast.json
python recheck/repro/run_recheck.py --repo /path/to/baseline --native --expect baseline --out /tmp/audit-baseline-native.json
```

修复后原生小范围核对：

```sh
python recheck/repro/run_recheck.py --repo /path/to/fixed --native --expect fixed --out /tmp/audit-fixed-native.json
```

最后一条只检查新增整数、声明摘要、执行状态，不代表七项发现全部关闭。上面原生运行命令本环境未执行。完整解释器和GPU回归仍需Codex编写，并提交实际日志。
