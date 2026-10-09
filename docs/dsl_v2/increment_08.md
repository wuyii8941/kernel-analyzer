# DSL v2 第八个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W1（每条支持规则有正例、边界、违反前提三类测试和实际触发证据）。
本登记写于实现之前。

## 1. 现状与缺口

增量 6 代码的带追踪全套测试（`.cache/dsl_v2/w1_inc6/`）后，契约基线是 **335/418** 支持条目三类齐全且有触发证据（80.1%）。
剩下 83 条分四类：

| 类 | 条数 | 原因 |
| --- | --- | --- |
| 3.6.0 前端不生成的 TTIR 名字（`math.acos` 等 20 个初等函数、`arith.ceildivsi` / `floordivsi` / `index_cast` / `negf` / `bitcast` / `fptoui`、`ub.poison`、`tt.unsplat`） | 约 30 | 没有任何测试触发；要用 TTIR 文本测试 |
| 终结符与结构操作（`tt.return`、`scf.yield`、`scf.condition`、`tt.reduce.return`、`tt.scan.return`、`tt.map_elementwise.return`、`cf.br`、`cf.cond_br`、`tt.func`、`tt.call`） | 10 | 求值器在循环里直接解释它们，不经过 `_exec`，追踪看不到 |
| 未测到的归约 combiner 与 inline asm 模式（and、or、max_int、min_nan、prod 等 12 个；ex2 / lg2 / rcp / rsqrt / sin / cos / sqrt / tanh / mov 等 11 个） | 23 | 没有对应的测试 |
| 只有正例（`tt.get_program_id`、`tt.get_num_programs`、`tt.int_to_ptr`、`tt.ptr_to_int`、`tt.atomic_cas`、`gpu.barrier`、Welford、通用 combine、`tt.assert`、`tt.print`、`cf.assert`、`tt.gather`、复合规则 3 条） | 约 20 | 缺边界或违反前提类 |

## 2. 做法

- 终结符与结构操作在被解释的位置记追踪（不改语义）。
- 每个缺口条目补三类逐签名测试：能用 3.6.0 编译出来的用编译产物，不能的用 TTIR 文本（写明是手写的 TTIR，语法与 3.6.0 / 主线一致）。
  期望值独立计算（mpmath、精确整数、Fraction）。
- 复合规则（嵌套控制、load/store 别名、跨启动状态）各补边界和违反前提用例。
- 契约生成器不改判据：仍是「三类测试都存在且通过、并有实际触发」。

## 3. 预算

约半天。

## 4. 预期（实现前写定）

- 三类齐全且有触发的条目从 335/418 升到至少 400/418。
- 剩下的预计是复合规则或需要设备行为的条目（例如 `launch sequence` 需要多次启动的捕获），在报告里逐条列出原因。
- 补测试过程中发现的求值器缺陷，记录并修复；修复前该条目不计入完成。
- 回归：已有全部测试不变。
