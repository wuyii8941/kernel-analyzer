# GRAPH 合同审阅请求（2026-10-20）

作者：Claude（spec v3 作者）。审阅人：由用户本人担任，不是作者。

这份材料只陈述事实和待决问题，不替审阅人下结论。审阅结论请直接写在本目录 `GRAPH_REVIEW.md` 中，或在本文件末尾逐项填写。
按 2026-10-16 `REVIEW.md` 的三关组织：语义有依据、参照可信、对应关系成立。三关都通过，GRAPH 才能从 drafted 改为
`reviewed_for_implementation`，实现证据补齐后再改为 `validated`。

## 0 为什么现在审它

- 充分性评价中，E1 和 E3 严格层的完整参照都是 0，唯一原因是 GRAPH 为 drafted。
- 被组合的基础合同全部已经准入。
- GRAPH 一旦准入，按现有数据，E1 约 35%（141/400）、E3 约 29%（104/357）的编译单元可以成为「原样可用」。
  见 `sufficiency/results/REPORT_20261020.md`。

## 1 要审的文件

| 文件 | 内容 |
|---|---|
| `contracts/GRAPH.yaml` | 合同本体，包括 profile `eager`、读法 `inductor_default`、4 个决策、5 个条款、13 个例子、4 个变异要求、nonclaims、changes |
| `adapters/GRAPH.py`（56 行） | 适配器，调用 `kernel_spec.graph.evaluate`；读法与变异各改动一个方面 |
| `kernel_spec/graph.py`（326 行） | 完整求值，含覆盖检查 `covered_ops()`、共享选择、局部更新例外 |
| `kernel_spec/lazy.py` 中的 `evaluate_lazy`（第 822 行起） | 编译单元按坐标惰性求值，`corpus/dynamic_three_way.py` 用的是这个 |
| `tests/test_graph_*.py`（7 个文件，共 29 个测试） | 结构、调用参数、局部更新、原地修改、离散选择、来源、结果选择 |
| `contracts/DEVELOPMENT_CHECKS.json` 中的 `GRAPH` 项 | 2026-10-18，13 个例子、4 个变异、1 个读法，全部相容 |

## 2 第一关：语义有依据

GRAPH 的 f 是**程序本身的计算**：按拓扑顺序求值节点，每个节点用覆盖它的基础合同的语义，图中的类型转换按 CAST 规则舍入。
它检查的是编译器下降是否忠实，不检查模型作者的意图（GR.IDENTITY，已写入 nonclaims）。

待审的 4 个决策（都是 drafted）：

1. **GR.IDENTITY：图的 f 就是程序本身。**
   - 问：以「程序如写」作为任务，作为对编译 kernel 的语义裁决，你是否接受？
   - 接受的后果：GRAPH 结论只能写成「kernel 与其图片段一致」。B010、B017、B018 这类图本身有错的缺陷不在它的检出范围内。
   - 另外，图中嵌入的舍入常数（例如 float32 eps）属于 f 的一部分。
2. **GR.POISON：未定义的节点只影响读取它的节点。**
   - 问：是否接受局部传播？被否决的另一方案是「任一节点未定义，整图未定义」。
   - 例子：GR.E4 中 log(−1) 的结果没有被使用，输出仍为 6；GR.E5 中被使用，且乘以 0，结果仍为 domain_undefined。
3. **GR.JOINT_CHOICES：排序和 scatter 的离散选择。**
   - 规则：每个节点选择一个完整、合法的结果，同一节点的所有使用者看到同一个选择，不同节点各自独立选择。
   - 小规模时联合枚举，返回完整的输出集合；超出预算或懒查询遇到歧义时返回 not_established。
   - 问：「同一节点的使用者看到同一选择」是否符合 FX 的值语义？不同节点独立选择是否过宽？例如两个对相同输入的 argsort 可能选出不同的排列。
4. **GR.EFFECTS：原地修改。**
   - 规则：一般情况不支持。唯一例外是一个全新的 `aten.full` 分配，只作为一次 `index_put_` 的目标、不被返回、不别名、不被别处读取；
     这时把这次更新替换为函数式 `index_put`。
   - 问：所列条件是否足以保证替换前后图的输出相同？还缺什么条件？

还请看读法 `inductor_default`（省去融合内部中间转换的舍入）。它只用于诊断，不是程序语义。请确认这个定位。

## 3 第二关：参照可信

已有证据：
- 13 个例子都给出精确值，或给出完整结果集合。
- 4 个变异都由例子杀死：忽略转换、整图毒化、getitem 从末尾取、操作数顺序颠倒。
- 29 个测试。
- `changes` 记录了 2026-10-18 的 5 次修正，每次都附有修正前失败的测试。

缺的证据：没有类似 `evidence/mathfn_implementation.md` 的实现证据，也就是没有与一个独立实现的对照，例如逐节点用 Arb 或精确有理数重算。
所以 `implementation_status` 仍是 pending。这一项可以由作者补做，但对照实现的设计需要审阅人认可。

**作者自查发现的问题，需要你裁决：**

**问题 A：惰性求值没有执行合同覆盖检查。**
- 完整求值 `graph.evaluate` 会检查 `covered_ops()`：只接受 13 个基础合同（ARITH、MATHFN、LOGIC、CAST、LAYOUT、INDEX、
  REDUCE、SOFTMAX、MATMUL、CONV、RANDOM、ACT、SORT）的 covers，加上 GR.EFFECTS 和 GR.JOINT_CHOICES 允许的
  `aten.full`、`aten.index_put`、`aten.scatter`。
- 但 `lazy.evaluate_lazy` 不检查覆盖，它的文档字符串写明调用者必须先检查；而 `corpus/dynamic_three_way.py` 没有调用 `covered_ops()`。
- 结果：求值器实现了但没有任何合同覆盖的 27 个操作，在编译单元的开发层参照中被实际使用了。这 27 个是：
  `aten.pow`、`prims.iota`、`aten.scalar_tensor`、`aten.index`、`aten.arange`、`aten.full_like`、`aten.zeros`、`aten.ones`、
  Inductor 随机等。
- 充分性评价已按偏差 D5 把这类单元从开发层的可辩护数字中去掉：E1 35 个，E3 15 个。
- 问：准入时，是否要求 dynamic 路径在求值前必须调用 `covered_ops()`，使惰性求值与完整求值在覆盖上一致？
  作者建议要求，并先写修复前失败的测试。

**问题 B：`aten.full`、`aten.index_put`、`aten.scatter` 只由 GRAPH 的草案规则准入，没有放回 INDEX 合同。**
问：准入 GRAPH 时，是否连带接受这三个操作的语义？还是要求它们先进入一个已审阅的基础合同？

## 4 第三关：对应关系成立

这一关不属于 GRAPH 合同本身，而是编译单元绑定的事：源码匹配到图片段、provenance、输入输出映射，由 `bind_inductor.py` 和
`dynamic_three_way.py` 承担。它影响 GRAPH 准入后能否宣称「原样可用」。请确认审阅范围：

- [ ] 只审 GRAPH 的语义与求值，绑定另行审阅；
- [ ] 一并审绑定的拒绝条件，包括 K−Gπ 必须逐位相同、元组输出映射、launcher grid 与符号标量。

## 5 审阅结论（请填写）

| 项 | 结论（接受 / 修改 / 拒绝） | 意见 |
|---|---|---|
| GR.IDENTITY | | |
| GR.POISON | | |
| GR.JOINT_CHOICES | | |
| GR.EFFECTS | | |
| 读法 inductor_default 的定位 | | |
| 问题 A：惰性路径覆盖检查 | | |
| 问题 B：三个局部值操作 | | |
| 实现证据的对照设计 | | |
| 审阅范围（第 4 节） | | |
