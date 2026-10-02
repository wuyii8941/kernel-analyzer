# 算子接入与自动化边界

本页说明实际可调用的代码，不把科研目标当作已完成能力。它取代旧的 T1–T4 系统宣传；
历史接口及其代码仍可用于复现。

## 一句话回答

**已有“审核接入后自动测试”的系统，没有“只放任意 kernel 源码就自动完成训练分析”的系统。**

这里的接入不仅是给算子起名。需要声明调用方式、输入、输出和参考；若研究训练影响，
还需要恢复训练状态并采集真实 backward 和参数写入。新模型或新编译版本也可能需要重新接入。

## 从简单误差测试开始

若已经有两个可调用实现，只需让它们处理相同输入值，再交给公共比较函数：

```python
import torch
from kernel_analyzer.local_tolerance import compare_outputs

x = torch.linspace(-3, 3, 64)
candidate = x.clone() * torch.sigmoid(x.clone())
reference = torch.nn.functional.silu(x.clone())
report = compare_outputs(candidate, reference, rtol=1e-5, atol=1e-7)
print(report)
```

这是实际 CPU 数值比较，不是科研阳性案例。容差仅用于示范，不能自动当作任意算子的
正确性标准。函数检查 shape、有限值、逐坐标误差及超容差坐标数；
它**不负责启动任意 Triton kernel、生成参考、准备输入或推导 backward**。
原地修改的算子必须分别恢复输入；随机算子还需声明随机事件怎样配对。

## 轻量 bias 入口

当新 Triton kernel 已经有一个可调用的 reference 和输入生成器时，可以直接复用
配对采样、方向和缩放统计：

```python
from kernel_analyzer import check_bias

report = check_bias(
    candidate=my_triton_operator,
    reference=reference_operator,
    make_inputs=lambda index: (make_input(index),),
    samples=32,
    check_backward=False,
)
print(report["status"])
```

`make_inputs(index)` 可以返回参数元组，也可以返回
`{"args": (...), "kwargs": {...}}`。入口会为每个样本分别调用两种实现，记录输出
差异的总 RMS、单位参照方向投影，以及只用前半样本选择、后半样本确认的
方向性。`check_backward=True` 时，单张量输出还会对位置参数和 `kwargs` 中所有不重复的浮点输入测量梯度差异；默认使用全 1 和交替正负两个上游 cotangent。自定义 `make_cotangent` 可以返回一个或多个上游梯度；模型、optimizer 和 loss 不是这个入口的前置条件。v3 报告中的 `aligned_projection_interval` 检验 `E[dot(u,r)/norm(r)]`，与误差向量同单位；`aligned_ratio_of_sums` 只作有限样本描述，不参与 aligned 判定，也不是固定坐标下的向量均值证明。每个阶段保存 `paired_sufficient_statistics`（逐样本误差能量、参照能量、内积），可直接复算该端点。

如果算子本身接收“一个 tuple 作为单个参数”，请使用显式 `args` mapping，避免和
“多个位置参数”的 tuple 约定混淆。

结果状态只有三种科研含义：

* `SYSTEMATIC_BIAS_CONFIRMED`：冻结方向的 held-out 均值端点或单位参照方向投影均值端点越过零，具体证据看两个分开的 decision 字段；
* `SYSTEMATIC_BIAS_NOT_CONFIRMED`：这批输入没有确认所检验的结构，**不等于证明没有 bias**；
* `UNRESOLVED_MEASUREMENT`：执行失败、输出不匹配、非有限值或数据不足，不能作阴性结论。

入口输出的是 `DECLARED_INPUT_DISTRIBUTION_ONLY` 范围内的 bias 检查，不自动解释根因，
也不自动声称训练或 loss 后果。方向的精确符号频率端点要求后半样本是独立抽样；若
输入只是固定样本表，结果应理解为该表的描述。零投影会单独计数，不会被当作相反方向。
符号正负比例单独报告，不会因为比例不平衡就替代均值 bias 判定；两个主要端点使用 Bonferroni 分配的区间显著性。`allclose` 只作为辅助字段，不参与上述 bias 状态判定。输入中的共享 Tensor 别名会在 candidate/reference 两侧保持；不同输出 shape 会按签名分组，并返回 scoped 结果，不会跨向量空间强行堆叠。通过后，再把算子接入现有采集器，进行 local → gradient → update 和
人工主导的来源分析。

默认两端点均检验零均值，不要求工程容差；检出非零不代表值得修复或影响训练。
可选的 `directional_margin` 与 `aligned_projection_margin` 都使用误差向量的单位，
仅用于另行声明的应用阈值。旧版 `aligned_margin` 是无量纲比例，不能原数迁移：非零
旧参数会明确报错，零值调用保持兼容。参照能量不超过 `reference_energy_floor` 时，
aligned 端点返回 `NOT_ASSESSED`，记录对应样本，不静默删除它们后宣称覆盖原总体。

### 对齐端点迁移与旧结果（2026-09-21）

当前统一入口只采用单位方向投影，不同时运行两个 aligned 判定。旧 JSON、来源干预、
训练 loss 及冻结协议验证器保留原义，不把历史 gain 区间改名成 projection 区间。
训练 profile 的旧能量加权协议也没有被冒充为新检验。复算命令为
`PYTHONPATH=src:. python scripts/audit_aligned_projection_migration.py --output <仓库内新JSON路径>`。
唯一机器对照记录是 [aligned_projection_migration_v1.json](../results/property/root_cause_closure_v1/aligned_projection_migration_v1.json)：

* AdamW8bit 的 32 条独立历史仍为正；新均值约 `1.00912e-4`，95% 条件 Student 区间
  `[8.07313e-5, 1.21093e-4]`。16 条 source-link 历史的默认/补偿两臂结论也不变。
* 两版 TileLang sum-spec 的 8 个输入库端点比较不变；精确有限库的零均值结论不受影响。
* 冻结 benchmark 的 17 个原坐标端点中，Qwen `backward_540_out_ptr1` 的梯度、
  moment1、moment2 三个端点由旧比例的负区间变为新投影区间跨零。这里采用共同
  endpoint alpha=0.025 做事后敏感性比较，不是修改原 benchmark 的协议或新增总体阳性。
* 15 份旧 checker 报告、250 个去除重复别名后的阶段只有汇总，不能完整复算；
  另有 61 个 benchmark 阶段只有 sketch 几何，不能当作原坐标结果重判。

迁移保留既有 Student 近似及其局限，未新增有限样本分布无关保证；尤其样本方差为零
不能自动证明总体没有波动。此次比较隔离指标变化，两侧用零门槛，不把旧版非零比例
门槛移植到新单位。以上端点数不是新的问题组数量，根因证据和固定方向证据并未改写。

### TileLang 轻量接入

TileLang 的编译结果可以直接复用同一个 bias 检查器，不需要为 TileLang 另写一套统计公式：

```python
from kernel_analyzer import check_tilelang_bias

report = check_tilelang_bias(
    candidate=compiled_tilelang_kernel,
    reference=reference_operator,
    make_inputs=lambda index: (make_input(index),),
    samples=32,
)
```

如果编译结果返回多个输出，可通过 `candidate_output_index` 或
`candidate_output_key` 选择待检查的张量。批量入口为
`scripts/run_tilelang_bias_check.py`；它要求模块显式提供 candidate、reference 和
输入生成器，拒绝覆盖已有报告。该接入只负责调用与统计复用，不自动编译任意源码、生成
reference 或解释根因；真实 TileLang GPU 结果仍需在安装 TileLang 的环境中运行。
统计入口默认检验非零平均作用，不把微小的非零结果直接包装成工程意义。若使用 TileLang
autotune，候选配置必须先在开发输入上选定，再在独立确认输入上运行；这个选择流程由
接入脚本记录和约束，不由统计入口自动替用户生成。

本仓库已用 TileLang 0.1.14 在 RTX A6000 上实际编译并运行三类 smoke kernel：
`scripts/tilelang_gemm_bias_smoke.py`、`scripts/tilelang_elementwise_bias_smoke.py` 和
`scripts/tilelang_reduction_bias_smoke.py`。本轮真实运行结果分别保存在
`results/property/tilelang_gemm_bias_smoke_v1.json`、
`results/property/tilelang_elementwise_bias_smoke_v1.json` 和
`results/property/tilelang_reduction_bias_smoke_v1.json`；三者测量状态均为 `VALID`，
且在声明的随机输入分布上未确认固定方向 bias。GEMM、elementwise、reduction 的输出
RMS 分别约为 `8.71e-6`、`0`、`3.28e-8`，这是输出级有限精度差异或逐位相同，
不是训练级结论。该验收使用 CUDA 12.1 与 GCC 11；系统默认 CUDA 11.1/GCC 9 会因
TileLang 需要的 C++20 编译选项而失败，这类编译失败只能记为环境不兼容，不能当作
算子阴性结果。

另外，`scripts/tilelang_known_positive_bias_smoke.py` 在同一真实 TileLang GPU 路径上
加入预先知道的固定输出偏移，结果为 `VALID / SYSTEMATIC_BIAS_CONFIRMED`，记录在
`results/property/tilelang_known_positive_bias_smoke_v1.json`。这只是检测器的阳性校准控制，
不是自然 TileLang 算子的根因案例；它与三个真实算子样例一起说明入口不会把所有结果都
强行判成阳性或阴性。
同一套入口的命令行回归也已在真实 GPU 上完成：`results/property/tilelang_elementwise_cli_v1.json`
验证 reference-based CLI，`results/property/tilelang_reduction_family_cli_v1.json` 验证
reference-free family CLI；二者均为 `VALID`，没有另写统计判据。

### TileLang 的无 reference 家族调用

在没有完整 reference 时，TileLang 也可以复用同一套“证据而非伪造 verdict”的家族接口，
而不是另写一套统计公式。两个最小调用形态是：

```python
from kernel_analyzer import (
    check_tilelang_reduction_order,
    check_tilelang_softmax_saved_state,
)

# candidate 与 variant 都是已经编译或可调用的 TileLang kernel。
reduction_report = check_tilelang_reduction_order(
    candidate=tilelang_sum_forward,
    variant=tilelang_sum_reverse,
    make_inputs=make_inputs,
    make_negated_inputs=make_negated_inputs,
    samples=32,
)

# candidate 输出是一个 mapping 时，用 candidate_output_key 选择概率张量。
softmax_report = check_tilelang_softmax_saved_state(
    candidate=tilelang_saved_probability,
    make_inputs=make_softmax_inputs,
    candidate_output_key="probability",
    samples=32,
)
```

批量调用入口是 `scripts/run_tilelang_family_check.py`。模块只需要导出 callable 和输入
工厂；归约族另外导出一个同语义 arithmetic variant，必要时导出取负输入工厂。例如：

```bash
PYTHONPATH=src:. python scripts/run_tilelang_family_check.py \
  --family reduction_order \
  --module scripts.tilelang_reduction_family_smoke \
  --candidate reduce_sum_forward --variant reduce_sum_interleaved \
  --make-inputs make_inputs --make-negated-inputs make_negated_inputs \
  --samples 4 --output results/property/tilelang_reduction_family_smoke_v1.json
```

同一脚本也可运行 `--family softmax_saved_state`，并用
`--candidate-output-index` 或 `--candidate-output-key` 绑定输出。该入口仍要求调用者声明
数学家族、输入分布和 variant 的同语义前提；它不会从陌生 TileLang 源码自动生成
reference、自动解释根因，也不会把受控差异直接判成总体 bias。`scripts/tilelang_reduction_family_smoke.py`
是一个真实的 TileLang reduction-order 调用用例，和前三个 reference-based smoke 分开计数。
在 CUDA 12.1、GCC 11、RTX A6000 上的实测报告为
`results/property/tilelang_reduction_family_smoke_v9.json`：4/4 个样本均完成，归约变体
差异均非零且同号，平均输出差异为 `0.05859375`、最大相对 RMS 约 `1.59%`；取负输入的
奇对称残差均为零。这是声明输入库下的受控算术证据，报告仍保持
`bias_decision=NOT_ASSESSED`，因为它没有完整 reference，也不是总体 bias 证明。

### 只给一个 TileLang 算子的自动性质筛查

如果新算子属于已声明的奇对称族（例如有符号归约），现在不需要再提供 reference 或
第二个 variant。工具会自动对每个输入的所有浮点 Tensor 取负，运行同一个 candidate，
检查 `f(-x) == -f(x)`：

```python
from kernel_analyzer import check_tilelang_odd_symmetry

report = check_tilelang_odd_symmetry(
    candidate=my_tilelang_kernel,
    make_inputs=make_inputs,
    samples=32,
    tolerance=1e-5,
)
```

这个入口只有一个算子和一个输入工厂；输出中的
`property_decision=PROPERTY_VIOLATION_OBSERVED` 表示该声明性质在样本上被违反，
同时给出 `bias_signal=BIAS_CANDIDATE_PROPERTY_VIOLATION` 作为需要继续调查的 bias 候选信号；
`PROPERTY_NOT_VIOLATED_IN_DECLARED_SAMPLE` 表示当前样本没有观察到违反，顶层
`bias_decision` 仍为 `NOT_ASSESSED`。这是因为任意陌生 kernel 没有通用的“正确值”定义，
奇对称也不适用于任意算子；它是自动化的 reference-free 证据筛查，不是无规格的 bias
Oracle。

命令行形式为：

```bash
PYTHONPATH=src:. python scripts/run_tilelang_family_check.py \
  --family odd_symmetry \
  --module scripts.tilelang_reduction_family_smoke \
  --candidate reduce_sum_forward --make-inputs make_inputs \
  --samples 32 --output results/property/tilelang_single_operator.json
```

真实 GPU 回归已经覆盖两种结果：`tilelang_single_operator_reduction_v2.json` 对真实
TileLang reduction 通过该性质；`tilelang_single_operator_positive_v2.json` 对带固定
输出偏移的单个 TileLang candidate 报告 4/4 个样本性质违反。这两个报告都保留
`bias_decision=NOT_ASSESSED`，不会把性质违反越级称作完整 bias 或训练后果。
同一入口对真实 elementwise candidate 的结果保存在
`tilelang_single_operator_elementwise_v2.json`，也未观察到奇对称性质违反。

## 没有完整 reference 时的证据模式

任意陌生 Triton kernel 都不能仅凭源码自动推断数学规格；因此没有完整 reference 时，
系统不会伪造 `SYSTEMATIC_BIAS_CONFIRMED` 或 `SYSTEMATIC_BIAS_NOT_CONFIRMED`。但可以
运行一个更轻的证据入口：

```python
from kernel_analyzer import diagnose_kernel

report = diagnose_kernel(
    candidate=my_kernel,
    make_inputs=make_inputs,
    rounding_variants={"FP32_ACCUMULATION": fp32_variant},
    input_consistency_check=check_saved_state_contract,
    specification_check=check_declared_formula,
)
```

它支持三种由接入方明确声明的检查：

* `rounding_variants`：同一语义、只改变精度、cast 或合法归约顺序；报告
  `ROUNDING_VARIANT_DIFFERENCE_OBSERVED`，并汇总样本级有符号差异是否同号；即使报告
  `CONTROLLED_DIRECTIONAL_EVIDENCE`，也不把它自动称为总体 bias；
* `input_consistency_check`：检查输入、saved tensor 或递推状态的关系，例如概率行和、
  forward/backward 共享统计量和状态边界；违反时报告
  `INPUT_CONSISTENCY_VIOLATION`；
* `specification_check`：检查接入方已经声明的数学性质或近似范围；违反时报告
  `DECLARED_SPEC_DEVIATION`。

没有提供某一项契约时，对应字段为 `NOT_DECLARED`；没有完整 reference 时固定输出
`bias_decision=NOT_ASSESSED`。因此这个入口能自动筛出值得继续分析的舍入、输入一致性和
规格偏离证据，但不会把“检测到差异”越级写成总体 bias、训练后果或根因闭合。

当前 `scripts/run_reference_free_diagnostic_examples.py` 只做**人工构造的工程演示**：
重复同一组手写加数、手动设置不一致的 softmax denominator，以及把 SiLU 与 identity
契约作比较。它复用了已有诊断函数，但没有重放真实 Liger/Granite 或 saved-P 捕获。
重复同一个输入八次不是八个独立状态；同号差异不是总体均值证明；刻意不匹配的公式
不是新发现的近似 bug。真实案例验收属于下面的待实施任务。

## 轻量诊断工具实施计划与当前实现

计划制定：2026-09-21。两个首批家族模板、TileLang 包装器和命令行入口已经实现并经过
CPU 端适配器测试；真实 TileLang GPU 执行仍以安装了兼容 TileLang 的环境为前提。这里不
据此改写旧实验结果，也不把受控家族证据写成完整 bias 证明。

目标是“给出调用包装、输入和一个已支持的语义族，即可测试该族的数值性质及所声明的
bias 端点”。不要求每个使用者重写完整 reference，但仍须有人审核家族规格。
没有规格的陌生源码只能报告无法判断，不能靠升精度或多次运行推断它应该算什么。

### 当前可以直接复用

- `check_bias`：配对调用、输入克隆和 RNG 配对、输出形状分组、有限 backward 探针、
  开发/确认方向和现有统计输出；reference 仍是必需输入。
- `diagnose_kernel`：合法数值变体和性质回调的通用执行骨架；当前性质由调用者编写。
  两个首批家族的模板入口是 `check_softmax_saved_state` 和 `check_reduction_order`，
  不需要每次重写这些回调。
- `source_reference_registry.py` 与现有 saved-state/reduction 诊断：复用已审核语义和
  比较代码，不再建立第二套家族公式。
- 训练采集和实际写入记录：仅作可选后续连接，模型、optimizer、loss 不阻塞小工具验收。

### 顺序一：冻结两个家族规格（已实现模板）

规格在查看本轮验收输出前固定，不根据希望得到的结论临时编写：

| 首批家族 | 使用者提供 | 工具复用的检查 | 结论边界 |
|---|---|---|---|
| softmax backward / saved probability | callable、输入生成器、概率/保存量与 cotangent 的字段绑定 | 常数 cotangent 响应、平移不变性、概率行和；保存 scores/statistics 可用时检查一致性 | 性质残差不为零不等于全部候选误差，也不自动证明平均 bias |
| 固定加数的归约 | callable、真实加数与轴/mask/dtype；需要时提供合法变体包装 | 同输入顺序对照、条件满足时的符号对称性；可行时计算准确和或有误差界的参照 | 变体间差异没有天然真值；纯差异报告与相对数学和的误差报告分开 |

软件提供模板，不承诺自动改写任意 Triton/TileLang 源码。语义不匹配（例如两个不同
GELU 近似规格）先标为不可直接比较，不归入同规格舍入 bias。性质通过只覆盖被检查
的约束，不能取代完整 reference 的正确性结论。

验收已完成：每个家族有可读契约、Python 调用入口、命令行参数和最小调用示例；同族换
实现只改调用/字段绑定，不改统计公式。对应入口为
`check_tilelang_softmax_saved_state` 与 `check_tilelang_reduction_order`。

### 顺序二：补齐统一执行与推断边界（已实现基础版本）

沿现有入口增加契约选择和公共报告，不再创建一套大系统。报告分别记录：

- 是否有效执行、哪项失败或未提供；保留分母中的失败样本。
- 比较的是“两个实现之差”还是“相对理论零值的性质残差”。
- 观测差异、性质违反、平均端点的确认/未确认/条件不足，不能共用一个 PASS。
- 独立单位、数据用途、方向规则、margin、区间方法及适用范围；固定集合只给描述。
- 有 reference 时调用已有 `check_bias`；无 reference 时，只能对数学上应为零的
  有符号性质残差检验均值，不能称为任意输出向量的完整 bias 检验。

先处理已发现的工程和统计边界：没有检查项不报告有效诊断；非有限输出、部分变体
失败、错误的 callback 返回值不得变成通过；维持已有别名、kwargs、shape 分组规则。
重复回放固定输入不增加独立样本数，但合法有放回抽样出现重复内容也不得被自动剔除。

统计实现优先复用公共均值推断代码。统一处理样本零方差、重尾、小样本和方向选择；
条件不足时不给精确保证。有观察前有效幅度界时再接入经核对的有界均值区间（包括
讲稿讨论的经验 Bernstein 路线）；否则 t 路线明确标为有条件近似。新增路线的单元
测试和校准必须调用生产代码，不能由另一个脚本代算。零 margin 的检出与正 margin
的实际幅度判断分开；未确认不增加“无 bias”标签。多端点和多家族的错误率口径在
运行前固定，不能只校正单个示例内部的两项检验。

基础验收已覆盖全零/低效应、已知属性违反、零方差、输出结构错误和执行失败的保留；
报告仍将 reference-free 结果固定为 `bias_decision=NOT_ASSESSED`。经验 Bernstein 等
有界总体路线不是 TileLang 入口的默认保证，仍需调用方提供观察前有效界并经过单独校准。

### 顺序三：复现已有真实案例，不另开大训练（接入模板已具备）

1. 从已有 saved-P 捕获绑定真实保存量，执行原 backward 的常数 cotangent/平移探针；
   对照一致保存量的版本，报告性质残差及其对应的源码边界。
2. 从 Liger 或 Granite 取真实冻结加数，只改变合法归约顺序，连接现有局部/写入结果。
   先证明捕获可读、调用绑定正确；缺 operands 时仅补一次有限采集，不从摘要虚构。
3. 用同语义的正确实现和已保存阴性控制检验误报；人工注入只作校准，单独标注。
4. 生成一份机器报告并在本页链接。Triton 的公共入口是
   `scripts/run_reference_free_family_check.py`，TileLang 的对应入口是
   `scripts/run_tilelang_family_check.py`；两者都从 importable module 加载 kernel、输入
   工厂和归约变体。已有案例是回归复现，不宣称“盲发现”或新的独立训练确认。
   固定 bank 与独立有放回抽样分别表述。

已具备的 TileLang 用例是 `scripts/tilelang_reduction_family_smoke.py`（两个真实
TileLang reduction callable）以及 GEMM、elementwise、reference-based reduction smoke。
本轮已经在 CUDA 12.1、GCC 11、RTX A6000 上完成真实 GPU 编译运行；除上述三类阴性/未确认
样例和人工阳性控制外，`results/property/tilelang_reduction_family_smoke_v9.json` 还记录了
同语义、同精度而改变归约顺序的受控差异。完整验收仍不要求自然算子都出现平均 bias：至少
能从真实实现复现一个可核查的性质诊断和一个归约对照，保留阴性/未确认，结果能回到已有
来源记录。

### 顺序四：用新调用包装验证可复用性后停止扩展

冻结模板后，再接一个此前未用于模板调试的 Triton 或 TileLang 实现。使用者只补
callable、输入、家族绑定，工具复用执行与统计；记录实际新增代码和运行成本，不补造
历史人工时间。失败也保留，不以必须阳性作为接入验收。

最小版完成条件：上述两个家族可用、真实案例回归成立、阴性和无效条件有检查、第三个
实现无需重写判据。随后回到主线的根因研究，不追加任意源码理解、自动修复、大模型
训练或所有五环因果闭合。小工具与讲稿五步链的区别是：它自动化前端检查与部分推断，
后续根因和训练验证仍是实验工作。

## 三阶段训练分析怎样复用

| 环节 | 现有代码 | 仍由接入方提供 |
|---|---|---|
| 家族参考 | `src/kernel_analyzer/source_reference_registry.py` | 审核过的公式、数值变体及源码/输出边界；未知家族明确拒绝 |
| 训练执行 | 已有家族采集器与 `scripts/run_training_numerical_analysis.py` | 模型、输入、调用位置、参数关联和状态恢复 |
| 公共采集 | `FixedSuiteImplementationCapture` | 已实际执行得到的各阶段 candidate/reference 张量 |
| 统一分析 | `training_numerical_analysis.analyze_artifact` | 原坐标充分统计量、状态划分、比较范围和实际写入记录 |
| 来源分析 | `source_factorial.two_factor_source_decomposition` | 相同输入下测得的单因素/联合修改输出及合理的因素解释 |

公共采集器不是运行器：它接受张量，不接受任意算子源码后自行编译。
当前这个采集器固定为 32 个状态、16+16 划分，并经 FP32 表示采集差异；
不能据此承诺任意样本数、FP64 原值保真、多输出布局或随机执行都自动受支持。
小向量保留完整方向数据，大向量使用摘要；原坐标能量和摘要方向的解释分开。

实际写入必须来自 optimizer step 前后参数之差，不能把计算出来但没有写入的 update
改名为实际写入。固定集合 Q 判断不自动等于随机总体 bias 或最终 loss 判断。
源码审核、数学解释与选择干预仍需要研究判断，不能由自动采集替代。

## 当前执行入口

使用已有包含 PyTorch、NumPy、SciPy 和 pytest 的 Python 环境，从仓库根目录运行：

```bash
PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py --help
PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py coverage --help
PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py analyze --help
```

- 现有 release 与参数映射的 `coverage freeze/run/report` 用法见[采集说明](numerical_coverage_execution.md)。
- 家族专用接入走已有对应 runner，不保证一个 `coverage` 命令覆盖所有注册参考。
- 已采集结果用 `analyze RAW OUTPUT --protocol PROTOCOL` 复算；未执行新 kernel。
- 安装后的 `kernel-analyzer analyze SPEC`、`AnalysisSpec` 和 T1–T4 是历史路径。
  旧示例 `examples/qwen_retained_spec.py` 不是新算子的即插即用模板。

缺参考、参数关联或有效执行路径时，系统应报告缺项，不把未完成测试算作通过或阴性。
在已审核家族之外增加实现，可能只需补调用适配，也可能必须补家族参考；不能保证零代码接入。

## 本轮如何验证系统，而不是只检查报告是否存在

`tests/test_operator_analysis_smoke.py` 实际运行两个 PyTorch 算子、autograd 和
AdamW，把三个阶段交给已有采集器和统一分析，再核对原坐标统计与直接计算的结果。
它包含相同实现对照、等价数学表达式和刻意错误的测试实现。
这是 CPU 端到端工程验证，不是新发现、GPU kernel 复现或总体统计确认。

还需运行已有参考注册、未知家族拒绝、缺失参数关联、实际写入和统计边界测试。
本轮不启动大模型训练，不从测试通过推断所有 Triton 家族均能运行。

## 已有验证记录与复现边界

轻量入口的真实 Triton 接入见[算子示例](bias_checker_triton_examples_20260914.md)。
主流模型配置扫描、行为级来源探针、TileLang smoke 和人工阳性控制的机器结果统一从
[结果索引](../results/README.md)进入，不在本页维护多批次计数与逐模型结论。
配置形状的生成输入扫描不是完整 checkpoint 训练；statewise aligned 端点不等于固定
向量均值；字段中的 `CONFIRMED` 必须连同端点与输入分布阅读。

历史 `AnalysisSpec`、T1–T4、ConcreteFBProof 与训练 equivalence v1/v2 保留各自含义。
VJP/参数可达性证明不是 bias 证明；旧投影协议不能改称原坐标全空间等价。
这些旧入口用于复现，不是陌生 kernel 的即插即用模板。

## 维护边界

本页只维护 API、输入责任、限制和小工具实施计划；案例状态只在
[案例总表](root_cause_closure_current.md)维护。运行日志与生成报告放在所属的
`results/` 目录，不为每轮试验新增一份主线文档。

不修改用户讲稿，不删除原始数据、阴性或失败结果。仍被 runner、测试或复现协议使用
的旧代码不因版本名而删除。新输出和缓存放在本仓库内；历史外部缓存路径只按原记录
注明，不把未移动的文件写成已经迁移。
