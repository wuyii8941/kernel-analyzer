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
差异的总 RMS、reference-aligned scaling，以及只用前半样本选择、后半样本确认的
方向性。`check_backward=True` 时，单张量输出还会对位置参数和 `kwargs` 中所有不重复的浮点输入测量参数梯度差异；默认使用全 1 和交替正负两个上游 cotangent。自定义 `make_cotangent` 可以返回一个或多个上游梯度；模型、optimizer 和 loss 不是这个入口的前置条件。报告中的 `aligned_statewise_gain_interval` 对应逐状态比例均值；`aligned_ratio_of_sums` 只作有限样本描述，不与该推断端点混称。

如果算子本身接收“一个 tuple 作为单个参数”，请使用显式 `args` mapping，避免和
“多个位置参数”的 tuple 约定混淆。

结果状态只有三种科研含义：

* `SYSTEMATIC_BIAS_CONFIRMED`：冻结方向的 held-out 均值端点或 statewise aligned scaling 端点越过零；
* `SYSTEMATIC_BIAS_NOT_CONFIRMED`：这批输入没有确认所检验的结构，**不等于证明没有 bias**；
* `UNRESOLVED_MEASUREMENT`：执行失败、输出不匹配、非有限值或数据不足，不能作阴性结论。

入口输出的是 `DECLARED_INPUT_DISTRIBUTION_ONLY` 范围内的 bias 检查，不自动解释根因，
也不自动声称训练或 loss 后果。方向的精确符号频率端点要求后半样本是独立抽样；若
输入只是固定样本表，结果应理解为该表的描述。零投影会单独计数，不会被当作相反方向。
符号正负比例单独报告，不会因为比例不平衡就替代均值 bias 判定；两个主要端点使用 Bonferroni 分配的区间显著性。`allclose` 只作为辅助字段，不参与上述 bias 状态判定。输入中的共享 Tensor 别名会在 candidate/reference 两侧保持；不同输出 shape 会按签名分组，并返回 scoped 结果，不会跨向量空间强行堆叠。通过后，再把算子接入现有采集器，进行 local → gradient → update 和
人工主导的来源分析。

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

## 清理边界

本轮移除三个已被机器表和当前入口取代的覆盖快照：
`coverage.md`、`coverage_table_v1.md`、`model_coverage_audit.md`。
它们可由 Git 历史恢复；对应[覆盖表](../results/coverage/coverage_table_v1.json)、
[模型记录](../results/coverage/model_coverage_audit_v1.json)及实验数据原样保留。

仍被 runner、测试或复现协议引用的旧代码不因文件名带旧版本号而删除。
不修改讲稿，不删除失败、阴性或未决结果。所有新测试输出与缓存留在本仓库内。
