# 2 现有工具

工具 4.0（`dsl-v2` 分支，未发布、未冻结；冻结版本仍是 `general-v3.1` / 工具 3.1，见 [CURRENT.json](../CURRENT.json)）。

## 做什么

对一个 Triton kernel 的每次启动，从捕获的 IR 自动构造实数语义 G 的严格包围（参照 K_R），与实际输出 K 比较，
再用固定的方向规则与端点保守统计判断数值差异 e_num = K − G 的平均作用是否非零。有任务规格时（模式 B）另报 e_sem = G − f。
状态与参数的写入可以放在增量观察层上看（`compare.observe`）：比较实际写入的增量与参照增量，报告未写入的比例、保留比例与方向。
没有 IR 的实现（ATen、cuBLAS）用黑箱模式（`compare.mode = "black-box"`）：只比较 K − f，规格要手写。

| 层 | 代码 | 说明 |
| --- | --- | --- |
| 捕获 | `2_tool/src/kernel_analyzer/reference_eval/capture.py` | 记录每次启动的参数、张量前后值与编译产物（TTIR / TTGIR / PTX） |
| 解析与映射 | `reference_eval/ttir_parser.py`、`ttir_mapping.py`、`rule_registry.py` | IR 解析；每个操作的规则（支持 / 声明前提 / 未建立 / 拒绝），注册表 [data/rule_registry.json](data/rule_registry.json)（Triton 3.6.0 回归 profile），报告 [pre-reorg-20261009:docs/rule_registry_report.md](docs/rule_registry_report.md) |
| 求值 | `reference_eval/ttir_eval.py`、`intervals.py`、`numbers.py`、`layouts.py`、`ptx_bits.py` | 区间语义（SumK / DotK、MPFR / Arb 初等函数）、控制流、原子与执行有效性、共享存储、NVIDIA / AMD 目标操作、位级 inline PTX |
| 核内定位 | `reference_eval/emulate.py`、`tensor_core.py`、`scan_order.py` | 逐位模拟与逐节点精确替换（锁定 3.6.0 / sm_86 的下降规则；张量核 dot 与一维扫描的组合规则、FTZ、f64 精确求值见组合研究） |
| 偏差预测与审查 | `reference_eval/bias_predict.py`、`audit.py` | 不运行 kernel 的组合偏差预测（局部契约 + 切向传播，契约表 `data/bias_contracts.json.gz`）；审查入口 `audit(launch, spec=..., names=..., programs=None | "all" | [...])`：预测、逐位核对、逐节点归因、规格对比、常用容差检查、改法与代价、各 program 的偏差范围；报告先给作者摘要、技术细节在附录（案例见 1_experiments/flash_attention_case） |
| 统计 | `reference_eval/analysis.py`、`detect.py`、`sensitivity.py`、`update_layer.py`、`observation.py` | 方向规则 R1 / R5（固定均值）、R2 / R3（对齐），类内 Holm，契约 v3 的「无法判断」规则（`kernel_analyzer/contract_v3.py`）；有界路线类内 Holm，非零 / 等价两轴分报；观察层（输出 / 增量）与每单位的描述量（`observation.output_unit`：相对 RMS、对齐、保留比例、整体缩放的能量份额）；每个独立单位只有一个确定值时（配对训练的差）用同一推断（`analysis.unit_axes`）。实验脚本调用这些函数，不另写统计 |
| 入口 | `2_tool/src/kernel_analyzer/measure.py`（统一入口）、`check.py`（绑定入口）、`cli.py`（安装后的 `kernel-analyzer-measure` / `kernel-analyzer-analyze`） | 声明调用、输入来源、比较方式与预算，工具展开其余部分（含固定的 bias 查询 `query`） |
| 组合规则 | `reference_eval/storage_effects.py`（来源 / 内存效果）、`composition_rules.py`（登记） | 存储身份与生命周期、真实覆盖、内容不是来源、跨启动沿用、生产者格；精度相关规则与停止类别；查询固定、有界路线族保证、两个判定轴 |

DSL v2 的实现进度、每个增量的预期与结果见 [1_experiments/dsl_v2/README.md](../1_experiments/dsl_v2/README.md)；
外部审计（基线 1aee15e）的七项发现 F01–F07 已按任务书修复，欠账也已补齐（[3_audits/fix_1aee15e](../3_audits/fix_1aee15e/README.md)，
等审阅方复核）；仍开放的工作包差距见该页第 4、5 节。第一批共同语义内核（来源 / 精度 / 安装 / 查询）的修复、组合规则与验收见 [3_audits/batch1_semantic_core](../3_audits/batch1_semantic_core/README.md)。

## 用法

```bash
export HOME=$PWD/.cache XDG_CACHE_HOME=$PWD/.cache TRITON_CACHE_DIR=$PWD/.cache/triton TMPDIR=$PWD/.cache/tmp
export PYTHONPATH=2_tool/src
PY=/data1/tzh/envs/ka_main/bin/python
$PY 2_tool/scripts/measure.py --declaration 2_tool/examples/general/bsr_softmax.json --out report.json   # 统一入口
$PY 2_tool/scripts/run_reference_analysis.py ...      # 其他环境捕获的启动包，在 ka_main 中分析
$PY -m pytest -q 2_tool/tests                          # 测试：不得改写已跟踪文件（2_tool/tests/conftest.py 守卫）
```

声明只写四样：合法调用、每个输入的来源、比较方式、预算（示例 [2_tool/examples/general](examples/general/)）；用 Python 绑定 kernel 见
[2_tool/docs/binding_guide.md](docs/binding_guide.md) 与 [2_tool/examples/bindings](examples/bindings/)。报告给出每个输出的完整参照率、宽度相对 ulp 的分布、
平均作用的判定（非零（方向）/ 未确认 / 无法判断）与失败归因。「未确认」不等于「没有 bias」。

## 能做与不能做

- **能**：单个 Triton kernel（classic 由 TTIR，Gluon 由 TTGIR）、输出由 Triton 写出、操作在注册表内；模式 A 测数值差异的平均作用，
  有规格时模式 B 另报 e_sem。NVIDIA sm_90 / sm_100 与 AMD gfx942 / gfx950 的 TTGIR 操作有 CPU 语义（设备验证 0）。
- **不能**：注册表外的操作与未登记的 libdevice / inline asm（明确拒绝）；cuBLAS / ATen 写出的输出没有自动参照（黑箱模式只能对手写规格比较）；warp specialization、
  asm 内访存；保存值、反向与优化器写入作为一个自动调用图。
- **证明状态**：CAS 自旋锁（单一临界区）的结果在 z3 证明临界区两两可交换时无条件成立，scan 在 z3 证明组合区域结合时无条件成立；
  没有证书时单列为「complete_under_premise」，不进完整率与统计（需要 `z3-solver`）。
- **调用级来源**：非 Triton 上游只有在不计入测量的轨迹运行证明它只由声明输入的复制或精确常数产生时才算调用级；值相等不算。
- **已知局限**：工作精度（三级）只作用于求和型规则，端点是 float64；按 program 批处理未实现；声明 M 时报告逐规则有界路线，没有族级控制；
  多把锁、嵌套锁等同步模式没有证书。
- **审查入口（`audit.py`）的局限**（详见 [docs/audit_status.md](docs/audit_status.md)）：只审查调用里的第一个 kernel；归因每个位点一次完整仿真、
  不缓存、只限精度；`spec=` 不用规格库的输入契约；执行有效性与 inf / NaN 源头不进报告。
- **第二轮实测的局限**（[1_experiments/tolerance_study](../1_experiments/tolerance_study/README.md)）：
  - 常见机制里最常用的实现（融合 AdamW、cuBLAS TF32）是 ATen 内核，只能黑箱比较。
  - k = 4864 的 fp32 矩阵乘，每单位一个 128 × 896 的输出，在模式 A / B 下单个作业接近或超出 1 小时 CPU 预算（区间参照逐元素求值）。
  - 调用内部临时切换精度开关时，`precision_environment` 记录的是进入调用时的状态（T2 只记录、不设置）。
  - 被测函数经 Inductor 编译时，生产者追踪按 eager 重放，来源记为 not aligned。
  - 单步的等价轴需要事先声明界限；没有声明时只报非零轴，fp32 舍入级的零星非零判定不能写成「等价」。

## 目录

| 路径 | 内容 |
| --- | --- |
| `2_tool/src/kernel_analyzer` | `measure.py`、`check.py`、`provenance.py` 与 `reference_eval/`（共 24 个模块，含证书 `certificates.py`、`lock_certificate.py`） |
| `scripts/` | `measure.py`、`run_reference_analysis.py`、`enumerate_ttir_registry.py`；`dsl_v2/`（W0 清单、捕获插件、注册表与契约生成、两层对照、AMD 阶段）；`acceptance/`（结构验收 v1.1 适配器）；`essential/contract_v3.py`、`common.py`；`general/acceptance_run.py`（冻结检查）；`reference_eval_kernels.py`、`mutation_kernels.py`（测试用 kernel） |
| `tests/` | 52 个测试文件与 `data/`（官方 TTGIR fixture、TTIR 语料、反例）；`test_audit_findings.py` 是外部审计 F01–F07 的回归 |
| `data/rule_registry.json` | 规则注册表（3.6.0 回归 profile；冻结的 general-v3.1 注册表在 1_experiments/general_round_v3_1） |
| `examples/` | 统一入口声明与绑定文件示例 |
| `docs/` | [binding_guide.md](docs/binding_guide.md)、[environments.md](docs/environments.md)、[resource_policy.md](docs/resource_policy.md)、[changelog.md](docs/changelog.md)（工具版本变更）、[audit_status.md](docs/audit_status.md)（审查入口的阶段状态、不足与下一步）、[rule_registry_report.md](docs/rule_registry_report.md) |

## 环境

主环境 `/data1/tzh/envs/ka_main`（torch 2.10.0+cu128、Triton 3.6.0、gmpy2、python-flint），GPU RTX A6000（sm_86）。官方主线 Triton
（e50b186e）的构建在 `/data1/tzh/envs/triton_main`（DSL v2 的官方测试捕获与 W0 清单用它运行；运行脚本在
`1_experiments/dsl_v2/captures/run/`）。系统 `python` 是 2.7，不可用。详见 [2_tool/docs/environments.md](docs/environments.md)。

整理前的研究阶段代码（早期 Analyzer 框架、训练 bias 研究、覆盖与普查脚本及其测试）已从工具中删除，可从标签
`pre-reorg-20261009` 取回。
