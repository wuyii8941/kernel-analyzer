# 2 现有工具

工具 4.0（`dsl-v2` 分支，未发布、未冻结；冻结版本仍是 `general-v3.1` / 工具 3.1，见 [CURRENT.json](../CURRENT.json)）。

## 做什么

对一个 Triton kernel 的每次启动，从捕获的 IR 自动构造实数语义 G 的严格包围（参照 K_R），与实际输出 K 比较，
再用固定的方向规则与端点保守统计判断数值差异 e_num = K − G 的平均作用是否非零。有任务规格时（模式 B）另报 e_sem = G − f。

| 层 | 代码 | 说明 |
| --- | --- | --- |
| 捕获 | `2_tool/src/kernel_analyzer/reference_eval/capture.py` | 记录每次启动的参数、张量前后值与编译产物（TTIR / TTGIR / PTX） |
| 解析与映射 | `reference_eval/ttir_parser.py`、`ttir_mapping.py`、`rule_registry.py` | IR 解析；每个操作的规则（支持 / 声明前提 / 未建立 / 拒绝），注册表 [data/rule_registry.json](data/rule_registry.json)（Triton 3.6.0 回归 profile），报告 [pre-reorg-20261009:docs/rule_registry_report.md](docs/rule_registry_report.md) |
| 求值 | `reference_eval/ttir_eval.py`、`intervals.py`、`numbers.py`、`layouts.py`、`ptx_bits.py` | 区间语义（SumK / DotK、MPFR / Arb 初等函数）、控制流、原子与执行有效性、共享存储、NVIDIA / AMD 目标操作、位级 inline PTX |
| 核内定位 | `reference_eval/emulate.py` | 逐位模拟与逐节点精确替换（锁定 3.6.0 / sm_86 的下降规则） |
| 统计 | `reference_eval/analysis.py`、`detect.py`、`sensitivity.py`、`update_layer.py` | 方向规则 R1 / R5（固定均值）、R2 / R3（对齐），类内 Holm，契约 v3 的「无法判断」规则（`2_tool/scripts/essential/contract_v3.py`） |
| 入口 | `2_tool/src/kernel_analyzer/measure.py`（统一入口）、`check.py`（绑定入口） | 声明调用、输入来源、比较方式与预算，工具展开其余部分 |

DSL v2 的实现进度、每个增量的预期与结果见 [1_experiments/dsl_v2/README.md](../1_experiments/dsl_v2/README.md)；
外部审计的未关闭发现（F01–F07）见 [3_audits/README.md](../3_audits/README.md)，这些发现在修复前仍限制工具结论的可信范围。

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
- **不能**：注册表外的操作与未登记的 libdevice / inline asm（明确拒绝）；cuBLAS / ATen 写出的输出（没有参照）；warp specialization、
  asm 内访存；保存值、反向与优化器写入作为一个自动调用图。
- **已知局限**：自适应工作精度未实现；按 program 批处理未实现；有界路线只在校准中使用；外部审计 F01–F07 未全部关闭。

## 目录

| 路径 | 内容 |
| --- | --- |
| `2_tool/src/kernel_analyzer` | `measure.py`、`check.py` 与 `reference_eval/`（共 20 个模块） |
| `scripts/` | `measure.py`、`run_reference_analysis.py`、`enumerate_ttir_registry.py`；`dsl_v2/`（W0 清单、捕获插件、注册表与契约生成、两层对照、AMD 阶段）；`acceptance/`（结构验收 v1.1 适配器）；`essential/contract_v3.py`、`common.py`；`general/acceptance_run.py`（冻结检查）；`reference_eval_kernels.py`、`mutation_kernels.py`（测试用 kernel） |
| `tests/` | 46 个测试文件与 `data/`（官方 TTGIR fixture、TTIR 语料、反例） |
| `data/rule_registry.json` | 规则注册表（3.6.0 回归 profile；冻结的 general-v3.1 注册表在 1_experiments/general_round_v3_1） |
| `examples/` | 统一入口声明与绑定文件示例 |
| `docs/` | [binding_guide.md](docs/binding_guide.md)、[environments.md](docs/environments.md)、[resource_policy.md](docs/resource_policy.md)、[changelog.md](docs/changelog.md)（工具版本变更）、[rule_registry_report.md](docs/rule_registry_report.md) |

## 环境

主环境 `/data1/tzh/envs/ka_main`（torch 2.10.0+cu128、Triton 3.6.0、gmpy2、python-flint），GPU RTX A6000（sm_86）。官方主线 Triton
（e50b186e）的构建在 `/data1/tzh/envs/triton_main`（DSL v2 的官方测试捕获与 W0 清单用它运行；运行脚本在
`1_experiments/dsl_v2/captures/run/`）。系统 `python` 是 2.7，不可用。详见 [2_tool/docs/environments.md](docs/environments.md)。

整理前的研究阶段代码（早期 Analyzer 框架、训练 bias 研究、覆盖与普查脚本及其测试）已从工具中删除，可从标签
`pre-reorg-20261009` 取回。
