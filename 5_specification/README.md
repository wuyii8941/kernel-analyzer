# 规格（specification）

规格只存放在这里：任务规格 f 的方法、来源登记、合同、可执行参照、证据与审阅记录。2026-10-15 从 `1_experiments/specs`
迁来，同时换成 v2.4。规格语义只经审阅方修改，执行方不改。

## 1 规格在工具里管什么

工具比较三个量：K 是设备输出，Gπ 是程序按其声明语义的精确值，f 是任务要求的值。

| 差异 | 回答什么 | 参照从哪来 |
|---|---|---|
| K − Gπ（数值偏差） | kernel 是否按它自己的代码在算：舍入、累加精度、近似指令、归约次序 | DSL v2 rc3 从 TTIR 自动提升，不需要本目录。定义见 [design_rc3](../1_experiments/dsl_v2/design_rc3/README.md)，代码在 `2_tool/src/kernel_analyzer/reference_eval/` |
| Gπ − f（语义偏差） | kernel 写的是不是任务要的计算：公式、归一化、掩码、状态语义 | 本目录 |

- 精度审查只用 Gπ。没有 f 时只报告 K − Gπ，并写明「无语义参照」。
- Gπ 不是把一切换成实数：π 声明的量化、位语义、允许集合都保留。

## 2 f 怎样产生

三个对象，逐级可追溯：

```
来源原文      SOURCES.json                     版本钉死的文档，按条款编号
  → 数学合同  contracts/*.yaml                 条款、合法域、profile、输出与保证、例子、误读
  → 可执行参照 specification_v2/algorithm_specs/  兑现合同的代码
```

- **条款依据**：只能是原文条款编号（如 `PCE.REDUCE`），或一条登记过的决定（如 `CE.R_A`）。
- **决定的两种状态**：
  - 已审阅；
  - 需任务绑定：任务负责人的选择，工具不能替用户选。例如 RANKMEAN 的有效 token 等权、CE 加权 label smoothing 的 R_A 读法。
- **原文没规定的行为**：只能记为未规定、允许集合或任务选定的约定，不能为了让某个实现通过而事后补读法。
- **保证标签**：每个输出带一个，取 exact、rigorous_enclosure、allowed_set、declared_approximation 之一。
- **比较结果**：只有成立、矛盾、未建立三种。区间不相交才判矛盾；相交只说明相容。
- **结果类别**：value、domain_undefined、invalid_input、unsupported、not_established、implementation_error。
- **三种「通过」分开记**：合同内容已审阅、参照实现已验证、本次查询可用。前一项推不出后一项。
- **证据**：关系检查、独立复算、ONNX Runtime 对照、历史误读重建都按条款登记。它们不决定条款状态，但可信的矛盾会暂停相关条款。

完整口径见两份原文：[METHOD.md](docs/METHOD.md)（方法定稿）和 [CONTRACT_PROTOCOL.md](docs/CONTRACT_PROTOCOL.md)（审阅后的实施口径 v1.1）。

## 3 现状（2026-10-15）

| 层 | 状态 |
|---|---|
| 可执行参照 v2.4 | 31 个角色都有参照代码，另有 SPARSE、STATEIO 两条路径规格；222 项测试通过；8 个角色修过 v1 的错误 |
| 合同 | CE、ADAMW、RANKMEAN 三份，内容已审阅（reviewed_for_implementation）：15 条条款、6 个 profile、29 个例子、20 条变异要求。2026-10-16 起有检查器与适配器：例子全部一致，变异全部被拒绝；规格 v3 正在按 `docs/PLAN_v3.md` 推进 |
| 其余 28 个角色 | 只有参照代码，没有合同。按方法，对它们的语义判断只能报「未建立」 |
| 工具接入 | 尚未接入：审查入口的 `spec=` 收的是一个返回 float64 的函数，与合同无关 |

所以眼下没有任何 f 能称为「已认证的参照」。角色清单见 [roles.md](docs/roles.md)；已知问题、待决事项与版本沿革见
[status.md](docs/status.md)。

## 4 目录

| 路径 | 内容 | 来源 |
|---|---|---|
| `docs/METHOD.md` | 方法定稿 | 规格包 2026-10-15，原样 |
| `docs/CONTRACT_PROTOCOL.md` | 实施口径 v1.1 | 审阅交付包，原样 |
| `docs/roles.md`、`docs/status.md` | 角色清单；现状、已知问题、待决事项、版本沿革 | 本次整理 |
| `SOURCES.json` | 来源登记：每个来源的版本、URL、条款编号 | 审阅交付包 |
| `contracts/*.yaml` | 三份审阅后的合同 | 审阅交付包 |
| `contracts/draft/` | 被取代的合同草案、适配器与草案检查器。检查器会给出假绿（审阅 F1–F5）；适配器是下一步实现的起点 | 规格包，原样（只改了一处相对路径） |
| `tools/`、`tests/` | 合同结构校验、例子对照参照、校验器回归 | 审阅交付包 |
| `specification_v2/` | 可执行参照 v2.4（`algorithm_specs/phase1–3`）、舍入 oracle、关系检查与随机验证、目录与计数、`SPECIFICATION_v2.md`、`THEORY.md` | 规格包，原样 |
| `evidence/` | ONNX Runtime 对照、v1 误读重建、THEORY 撤回部分的核对、Liger CE 对照 | 规格包，原样 |
| `reviews/2026-10-15/` | 审阅报告、实施任务书、审阅方的运行记录与复现脚本、被审规格包的顶层文件 | 审阅包，原样 |
| `run_checks.py` | 一条命令跑全部检查 | 本次整理 |

## 5 运行

```bash
export HOME=/data1/tzh/kernel-analyzer/.cache TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp
/data1/tzh/envs/liger/bin/python 5_specification/run_checks.py
```

- **环境**：需要 Python ≥ 3.10 与 numpy、scipy、sympy、mpmath、pyyaml。本机 `liger` 环境齐全；`ka_main` 缺 pyyaml。
- **用时**：约 25 秒。
- **行为**：
  - 每项检查单独运行，任何一项失败即非零退出；
  - 会写文件的脚本在临时目录里重算，再与提交的记录比较，不写仓库。
- **2026-10-15 的结果**：
  - 16 项必需检查全部通过；
  - 参照测试共 222 项；
  - 关系检查、目录计数、随机验证、phase1 独立复算四份记录，重算后与提交的一致；
  - 例子对照的结果与审阅方的一致；
  - ONNX Runtime 对照未运行（没装 onnxruntime）。

## 6 阅读顺序

1. 本页；
2. [METHOD.md](docs/METHOD.md)，一页；
3. `contracts/CE.yaml`：一份完整的合同长什么样；
4. [roles.md](docs/roles.md)：31 个角色各自的来源与状态；
5. [status.md](docs/status.md)：已知问题与待决事项；
6. 需要细节时再看：[CONTRACT_PROTOCOL.md](docs/CONTRACT_PROTOCOL.md)、`reviews/2026-10-15/REVIEW.md`、`specification_v2/SPECIFICATION_v2.md`。
