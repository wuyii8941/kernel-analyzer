# 执行协议 v3：保证链闭合与第一档全家族运行（2026-10-08，在任何新的捕获与统计判定之前提交）

依据：任务书「保证链闭合与第一档全家族运行（基于收束包 v2.0，2026-10-08）」；收束包 `phase2_closure_package_v2.0.tar.gz`
（SHA-256 fcecd0ef6870639a2f06dc6c90b6723f1d78ec07928889d82fa940cd519ce941，与任务书一致；包内 `SHA256SUMS` 20 个文件全部核对通过），
安装到 `specs/phase2/`（逐字复制，不改）；包内自检 `test_specs_phase2.py` 88/88 通过（ka_main，2026-10-08）。第一阶段规格仍是
`specs/phase1` v0.4。检测器 `detector-v2.2`；冻结的检测阈值不改。规格语义只能由审阅方修改：发现规格问题写 issue
（`docs/spec_issues_phase2.md`），不改代码。

## 1 顺序约束的状态

1. **A1 来源绑定修复**：已在 2a 完成（提交 17315bb，`detector-v2.2`）。2026-10-08 在本协议之前重跑回归：
   `tests/test_output_binding.py`（三个场景 × keep 开关 + Triton 写出仍绑定，7 项）、`scripts/essential/tests/`（A2 两处边界保护
   与触发检查，9 项）、`tests/test_reference_eval_guards.py`、`tests/test_reference_eval_equivalence.py`：28 项全部通过。新的捕获因此
   可以开始。A2（`classify.fr_category` 的 ok_elements = 0 → 未建立；`combine` 无可比配对 → 共有关系未建立）与 A3（FR 三个结论字段）
   在 2a 已入库（提交 6244f83）。
2. **契约 v2 即时生效**（`specs/phase2/contract_v2_all_families.md`、`ambiguities_phase2.md`）。2b 报告中所有「无定义」「合法性」判定
   对照契约重分（第 3 行的归类表）；契约未覆盖的情形标「契约外，待审阅」，执行方不再裁决。
3. **数值契约 v3 的参数与精度不变性判据**写在本协议第 2 节；之后才开始黑箱模式的语义分类与新的统计判定。
4. 规格可用的家族开放 F 与模式 B 的 FR；其余照常 E、P、模式 A。

## 2 数值契约 v3 的执行参数（运行前写定，不按结果调整）

### 2.1 三个问题分开
误差多大（参照包围）、平均作用是否非零（冻结的统计层 + 2.3 的前提规则）、是否违反任务要求（契约）。任何一栏不得用另一栏代替。
「在误差界内」「条件数」「eager 也一样」只能出现在「误差多大」的描述里。

### 2.2 黑箱模式的语义分类：精度不变性
按 `numeric_contract_v3.md` 第 2 节原样执行：d32_j = |K32_j − f_j|、d64_j = |K64_j − f_j|（f_j 取规格区间中点），r_j = d64_j / d32_j；
r < 2⁻¹⁶ 数值、r > 2⁻⁴ 语义、其间未判定；d32 = d64 = 0 一致；d32 = 0 而 d64 ≠ 0 异常单列；条件级「存在语义偏离的元素」要求
d64_j > 2⁻⁴⁰(1 + |f_j|)；非有限结果单列第四栏。
**适用的候选**（同一算法两种精度）：eager CPU float32 对 eager CPU float64、eager CUDA float32 对 eager CUDA float64（同设备、同 API、同参数）。
**不适用**：写死 dtype 的 kernel（SDPA flash / cuDNN 的 bf16、bnb、xformers、Liger、Unsloth 的固定路径）、bf16 专用路径、以及
Inductor（两种精度各自编译，不保证同一算法）——有 TTIR 的走 K_R 模式（K_R 对 f 的精确比较），没有的记「未判定」。
实现：`scripts/essential/contract_v3.py::precision_invariance`。

### 2.3 「无法判断」的三条触发（统计前提）
从 2a 之前已入库的校准（`results/reference_eval/calibration_equivalence.json`，每格 4000 次；`docs/statistics_calibration_20261006.md`
第 2 节，含 bootstrap-t 的校准）读出：

| 参数 | 值 | 读出依据 |
| --- | --- | --- |
| S₀（偏度绝对值） | **2** | 点单位下覆盖率跌破 0.93 的格只出现在对数正态（偏度约 3.3）：n = 16 为 0.894–0.902，n = 32 为 0.916–0.924，n = 64 为 0.923–0.936；偏度 ≤ 0.6（正态）与 t₃ 的全部格 ≥ 0.944；偏度 1.1–1.75 的格只记录了错误等价率（0.06–0.08）与 bootstrap-t 覆盖率（0.940–0.950），没有 t 区间覆盖率跌破 0.93 的记录。边界取在已校准的 1.75 与 3.3 之间，与 2026-10-06 预先写定的报告规则第 4 条（|偏度| ≥ 2）一致 |
| N₀（单位数） | **64** | 对数正态下 n = 64 仍有格跌破 0.93（0.923），所以「n ≤ N₀」取 64；n > 64 没有对数正态的校准，按前提成立处理并在报告中注明 |
| n_min（确认单位） | **16** | 校准的最小样本量；正态与 t₃ 在 n = 16 时覆盖率 0.944–0.961、非零误报 0.040–0.049 |
| bootstrap-t 的适用 | 偏度 > S₀ 且 n ≤ N₀ 时：n = 64 用 bootstrap-t（校准错误率 0.068 ≤ 0.07）；n < 64 记「无法判断（分布前提）」（n = 16 的校准错误率 0.095 > 0.07，n = 17–63 没有 ≤ 0.07 的校准） | 同上 |

方差为零（全部端点相同）→「无法判断（退化）」（冻结统计层的 `_sample_guard` 已如此）；n < n_min →「无法判断（样本）」。
实现：`scripts/essential/contract_v3.py::statistical_judgment`，在冻结统计层的规则记录之上执行，不改冻结字段。

### 2.4 四栏
1 文档明确条款的违反；2 声明解释下的差异；3 未定义情形的实现约定；4 数值失败（非有限、溢出、越界写入）。
「不是缺陷」只能引契约条款。

## 3 第 1 行：保证链
`results/closure/chain_table.json`（由 `specs/phase2/chain_table_template.json` 复制填写），用包内 `validate_chain_table.py` 校验到 0 problems。
每条降级路径附一条实际触发的记录（仓库内文件）；没有现成记录的，用最小构造输入经生产代码触发一次（`scripts/closure/trigger_degrades.py`），
记录入库 `results/closure/degrade/`。规则 → 函数对应表更新为 `docs/rules_to_code_20261008.md`（在 2026-10-06 版基础上补箭头 (0)、(2)–(4)，
每个函数名与测试名经校验器核对存在）。

## 4 第 2 行：结论复审
按 `readjudication_procedure.md`：在 docs/ 与 results/ 的结论性文字中检索九个模式，每一处一行复审表（`docs/readjudication_20261008.md`）；
语义归类用 2.2 的比值（已有 float32 / float64 双精度且规格可用的直接算，没有的记「未判定」）；平均作用由统计层回答；「不是缺陷」只引契约
条款；G7 的 GELU 作用保留为真实数值作用。历史文件不改，新版本报告替换（`docs/essential_bugs_phase2b_record_v2_20261008.md`）。

## 5 第 3 行：输入契约
逐家族、逐条件标注五类之一（`results/closure/contract_classification.json` + 摘要表）；契约外 →「契约外，待审阅」。七项「文档待抓取核对」：
从锁定版本（torch 2.10.0、transformers 4.57.3）的文档原文（已安装包的 docstring，即在线文档的来源；HF 的 lr_lambda 另附源码中的
公式位置作为对照）与规格 docstring 逐条比对，差异报告 `docs/doc_check_seven_items_20261008.md` 交审阅方；不改规格。

## 6 三类组合验证
按 `composition_checklist_template.md` 各一份（`docs/composition_checklists_20261008.md`）：纯计算 linear → GELU → 残差（G6 已有的组合重跑一次，
带绑定记录编号）；保存值 / 分支：带 activation checkpoint 的 RMSNorm → 注意力片段（模式 A）；状态更新：AdamW 短序列（初始化 → 正常 →
零梯度 → grad=None → 恢复），f 来自 `spec_optimizers`。三张清单合并后覆盖第 1 行表中的每一条降级路径。

## 7 第一档家族的运行
- 四组：E（K − K_eager）；F（K − f，不经 K_R）；P（规格文件的 prop_*、gradcheck、`torch.func.jvp` 对 VJP 的点积、D1–D3）；FR（模式 B：
  K_R 对 f；e_num 与 e_sem）。
- 已有的 2b 运行（`.cache/essential/p2b/**`、`results/essential/phase2b/**`）保存了每个候选收到的输入（由条件与 seed 确定地重建）与输出；
  F 在这些输出上用规格补算（规格在候选收到的、按候选 dtype 舍入后的输入上求值）。缺 float64 的黑箱候选补跑 float64 以做精度不变性。
  模式 B 的 FR 需要新的捕获（Triton 候选）。
- 单位：契约 × 实际实现路径 × 输入条件 × 调用 / 状态场景 × 方法；候选沿用 G4 登记，补「采用的契约 / 声明约定」两列
  （`results/closure/candidate_conventions.json`）；声明约定（ATT-C1、ROPE-C1、MOE-C1…C4、OPT-A1）先登记，只对照候选自己声明的
  那种，不同约定之间的差异记第二栏。
- 判定：有 TTIR 时 K_R 对 f_r 精确比较；黑箱时精度不变性；数值作用由冻结统计层 + 2.3；非有限记第四栏。所有计数附分母，未建立、
  不支持、超时单列。
- 数值作用流：语义运行结束后、作用测量之前，从未发现语义问题的实现中声明 20 个程序（写入本协议附录 A 后提交），在独立确认输入
  （seed 2000–2095，此前未使用）上测平均作用；不以 3 个 seed 作总体结论。

## 8 新发现、9 停止条件
按任务书第 8、9 节原文执行。规格 docstring 与文档原文不符 → 停该条款的裁决，交审阅方。

## 10 偏离记录
（运行中追加。）
