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
1. **契约 v2 分类器（执行方代码）细化**：归约 `neg_inf_row` 的 softmax / log_softmax 原先整条记 C；改为按行：全 −inf 的那一行 C（BASE-A2），
   其余行 A；`dims = all` 时行不是全 −inf，记 A（−inf 项概率为 0，规格 `softmax` 如此处理）。契约文本未改。
2. **精度不变性的条件级标签**按契约 v3 第 2 节原文：只有「存在语义偏离的元素」与「无语义元素」两种，另把超出 float64 噪声的异常单列复核；
   先前实现额外给出的条件级「未判定」「数值」标签删除（元素级计数照旧保留）。`test_contract_v3.py` 加 1 项（8 项通过）。
3. **补跑的 float64 / float32 配对**（第 2.2 节「同设备、同 API」）：attention（CUDA SDPA math float64、CUDA 手写 eager float64、CPU SDPA math
   float32）、optimizers（for_loop / foreach 的 CPU / CUDA float64）、packing（SDPA math、flex eager 的 float64）、rope（CUDA 参照 float64）、
   moe（CPU 循环 float32、CUDA 向量化 float64）；harness 增加环境键 `ka_main_f64`，2b 的候选集合与历史结果不变。
4. **规格问题 SPEC-ISSUE-1**（`docs/spec_issues_phase2.md`）：`spec_optimizers` O-D2 的 maximize 位置与 2.10 文档框不符；按第 8–9 节停止该条款
   裁决（`contract_v2.PENDING["optimizers"]`），规格未改。
5. **检测器 2.3**（`docs/detector_changelog.md`）：运行中发现上游（非 Triton）来源检测的两处漏判（按字节变化判写入；启动后取输入摘要），
   修正并加 4 项回归测试；检测阈值未变。模式 B 的 FR 与三类组合在 2.3 上重跑；F、P、E 与精度不变性不经过该检测，不受影响。
6. **模式 B 的归因诊断**（不作为 f）：超参数按 kernel 实际持有的 float32 值、GELU 常数按 TTIR 中的 float32 字面量重算一次 e_sem，只用来说明
   「K_R ≠ f_r」的来源；flex 的 `RCP_LN2 = 1.44269504` 与 BatchNorm 训练模式中编译期折叠的 1 − momentum（0.9）、n/(n−1)（1.1111111111111112）只按机制归因（生成的 kernel 源码），未重算。这些差异的契约归属为「契约外，待审阅」：
   契约 v3 第 1 节不允许用「在误差界内」回答是否违反，契约 v2 也没有关于编译期常数精度的条款。
7. **组合覆盖未完全达到**（`docs/composition_checklists_20261008.md`）：三份清单合并覆盖 9 / 12 条降级路径；κ = conditional、D_m 无区间扩展、
   无可比配对三条未在组合中出现（只有第 1 行的触发记录），交审阅方。声明变体：C1 保活关闭 / 8 个确认单位 / 未登记方向；C2 bf16 LayerNorm
   （1027 宽）；C3 原子 ticket 的持久化 kernel。
8. **P 的补充项**：gradcheck 只对 float64 eager；`|x| ≈ 1e30` 的条件中有限差分步长小于 float64 间距，gradcheck 记「不适用」（jvp 对 VJP 精确
   通过）；嵌入的 padding_idx 与 scale_grad_by_freq 按文档「梯度不是导数」，导数类检查记「不适用（文档约定）」；ReLU 子梯度集合检查按候选
   dtype 的 τ（bf16 只记录）。D2、D3 的对象（池化窗口、哨兵下标）不在第一档 15 个家族中，记「不适用」；D1 用 2b 的 amax / amin 并列检查。
9. **attention 的 F**：规格按 harness 给每个候选的显式 mask 求值（ATT-C1 取 SDPA 文档的左上对齐，flex / 手写实现收到同一 mask）；flex 的
   head_dim 72 在 2.10 无法编译（#164931），FR 记「未运行」。
10. **数值作用流**按附录 A 执行（提交 84fe39e 在测量之前）。
11. （审计补记）附录 A 声明的默认检测器在 20 个程序中都没有输出（`numerical_stream.json` 中为空），判定只用了方向规则。
12. （审计补记）seed 2000–2008 此前用于 `scripts/layer_table_m4.py` 的梯度生成（另一个程序、另一个生成器），附录 A「此前未用」对这 9 个
    seed 不准确。
13. （审计补记）附录 A 写「语义运行结束后」声明；检测器 2.3 的模式 B 重跑在声明提交之后完成。程序选择依据的 F、精度不变性与此前的模式 B
    结果在声明前已有，重跑只改变混合标记与耗时。
14. （审计补记）报告中的单位与分母更正（「条件」→「记录」，若干分母）、归因措辞更正（重算 5,360 个，机制归因 294,284 个）、κ = conditional
    的解释更正，见 `docs/audit_two_rounds_20261008.md`。

## 附录 A 数值作用流的 20 个程序（2026-10-08 声明，在任何作用测量之前提交）

**选取依据**：语义运行（F、精度不变性、模式 B 的 FR）结束后，从没有第 1–3 栏发现、也没有条款待审阅的实现中选；选择只看能否测量
（K_R 能否建立、输出是否由 Triton 写出），不看作用结果。排除：嵌入 max_norm（EMB-A1，第 2 栏）、SGD maximize（O-D2 待审阅）。
2b G7 的 20 个程序中 17 个能建立 K_R，原样保留（含 G7 的 GELU：任务书第 4 节要求它作为真实的数值作用继续研究）；不能测量的 3 个
（var_bf16：Welford 合并器被拒绝；scatter_add_bf16、embedding_bag_mean_bf16：输出不由 Triton 写出）换成 3 个同样来自无语义问题实现的
程序。语义运行中 Inductor float32 的 GELU 有「编译期常数舍入」造成的 K_R ≠ f_r（契约外，待审阅）；模式 A 的 e_num 以 kernel 自身的
TTIR（含这些常数）为参照，不受它影响。

| # | 程序 | dtype | 输入（每个单位） | 来源 |
| --- | --- | --- | --- | --- |
| 1 | layer_norm | bf16 | (8, 64) | G7 |
| 2 | rms_norm (eps 1e-5) | bf16 | (8, 64) | G7 |
| 3 | group_norm (4 组) | bf16 | (4, 8, 16) | G7 |
| 4 | batch_norm 训练模式 | bf16 | (8, 6, 12) | G7 |
| 5 | softmax | bf16 | (8, 257) | G7 |
| 6 | log_softmax | fp16 | (8, 257) | G7 |
| 7 | logsumexp | bf16 | (8, 1027) | G7 |
| 8 | sum（长行） | bf16 | (4, 4099) | G7 |
| 9 | mean（长行） | fp16 | (4, 4099) | G7 |
| 10 | cumsum | bf16 | (2, 1027) | G7 |
| 11 | gelu tanh | bf16 | (8, 257) | G7（GELU 作用） |
| 12 | silu(a) · b | bf16 | (8, 257) ×2 | G7 |
| 13 | gelu(a) · b | fp16 | (8, 257) ×2 | G7 |
| 14 | cross_entropy，label_smoothing 0.1 | bf16 | (16, 101) | G7 |
| 15 | 带 −1e4 上三角的注意力 softmax | bf16 | (2, 4, 33, 33) | G7 |
| 16 | AdamW 更新式（逐元素） | bf16 | (64, 16) ×3 | G7 |
| 17 | rms_norm(x + r) · 1.5 | fp16 | (8, 64) ×2 | G7 |
| 18 | RoPE rotate-half：x·cos + rot(x)·sin | bf16 | (2, 4, 11, 16)，表 (11, 16) | 新增（替换 var_bf16） |
| 19 | 梯度裁剪缩放：g · min(1, 1/(‖g‖₂ + 1e-6)) | bf16 | (64, 16) | 新增（替换 scatter_add_bf16） |
| 20 | layer_norm | fp16 | (8, 64) | 新增（替换 embedding_bag_mean_bf16） |

**输入生成**：与 G7 相同的生成器（`p2b_g6_g7.gen`：标准正态，float64 生成后转目标 dtype）；种子即单位编号。
**单位**：开发 32 个（seed 2000–2031），确认 64 个（seed 2032–2095）；这些 seed 此前未用于任何运行。
**量**：模式 A 的 e_num = K − K_R（工具 2.3，检测器阈值冻结），每个程序的输出 `out`，冻结统计层的方向规则（R1、R2、R3、R5）与默认检测器。
**判定**：`contract_v3.statistical_judgment`（S₀ = 2、N₀ = 64、n_min = 16；偏度 > S₀ 且 n = 64 时用 bootstrap-t 伴随结果）。
结论只说「平均作用非零（方向）/ 未确认 / 无法判断（原因）/ 未建立」；不由 3 个 seed 下总体结论；作用大小同时报告（相对 RMS）。
这些是实现的数值作用，不是缺陷判定；重要性另由阶段 B 的尺子回答。
**脚本**：`scripts/closure/numerical_stream.py`，结果 `results/closure/numerical_stream.json`。
