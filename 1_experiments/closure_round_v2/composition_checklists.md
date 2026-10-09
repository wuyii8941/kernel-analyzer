# 三类组合验证清单（任务书第 6 节；模板 `1_experiments/specs/phase2/composition_checklist_template.md`；2026-10-08）

脚本 `pre-reorg-20261009:scripts/closure/compositions.py`，记录 `1_experiments/closure_round_v2/results/compositions/*.json`（下文「记录」即其中的文件名）。工具版本 2.3
（混合来源检测修正，见协议 v3 第 10 节）。统计判定按 `contract_v3.statistical_judgment`（S₀ = 2、N₀ = 64、n_min = 16）。
「声明变体」是在同一程序上改一个事先说明的条件（记录器设置、单位数、dtype / 宽度、实现方式），不是另造的程序。

---

## C1 纯计算：linear → GELU(tanh) → 残差

组合名称：`y = x + gelu_tanh(x Wᵀ + b)`，同时返回 `h = x Wᵀ + b`；Inductor float32；x (8, 64)，W (64, 64)，b (64)。
类型：纯计算。
输入与绑定记录编号：`C1_main.json`（1 个开发 + 32 个确认单位，seed 0–32）；Inductor 生成 `extern_kernels.addmm`（cuBLAS）写 h，
Triton `triton_poi_fused_add_gelu_0` 读 x 与 h 写 y。
需要的规格：有（`1_experiments/specs/phase2/spec_base_ops.py` v0.1（规格包 v0.2）：`linear`、`gelu_tanh`，加法精确）——模式 B。

| 箭头 | 前提在本例中如何被检查（记录编号） | 规则实际执行的函数 | 本例触发了哪些降级（记录编号；无则写「未触发」） |
| --- | --- | --- | --- |
| 0 绑定 | 记录器保活开启时 y 按存储实例绑定到 Triton 写入（`C1_main` notes：无未绑定输出）；保活关闭的声明变体中 y 的配对不可证 | `TritonLaunchRecorder`、`storage_ids`、`check.run` 中的 `binding_unconfirmed` | **配对不可证 → y 未建立**（`C1_variant_keepalive_off.json`：`outputs_binding_not_established = ['y']`） |
| 1 包含性 | TTIR 覆盖完整（`ttir_coverage_complete = [True]`）；y 的完整比例 1.0（κ = complete）；上游检测：y 读取 cuBLAS 写的 h（`mixed_non_triton_sources = ['L0:triton_poi_fused_add_gelu_0:in_ptr1']`），h 的捕获值作为精确输入进入 K_R | `evaluate_sequence`、`rule_for`、`kernel_coverage`、`torch_intermediates` | 未触发 κ 降级；**上游非 Triton 值进入 K_R → e_sem 为混合**（不是语义判据） |
| 2 残差与投影 | e_num 按定向区间减法；方向 R1、R2、R3、R5 已登记 | `residual_interval`、`apply_direction_rules` | **方向来源未登记 → 不出正式结论**（`C1_variant_unregistered_direction.json`：生产代码以 `KeyError: 'unregistered_direction'` 拒绝，没有结构化的降级记录） |
| 3 均值区间 | 32 个确认单位 ≥ n_min；偏度未超过 S₀ | `_summarize`、`apply_holm`、`statistical_judgment` | **样本不足**（声明变体 8 个确认单位：`C1_variant_n8.json`，全部规则「无法判断（样本）」） |
| 4 结论 | 契约 v2：activations（gelu_tanh、高斯输入）类 A；组合本身不是契约 v2 的家族；h 不由 Triton 写出 | `classify_condition`、`four_column`；输出无可用元素的保护（`check.run` 的 not-Triton 判定，对应 `fr_assess` 的 ok_elements = 0） | **ok_elements = 0 → h 未建立**（`C1_main` notes：`outputs_not_written_by_triton = ['h']`） |

最终结论：
- y（16896 个元素）：差异——e_num 的平均作用 R5「非零（负）」，R1–R3「未确认」（n = 32）；语义——e_sem 为混合（含 cuBLAS 对 h 的舍入），
  不出语义结论；黑箱 F：0 / 16896 超出 τ32。组合本身契约外（`C1_main.json`：X），结论只记录；四栏均无条目。
- h：未建立（cuBLAS 写出，不参与数值 / 语义分解；黑箱 F 由第一档 matmul 家族的 F 覆盖：0 超出）。

---

## C2 保存值与分支：activation checkpoint 下的 RMSNorm → RoPE（按 position id 取 cos / sin 行）→ 因果注意力

组合名称：`h = checkpoint(rms_norm)(x, w)`；q、k、v 由 h 投影（cuBLAS）；`cos = cos_t[pos]`、`sin = sin_t[pos]`；RoPE；
因果 softmax 注意力；前向与反向（反向中 checkpoint 重算 RMSNorm）。Inductor float32；B 2、L 12、D 32、2 头。
类型：保存值与分支。
输入与绑定记录编号：`C2_main.json`（1 + 32 个单位，seed 0–32；声明：每个单位的输入尺度取对数正态，σ = 1.5）；11 个 Triton launch。
需要的规格：无（模式 A）。

| 箭头 | 前提在本例中如何被检查（记录编号） | 规则实际执行的函数 | 本例触发了哪些降级（记录编号；无则写「未触发」） |
| --- | --- | --- | --- |
| 0 绑定 | 6 个输出均按存储实例配对（无未绑定输出） | 同 C1 | 未触发 |
| 1 包含性 | 11 个 launch 的 TTIR 覆盖完整；h、cos_gathered、p、dx、dw 的完整比例均为 1.0；p、dx、dw 读 cuBLAS 的中间值（混合来源 1 / 3 / 6 个）；PTX 零填充前提已核验（`assumed:masked lanes zero-filled`） | 同 C1 | 主运行未触发；**未识别语义 → 拒绝**（声明变体：bf16 LayerNorm、1027 宽隐藏态，`C2_variant_layernorm_bf16.json`：`tt.reduce@37 / @77: unrecognized reduction combiner`，h、dx、o 的写入程序中止；dw 未建立） |
| 2 残差与投影 | 方向 R1、R2、R3、R5 已登记 | 同 C1 | 未触发 |
| 3 均值区间 | 32 个确认单位；偏度检查 | 同 C1 | **方差为零 → 无法判断（退化）**（cos_gathered 为纯取数，e_num 端点全部相同）；**分布前提不成立 → 无法判断（分布）**（dx：R2、R3、R5，偏度 > 2 且 n = 32 ≤ N₀） |
| 4 结论 | 契约 v2 未列出该组合：`classify_condition` → X | `classify_condition`；not-Triton 判定 | **契约未声明 → 只记录**（`C2_main` contract：`X 契约外，待审阅`）；**ok_elements = 0 → o 未建立**（o 由 cuBLAS 写出） |

最终结论（模式 A，只有差异一种结论；契约外，只记录）：h 的 e_num 平均作用 R2、R3「非零（正）」；p、dw「未确认」；dx「无法判断（分布）」；
cos_gathered「无法判断（退化）」；o 未建立。

---

## C3 状态更新：AdamW 序列 初始化 → 正常 → 零梯度 → grad=None → state_dict 保存 / 重载

组合名称：`torch.optim.AdamW`（lr 1e-3、betas (0.9, 0.999)、eps 1e-8、wd 0.01），`torch.compile(opt.step)`；参数形状 (5, 9)、(7,)、(3, 4)；
第 4 步参数 1 的 grad = None；第 5 步前保存并重载 state_dict。
类型：状态更新。
输入与绑定记录编号：`C3_main.json`（每步一组，seed 0–2：1 + 2 个单位）；A → B → A 对照同一文件 `A_B_A`。
需要的规格：有（`1_experiments/specs/phase2/spec_optimizers.py`：`adam_step`，在该步实际收到的参数与状态上求值）——模式 B。

| 箭头 | 前提在本例中如何被检查（记录编号） | 规则实际执行的函数 | 本例触发了哪些降级（记录编号；无则写「未触发」） |
| --- | --- | --- | --- |
| 0 绑定 | 5 步的全部输出按存储实例配对 | 同 C1 | 未触发 |
| 1 包含性 | 各步 TTIR 覆盖完整、完整比例 1.0；全部输出读 ATen 写的中间值（step 计数与偏差修正，混合来源 3–21 个） | 同 C1 | 主序列未触发；**κ = unestablished**（声明变体：持久化融合 AdamW kernel 从原子计数器领取 tile，原子返回值决定地址，`C3_variant_persistent_atomic_ticket.json`：p、m、v 全部 `store through a not-established address`） |
| 2 残差与投影 | 方向已登记 | 同 C1 | 未触发 |
| 3 均值区间 | 每步 2 个确认单位 < n_min | 同 C1 | **样本不足 → 无法判断（样本）**（全部输出） |
| 4 结论 | 按契约 v2 文本，AdamW 在 optimizers 行的合法域内（`C3_main.json` 未写入分类字段）；grad=None 一步中参数 1 不被写 | not-Triton 判定；`combine` | **ok_elements = 0 → 参数 1（grad=None 步）未建立**；无可比配对：未触发（候选一侧为「未建立」，`combine` 的共有关系不形成） |

最终结论：5 步的全部输出黑箱 F 0 超出 τ32（例如第 2 步 param0 0 / 135）；e_sem 为混合（不出语义结论）；平均作用「无法判断（样本）」；
A → B → A：重载后的一步与不中断的一步在全部 9 个输出上逐位相同（0 / 135、0 / 21、0 / 36 …）；grad=None 的参数 1 未建立（未被写）。
四栏均无条目。

---

## 三份清单合并后的降级覆盖（通过标准）

| 箭头 | 降级 | 组合中实际触发的记录 |
| --- | --- | --- |
| 0 | 配对不可证 → 未建立 | C1 声明变体（保活关闭） |
| 1 | κ = conditional | **未在组合中触发**（只有第 1 行的触发记录 `1_experiments/closure_round_v2/results/degrade/arrow1_kappa_conditional.json`）。更正（通用能力轮核实，`2_tool/tests/test_general_rules.py`）：check.run 中区间未定的谓词（select 与 scf.if）取两分支的并集，结果仍是完整包围；κ = conditional 只来自把加载钉在捕获值上的 `pin_loads` 选项，check.run 不使用它——这条路径在 check.run 的任何组合中都不会出现，原写「来自区间未定的控制条件」有误 |
| 1 | κ = unestablished | C3 声明变体（原子 ticket）；C2 声明变体的 dw |
| 1 | 未识别语义 → 拒绝 | C2 声明变体（bf16 LayerNorm 的 Welford 合并器） |
| 2 | D_m 无区间扩展 → 未建立 | **未在组合中触发**（需要 8 bit 状态的实际写入映射；只有第 1 行的记录 `pre-reorg-20261009:results/reference_eval/blind_test_v2/phase3/phase3_adamw_history_actual_write.csv`，2026-10-04 的旧工具记录，早于检测器 2.3） |
| 2 | 方向来源未登记 → 不出正式结论 | C1 声明变体 |
| 3 | 方差为零 → 无法判断（退化） | C2（cos_gathered） |
| 3 | 分布前提不成立 → 无法判断（分布） | C2（dx） |
| 3 | 样本不足 → 无法判断（样本） | C1 声明变体（n = 8）；C3（n = 2） |
| 4 | ok_elements = 0 → 未建立 | C1（h）、C2（o）、C3（grad=None 的参数 1） |
| 4 | 无可比配对 → 共有关系未建立 | **未在组合中触发**（只有第 1 行的构造记录 `1_experiments/closure_round_v2/results/degrade/arrow4_no_comparable_pair.json`） |
| 4 | 契约未声明 → 只记录 | C1、C2（组合不在契约 v2 中） |

**合并覆盖 9 / 12 条降级路径；3 条（κ = conditional、D_m 无区间扩展、无可比配对）未在三个组合中出现**，通过标准未完全达到。
这 3 条在第 1 行（`1_experiments/closure_round_v2/results/chain_table.json`）有实际触发记录。是否接受、或指定能自然触发它们的组合，由审阅方决定。

本轮运行中由组合发现的工具问题：检测器 2.2 的上游来源检测按「启动前后字节变化」判定写入，C1 的 y 因缓存分配器交回装着同一结果的缓冲区
而被漏标为「纯 Triton」；修正为按参照自身的写入判定（检测器 2.3，`2_tool/tests/test_upstream_sources.py` 4 项，修正前其中 3 项失败）。
