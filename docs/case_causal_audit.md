# 逐案例因果证据核对

## 当前统一根因账本

为避免历史审计表与新增案例角色混在一起，当前主线的去重问题组和机器复算结果统一见
[当前根因闭环账本](root_cause_closure_current.md)。该账本由
`scripts/build_current_root_cause_closure.py` 从保留的原始记录生成，补入 Gemma GELU、
Gemma RMS 归约顺序、Granite Top-k/专家累加，以及 Gemma 平方和绑定失败，并明确区分
端到端闭环、局部来源闭合、响应机制、开放分支与阴性控制。旧表在下文保留用于追溯，
不应与当前问题组数量相加。

本页核对的是“代码 → 数值误差 → 非零 bias 的理由 → 干预”，不是重新认证
历史统计标签。来源清单的分类已经完成；结论不是每个案例根因都已闭合，而是明确
区分闭合、部分闭合、响应机制和仅测量四类证据。
完整待核对清单见 `results/property/case_causal_audit_v1/inventory_corrected.json`。
其中 551 个去重测量位置和 315 条历史/角色记录有交集，不是 866 个独立问题。
去重后的科学问题主表由 `scripts/build_scientific_case_closure.py` 生成到
`results/property/case_causal_audit_v1/scientific_case_closure.json`。

## 去重后的科学结论

下方表格是历史审计快照，共 9 个条目（7 个科学问题组和 2 个覆盖集合）；它保留原始
证据链和当时的边界，不代表当前问题组总数。当前去重后的 12 个科学问题组及 2 个覆盖
集合请以本页上方链接的统一账本为准，不能把历史 9 与当前 12 相加。
AdamW8bit 在声明协议内连接了“数学递推来源—实际状态传播—针对性干预—独立训练改善”。
这不等于已证明平均 bias 是 loss 差异的唯一原因；方差、坐标结构与非线性响应的贡献仍需区分。

| 问题组 | 当前最强结论 | 尚不能声称 |
|---|---|---|
| AdamW8bit moment 量化 | 跨步丢失残差可重构 moment 差；正确读回改善写入与两批配对训练 loss；坐标打乱造成构造性数值失败 | 自然训练必然崩溃、跨模型普遍成立、生产优化器已完成 |
| Liger fused linear CE dW 累加 | 相同分块乘积的 FP32 加法顺序是局部来源；长度 64 和 256 的互斥确认银行复现了预先声明方向，并进入参数梯度与零矩 AdamW 首步摘要 update | 仍未保存原坐标实际写入；1024 步顺序对照的 loss 差异回到零，因此不支持该局部项已足以造成长期质量差异 |
| MM/GEMM 输出与累加 | 选定 fixed-input 条件下可分离 kernel arithmetic 与 output rounding；不同模型位置表现不同 | 所有 MM 共享一个根因；所有条件差都形成持久 bias |
| softmax saved-state backward | 114,688 行重构概率有非零行和缺陷，重归一化可恢复行和；新增 1024 步声明 warm-state 轨迹中，saved-P 修复进入 q/k 梯度并造成非零 update 与参数/轨迹 non-identity | loss gap 跨步变号，累计 update 近扩散型；尚未证明自然总体 bias、持久方向或 material quality loss |
| SiLU backward | 在同一调用前输入上，AST 受检的显式指数 source variant 改变了 gate-gradient，并贯穿 gradient、moment、update 和 write | 已闭合一个局部 source-choice 响应；尚未闭合自然输入总体 mean bias，也未分开 sigmoid、表达式次序和最终 cast 的贡献 |
| attention-state → q-proj 区域 | 公式和 S/K/joint/sham 干预闭合到 `S_bwd`；其中一个局部贡献已定位为 key RMSNorm+RoPE 融合延迟 BF16 中间物化 | 该局部根因解释整个区域；其余 upstream-logit 和 residual-stream 贡献来自唯一算子 |
| fused RoPE / position scaling | 相同输入差异与 optimizer-state 条件效应明确；低位置对照排除 scaling 为唯一根因 | 底层算术 source、moments 与 step counter 尚未分别隔离；没有该问题组的 loss 结果 |
| 99 个 common-input SiLU/RMS backward 位置 | 同一自动流程可测声明范围内的 update 差异 | 它们是覆盖集合；不能把每个非零位置升级为非零平均 bias 或唯一算术根因 |
| 31 个 reference-graph 区域 | 能测实际训练区域的 update effect | 区域差异就是某一个 Triton kernel 自身的差异 |

因此，最准确的总括是：**AdamW8bit 有较强的来源传播、干预与训练结果链；attention
和 softmax 各有已隔离的局部来源；Liger 的同精度顺序来源已在互斥确认银行复现，但
没有原坐标写入或长期质量闭环；SiLU 和 GELU 各有局部 source-choice 响应；其余为
条件来源、语义区域、响应机制、阴性控制或仅测量。**历史上的
`PASS_FLASH_STYLE_CASE` 只表示当时四项 gate 通过，不能自动改写为“唯一数值根因和
训练损害均已证明”。

这里的“闭环”需要按结论强度区分：若终点是“根因 → 定向干预 → 配对轨迹不再相同”，
AdamW8bit 和 Liger 都有证据；若还要求在独立训练中得到方向稳定、超过预设门槛的
质量改善，目前只有 AdamW8bit。Liger 的 4096/10000 步结果证明轨迹分叉，但 loss
差异会随窗口改变符号，不能写成持续恶化或稳定收益。局部根因闭合（如 attention
和 softmax）也不自动等于自然总体 bias 或训练质量闭环。

## 全来源记录已经逐条分类

`scripts/build_exhaustive_case_causal_audit.py` 将当前四个来源清单合并为
`results/property/case_causal_audit_v1/exhaustive_source_records.json`。866 条记录
全部具有代码位置、数值差异、非零 bias 理由、干预和证据判断五个字段，没有用
`PENDING` 填充未知项：

| 来源 | 记录数 | 正确解释 |
|---|---:|---|
| 有效测量位置 | 551 | 具体执行位置；不是独立根因数 |
| 历史长程审计 | 301 | 包含阳性、阴性、未决和重复协议 |
| 旧主线角色 | 6 | 证据入口；不是新实验 |
| 旧案例复审 | 8 | 历史 gate；底层证据决定最终口径 |

551 个测量位置中，265 个在声明固定集合上观察到零 update 差异；286 个为非零。
这 286 个记录都只能由总能量证明“存在差异”，不能由此证明非零平均 bias 或
唯一根因。301 条历史记录中，105 条含历史方向或后果标签，但历史总表本身不提供
来源均值的数学证明；152 条为对照或其他后果记录，44 条未决。这样的分类是审查
结论，不是将已有阳性降为阴性：更强的机制证据需在下节逐案连接。

## 已检查的机制证据与剩余连接

| 具体比较 | 代码与数值误差 | 为什么形成 bias：证据实际支持什么 | 干预及未闭合部分 |
|---|---|---|---|
| Phi lm-head dX，历史配对干预 | MM residual 经 final normalization backward 传播 | 保持 residual 能量的对应关系变化会改变梯度方向结构；不是由局部能量直接决定 | 16 状态 row permutation；解析传播重构相对误差 0.3244–0.6005，尚非唯一物理来源的精确重构 |
| Qwen Liger fused CE t128，历史 formation 协议 | 分块 dW 累加/转换；历史执行源码仍待逐项绑定 | calibration 阳性不能代替 confirmation；后者区间跨冻结 margin | 保留 `UNRESOLVED_CONFIRMATION_MARGIN_CROSSED`；不能借后来 GPT-2 结果补签这个协议 |
| Qwen layer-23 attention → q-proj | 实际 bmm_76 左输入 S_bwd；Gq=S_bwd K，dW=GqᵀH | 区域恢复定位 S_bwd；另有一个已隔离局部来源：融合使 key RMSNorm+RoPE 的 BF16 中间物化延后，沿 K→S_bwd→Gq→dW 传播 | eager-like materialization 与完整 key-forward 修复一致、reduction-schedule 对照不成立；它解释约 51.5% 所测 carrier，不是整个区域的唯一根因 |
| Qwen seq128 v-proj，conditional rounding | `run_mm_source_aligned_repair.py:SourceAlignedMMRepair` 拦截 `extern_kernels.mm`；`precision.py:source_aligned_mm_output` 构造替换 | 固定输入下确定性舍入相对随机舍入集合的条件效应；不是跨训练状态共同方向的证明 | 摘要记录 16 条条件下各 candidate-effect 标签为 CONDITIONAL_BIAS；仍需读取完整记录核对最后转换的残差保持误差与执行身份 |

前三项来源分别为
`results/property/bias_formation_final/phi_transport_mechanism.json`、
`liger_formation_analysis.json`、`qwen_l23_attention_mechanism.json`，并检查了
Phi 配对干预和 `results/coverage/cases/l23_qproj_attention_state_region.json`。
第四项同时阅读了执行脚本、精度实现、
`results/property/conditional_debias/qwen128_vproj.json` 和
[来源修改说明](source_aligned_repair.md)。这些是代码和保存记录审查，不是本轮 GPU 复现。

## 本轮新增的必要核验：ROUNDING_ONLY 的实际输出

代码先构造随机舍入值 `base`，再执行

```python
delivered = (base.float() + terms["kernel"]).to(actual_low.dtype)
```

因此，保留原算术残差是一个需要检查的数值性质，不是模式名称保证的性质。
最后一次转换可能引入附加误差。执行脚本已经记录
`kernel_residual_preservation_error`；接下来必须读取对应运行中的实际值，
才能决定是否支持严格的单来源隔离。
完整记录复算已确认：Qwen128 的 128 次重复中有 7 次非零，均在状态
`qwen3-1p7b-0109`；最大 L2 为 2.107342389479072e-8，最大坐标绝对值为
1.4901161193847656e-8。它否定“逐次精确保留”，但不单独证明这么小的附加差异
改变了方向或训练结论。需将其作为干预余项保留，不能事后设阈值把它叫作精确零。
复算脚本为 `scripts/audit_mm_conditional_sources.py`，带每次重复记录的
结果为 `results/property/case_causal_audit_v1/mm_conditional_sources.json`。
Qwen64 与 Mamba 各检查了 256 次 JOINT 重复；JOINT 有意移除算术残差，
该项非零不是其干预失败，不能套用 ROUNDING_ONLY 的保持要求。
此外，该 runner 拦截外部 MM，不能仅凭生成图中出现这个调用就归因为 Triton 内部。

随机舍入的条件均值公式适用于声明参考值与数学概率模型；实际概率计算和有限随机数
实现仍有数值条件。局部条件中心化不能推出非线性 backward/AdamW 后中心化。
固定输入下的非零条件效应，也不能直接替代跨输入分布的非零平均 bias。

## 批量核对规则

### 批量记录的来源核验

`scripts/audit_coverage_causal_records.py` 已读取旧覆盖快照中全部 421 个带原始
记录的去重位置，421/421 文件可读取。输出
`results/property/case_causal_audit_v1/coverage_causal_links.json`
保存每个位置的 runtime boundary、参考比较范围、参数写入协议和原坐标统计阶段。
这核实了来源链接，不等于独立重放，也不等于 421 个根因完成。
后续新增的 130 个位置未包含在这一旧快照核验中。进一步追溯后确认：它们不是
130 个 signature 测量，而是 31 个 signature 位置与 99 个补充家族位置
（64 个 SiLU、35 个 normalization）。此前的统称不准确，以此处分拆为准。

31 个 signature 位置已经由 `scripts/audit_signature_causal_records.py`
逐项核对 catalog → analysis → raw 的 case ID 与 task ID，全部匹配。
记录统一声明 `REFERENCE_GRAPH_ENDPOINT_SUBSTITUTION`、
`same_local_operands=false`、`includes_possible_upstream_differences=true` 和
`single_kernel_source_attribution=NOT_ESTABLISHED`。
这些结果不是相同输入单 kernel 归因，不可借 `LOCAL_IMPLEMENTATION_SUBSTITUTION`
这个较宽的 contrast 名称覆盖更具体的范围字段。
结果位于 `results/property/case_causal_audit_v1/signature_causal_links.json`。
余下 99 个位置也已沿 `family_execution_audit_20260907_all_three_complete.json`
追查：DeepSeek128 SiLU 36、Qwen64 SiLU 28、DeepSeek256 RMS backward 35。
原始文件 99/99 可读取。与 31 个区域替换不同，这 99 项明确使用相同调用前输入，
分别为 `COMMON_OPERAND_SOURCE_CHECKED_SILU_BACKWARD` 和
`COMMON_OPERAND_SOURCE_CHECKED_RMS_BACKWARD`。输出记录在
`results/property/case_causal_audit_v1/additional_family_causal_links_v2.json`。
初版链接表的 `release` 曾填家族别名，v2 分开 `audit_family` 与执行身份；
原始记录和统计数值未改。

核对 `src/kernel_analyzer/silu_backward_reference.py`：候选实际公式为
`dy * up * sigmoid(gate) * (1 + gate*(1-sigmoid(gate)))`，源代码 AST 校验只
覆盖指定 gate-gradient 输出，融合计算的另一个输出不替换。FP32 native 参考
改变求值实现，但没有逐一隔离 exp、除法、融合运算及最终写入的贡献。
因此支持“这个已审核输出的实现差异”，不证明每个位置的非零平均误差来源。

核对 `src/kernel_analyzer/rms_backward_reference.py`：公式包含原有梯度累加量
`a + g*r - x*sum(g*x)*r**3/width`，其中 `g=(p0+p1)*w`。
参考保留明确分组、验证布局和已存 inverse RMS 的 FP32 表示。
不能把它简化为一个纯 sum 算子，更不能由公式正确推出舍入误差有非零均值。
现有比较同时涉及归约与后续算术求值；单项来源干预及 bias 非零条件仍需另有证据。

例如 DeepSeek128 `backward:659:in_out_ptr0` 的记录明确为保存输入上的 NLL
FP64 重算再 BF16 写入，且不包括可能的上游差异；原始实际 AdamW 写入使用
FP32 master、每输入零 moments。这个位置不能仅凭长 fused kernel 名称推断
整个 fused 区域或所有 softmax backward 的 bias 成因。

### saved-P 与 SiLU：响应证据不能改名为自然来源证明

已读 `docs/effective_antithetic_symmetry.md` 的两项机制说明及
`results/property/extended_unified_profiles_v1/{saved_p_response,silu_response}.json`。
两个结果均明确记录 `ANTITHETIC_RESPONSE_REMAINDER`，使用 4096 维 CountSketch，
不是自然 candidate-reference 的完整原坐标均值证明。

- saved-P：说明中的 head permutation 未支持“特定 head 配对是主要解释”；
  正负 gradient 残差实验支持 AdamW 响应不为奇函数。说明报告响应偶分量能量
  99.51% 集中于前两步，不能写成每步持续同强度的来源。底层对应运行仍待进一步绑定。
- SiLU：候选为 decomposed AOT VJP，参考为 `aten.silu_backward`；正负响应
  与后续反馈证据不等于解释了自然 VJP 误差为何有非零期望。说明报告偶分量
  超过 99.99% 来自前两步。应单列“响应机制已测”和“来源均值机制未闭合”。

奇偶分解是恒等式；只有证明或测得某项非零，才构成相应现象的证据。
人为正负 residual 的分布不能无说明地替代自然实现误差分布。

每条记录保留独立协议、执行身份和原标签；同族可以共享公式说明，但不能共享未经
核实的干预结论。有效测量但未做根因实验的记录明确写“只有测量证据”，不能
因为 lack of evidence 改为阴性，也不能写成根因已闭合。本轮已经完成来源记录分类
与去重问题组的证据分级；尚未闭合的科学问题直接保留在主表中，不用“审查进行中”
掩盖它们。

## 根因收敛顺序

后续不再按测量位置数推进，而按下面的可证伪干预推进。每项都必须保持相同局部输入，
一次只改变一个数值选择，并同时保留 joint 与 sham 对照。

| 优先级 | 问题组 | 下一项最小干预 | 完成标准 |
|---|---|---|---|
| P0 | MM/GEMM | accumulation arithmetic、最终 cast 与 joint 的三因素拆分 | 在独立状态上说明哪一项产生已测 conditional effect，并报告 interaction |
| P0 | SiLU backward | 已完成显式指数 source variant 的同输入 factorial；保留 sigmoid 求值、表达式次序、最终 cast 的分量拆分 | 局部 source-choice 已闭合；自然 residual 的总体 mean bias 仍需独立状态和分量中间量 |
| P0 | softmax saved state | 在独立状态中重复同一 source measure，再只恢复一致 saved probability | source residual 与 gradient/update 变化使用同一状态和同一方向定义 |
| P1 | fused RoPE | RoPE 中间物化与 position scaling 分开；optimizer 对照固定 step counter | 分开算术来源与 optimizer-state response，不再用 high/low position 代替来源干预 |
| P1 | Liger | 长度64/256互斥确认已完成；两边均为 FP32，仅改变 dW chunk addition order | 已确认同精度顺序可复现局部/摘要方向；原坐标写入与长期 loss sufficiency 仍不作声称 |
| 已完成子项 | attention q-proj | key RMSNorm+RoPE BF16 中间物化 | 保留为局部根因；整个复合区域继续标为多来源 |
| 局部诊断 | softmax saved state | 重构概率的行和与保存统计一致性检查 | 重归一化不能单独隔离来源；需要真实实现干预及 gradient/write 传播确认 |
| 已完成局部诊断 | Liger | 同为 FP32、只改变 chunk addition order，并在长度64/256确认 | 方向与零矩 AdamW 首步摘要 update 已复现；仍无原坐标写入与长期质量结论 |
| 已完成主项 | AdamW8bit | blockwise moment 残差读回 | 只补外部条件确认，不再在相同数据上重复挑机制 |

99 个 common-input 位置和 31 个 reference-graph 区域只用于选择上述代表问题，不能
逐位置自动升级成根因实验。机器表中的 `next_root_cause_test` 与本节同步生成和检查。

## 2026-09-15 追加核验后的停止边界

本轮对 saved-P 做了实际修复后的 32 步预检和 1024 步声明 warm-state 运行。它补上了
“局部根因是否真的进入梯度和参数写入”这一缺口，但没有改变其他问题组的证据等级。
随后重新运行 `scripts/build_root_cause_exhaustion_audit.py`：除该新增轨迹证据外，
各问题组仍各缺一个能区分竞争解释的观测。缺失观测分别是独立状态总体、同输入分量
干预、上游/残差流互补干预、固定 step counter 的状态对照或自然 bias 的预注册终点；
这些量不能从已保存的聚合统计反推。

因此当前可以安全写出的闭环数量为：

- 以“根因到声明轨迹 non-identity”为终点：AdamW8bit、Liger 和 saved-P 三组；
- 以“独立训练中方向稳定且超过质量门槛的改善”为终点：仅 AdamW8bit；
- attention 的 key RMSNorm+RoPE 只闭合了整个复合区域中的一个局部来源；
- MM、SiLU、RoPE 其余竞争解释没有被现有数据唯一排除，不能用更多模型位置代替。

这不是把未完成案例改成阴性，而是对保留证据做完可复现的离线审计后，明确哪些新
观测仍是必要条件；在没有这些观测前，不再从公式或旧聚合文件推断唯一根因。
