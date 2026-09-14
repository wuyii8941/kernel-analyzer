# 逐案例因果证据核对

本页核对的是“代码 → 数值误差 → 非零 bias 的理由 → 干预”，不是重新认证
历史统计标签。来源清单的分类已经完成；结论不是每个案例根因都已闭合，而是明确
区分闭合、部分闭合、响应机制和仅测量四类证据。
完整待核对清单见 `results/property/case_causal_audit_v1/inventory_corrected.json`。
其中 551 个去重测量位置和 315 条历史/角色记录有交集，不是 866 个独立问题。
去重后的科学问题主表由 `scripts/build_scientific_case_closure.py` 生成到
`results/property/case_causal_audit_v1/scientific_case_closure.json`。

## 去重后的科学结论

按“同一计算问题不因模型、层、形状或重复协议而重复计数”，当前整理为 9 个问题组。
这里的 9 不是 9 个已证明根因：只有 AdamW8bit 在声明协议内达到
“数学递推来源—实际状态传播—针对性干预—独立训练改善”的端到端闭合。

| 问题组 | 当前最强结论 | 尚不能声称 |
|---|---|---|
| AdamW8bit moment 量化 | 跨步丢失残差可重构 moment 差；正确读回改善写入与两批配对训练 loss；坐标打乱造成构造性数值失败 | 自然训练必然崩溃、跨模型普遍成立、生产优化器已完成 |
| Liger fused linear CE dW 累加 | 一个真实末期状态的差异由相同分块乘积的有限精度加法精确重构；精度与顺序干预有效 | 已证明总体非零均值，或该局部项充分导致全部 loss 差异 |
| MM/GEMM 输出与累加 | 选定 fixed-input 条件下可分离 kernel arithmetic 与 output rounding；不同模型位置表现不同 | 所有 MM 共享一个根因；所有条件差都形成持久 bias |
| softmax saved-state backward | 保存/重构概率改变解析 VJP；严格正负重放证明 AdamW 有偶响应 | 自然误差分布自身具有已证明的非零均值 |
| SiLU backward | 不同导数求值进入真实梯度；严格正负重放证明早期 optimizer 偶响应 | 已解释自然 SiLU residual 为何产生非零 bias |
| attention-state → q-proj 区域 | 公式和 S/K/joint/sham 干预闭合到 `S_bwd` 语义区域 | 已定位制造 `S_bwd` 差异的唯一上游算子 |
| fused RoPE / position scaling | 相同输入差异与 optimizer-state 条件效应明确；低位置对照排除 scaling 为唯一根因 | 已隔离底层算术根因；已单独区分 moments 与 step counter；已有 loss 后果 |
| 99 个 common-input SiLU/RMS backward 位置 | 同一自动流程可测全空间 update 差异 | 每个非零位置都有非零平均 bias 或唯一算术根因 |
| 31 个 reference-graph 区域 | 能测实际训练区域的 update effect | 区域差异就是某一个 Triton kernel 自身的差异 |

因此，最准确的总括是：**框架已对全部记录给出证据范围；根因研究有一条端到端闭合、
数条局部或语义区域闭合，以及大量尚未进入根因干预的测量案例。**历史上的
`PASS_FLASH_STYLE_CASE` 只表示当时四项 gate 通过，不能自动改写为“唯一数值根因和
训练损害均已证明”。

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
| Qwen layer-23 attention → q-proj | 实际 bmm_76 左输入 S_bwd；Gq=S_bwd K，dW=GqᵀH | 公式定位传播路径，区域恢复定位 S_bwd 的作用；未说明唯一上游算术如何产生非零均值 | S-only、K-only、joint、sham 对照；区域归因不等于唯一 kernel 根因，区间跨零不等于全空间相等 |
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
