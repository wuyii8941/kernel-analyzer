# 案例与证据地图

本页将现有材料按“实现身份 → 成因推导 → 三阶段测量与判断 → 修改及训练验证”连接。
**同一行表示一个研究对象或机制族，不代表所有列来自同一协议。**
配置不一致的证据明确分开，不能靠同名模型或算子拼成一次验证。
本次是源码和结果核对，不是新 GPU 复现。

## 1. 主要机制与训练验证

| 模型与训练位置 | 成因推导/修改依据 | 更新与训练证据 | 当前怎样讲 |
|---|---|---|---|
| Qwen 中 Liger fused CE，输出权重 dW | 分块累加误差进入 tied weight gradient；提高累加精度检查该来源 [L1] | 历史 4096 步直接作用及独立配对 loss 记录；统一 v2 测量另见 [P1] | 有成因、更新和 loss 材料；同状态直接测量与配对轨迹分开报告 |
| 小型 GPT-2 中 Liger fused CE，tied embedding | 同类累加精度比较；不是上述 Qwen 模型 [L2] | 全参数自然训练；两条 4096 步数据流，一条续到 10000 步 | loss 分叉；不支持持续恶化。延续过程没有逐步直接/反馈分解 |
| 同一小型 GPT-2 的新冻结数据流 111 | 相同 BF16/FP32 dW 累加精度比较 [L3] | 新的 4096 步配对运行；验证 loss 差 −0.00946 | 轨迹不同得到复现，但质量差方向未复现；不支持稳定收益或损害 |
| Phi-4，lm-head backward dX → final norm | 实际 backward 公式、舍入与状态配对；同 AdamW 随机舍入 [P2] | 32 步来源干预、另行 warm-state 4096 步记录；v2 显示相对 update 缩小 [P1,H1] | 来源干预与长程后果都要讲，但不是一次同状态同协议实验 |
| Qwen3-1.7B，lm-head backward dX → final norm | 同类实际 backward 的传播差异 [P1,H1] | cold 短测未检出、另行 warm-state 长测有方向和配对 loss | 说明 optimizer 状态改变表现；不能永久标成“AdamW 抵消”，也不是独立于 Phi 的新公式 |
| Qwen，v-proj；Mamba，in-proj | 矩阵乘误差与输出舍入的分解，按来源分别修改 [S1] | 多轮自然/固定条件测量、v2 缩放及各自轨迹 [P1,H1] | 保留 bias 与 loss 材料；旧“FP32 后再转 BF16”不等于已经去掉输出舍入 |
| Qwen，layer-27 saved-P softmax backward | 保存状态影响实际 backward；严格正负响应 [R1] | 自然实现差与人工正负响应分别测量；历史 loss 记录 [H1] | response 不对称是实测机制证据，不能把人工响应的数值直接当自然差异的均值 |
| Qwen3-VL，SiLU backward | 非线性导数及 AdamW 正负响应 [R1] | 4096 步直接作用弱、反馈有方向；loss 差很小 [H2] | 反馈维持与自然直接 bias 分开；微小 loss 分叉不写成稳定质量损害 |
| Qwen，layer-23 attention backward → q-proj | 保存状态与 attention 区域的计算关系，区域修改 [A1] | 历史闭合区域对照与配对轨迹 [H1] | 可保留区域级解释，不能强行归因到一个 Triton kernel |

这里的“成因材料存在”不表示每个记录都已证明相同的非零均值命题。
传播公式、其非零条件、干预和对应轨迹应连同具体结果阅读。缺少的连接明确保留，
不以统计标签替代。

## 2. 后续发现、补充预测与对照

| 对象 | 已有事实 | 在主线中的位置 |
|---|---|---|
| DeepSeek normalization 与 attention-projection backward | 各自新案例 cold 测量 update 缩小约 13.68% / 10.69%；warm 后不足 1%，同 warm 参数重置 moments 后大效应恢复 [D1] | optimizer 状态作用的证据；不是两个位置的底层误差成因已被完整证明 |
| 上述两个 DeepSeek 位置 + Phi loss/CE 对照 | 每项四条 32 步输入流；实际/反馈方向均 4/4，直接方向均 0/4；窗口平均 loss 差区间均跨零 [D2] | 单 checkpoint、指定参数范围的后果；不冒充独立初始化全参数训练或稳定质量降低 |
| Liger dW，同 FP32 不同加法顺序 | 后 16 个输入中 15 个沿预测方向 [O1] | 非精度切换的补充机制预测；没有同实验的长程 loss 结论 |
| DeepSeek layer-35 attention dV | 参考更新坐标中的确认和三阶段补测属于不同轮次 [M1] | 有条件方向的补充；不能借其他 DeepSeek 位置的 loss 补齐它 |
| Phi normalization 0543 | 单行擦边方向在多重比较后未确认 [H1] | 未决候选；不能改成确证阴性，也不能计入强成因案例 |
| Llama、Ministral lm-head | 历史同族长测 [H1] | 跨模型补充，不自动增加一个独立成因 |
| Gemma-4、Phi loss/CE、冻结集合中的阴性与未决 | 有效阴性、零差异、反馈或无法执行记录分别保留 [B1] | 对照与范围限制，不为增加正例而改标签或删除 |
| Phi `lm_head dX` 新实际写入重采 | 同一 32 状态中，AdamW 公式 update 的 RMS 比为 5.47%，BF16 参数实际写入差异逐位为零 [N1] | 证明 optimizer 计算值与已提交参数变化必须分开；不回写旧结果 |
| DeepSeek 两个位置的新实际写入重采 | normalization / attention projection 的写入 RMS 比为 53.04% / 46.02%，相对缩放区间均完整越过 −1% [N2] | 证明这两个 cold-start effect 已提交到声明参数；属于已有开发案例的统一管线复核 |
| Liger 新实际写入重采 | 公式 update RMS 0.653%；BF16 实际写入 RMS 4.936%；写入层 additive/residual 方向未确认 [N3] | 存储舍入扩大总差异但未确认强共同方向；不覆盖历史机制与长程协议 |

## 3. 为什么不能报一个没有限定的“案例总数”

| 集合 | 原结果的计数/角色 | 不能解释成什么 |
|---|---|---|
| 四模型首轮覆盖 | 1,562 个具体输出位置 [C1] | 1,562 个 bias 成因与 loss 都已验证的案例 |
| 历史完整 F+B/参考/参数/32 步 | 六个历史记录，见旧登记 [H3] | 六个同 optimizer 的长期结论 |
| 旧统一 AdamW 短筛 | 15 行；两个确认正例，0543 未决 [H4] | 只有两个有研究价值的案例，或通用准确率 |
| 五案例 profile v2 | 五个对象；三个方向分支分别测量 [P1] | 五个新的独立成因 |
| 16 项冻结 benchmark，等价 v2 重新分析 | 9 个 material 标签、3 个固定集合等价标签、3 个 RMS 超范围、1 个未决 [B1] | 9 个已经完成成因推导与 loss 验证的案例；摘要 PASS 也不是原空间严格上界保证 |
| 历史 4096 审计 | 23 个主矩阵 ID、301 行；43 行旧 bias+loss 标签，其中 3 直接后期、8 整段直接、32 反馈 [H1] | 43 种独立机制，或覆盖后来全部实验的“当前总数” |
| 新 Liger 自然训练 | 普通 2048 步、高压力两条 4096 步、其中一条续至 10000 步 [L2] | 新增几种数学成因，或多条独立 10000 步复现 |

历史审计还保留 105 条更宽的 outcome-relevant 记录、45 条未决，以及缺少后期窗口的
结果。旧标签表示当时规则；本页不重写它们。新全参数 Liger 数据不在该旧审计的总数中。

## 4. 最容易混淆的版本与状态

- **旧 32 步 AdamW 轨迹/来源干预**：moments 从零开始后正常演化。
- **五案例 profile v2**：JSON 明写 `ZERO_AT_EVERY_INPUT_STATE`，每个冻结输入
  单独从零 moments 测量；不是一条连续 32 步训练。输入窗口也是固定集合，
  不是 32 个独立预训练运行。
- **warm 状态对照与 4096 步重放**：按各自 checkpoint、输入和参数范围解释。
- **10000 步**：只延续高压力小型 GPT-2 的第一条数据流，不能替换历史 Qwen
  的直接作用分数。
- **等价 v1/v2** 与 **profile v1/v2** 是不同工具的版本；旧数据上的新计算叫重新分析，
  不叫新前瞻发现。

## 5. 执行身份核查

2026-09-05 检查本机 Liger 0.7.0：
`/data1/tzh/envs/liger/lib/python3.10/site-packages/liger_kernel/ops/fused_linear_cross_entropy.py`
中，CE 使用 Triton，dW 分块结果通过
`grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()`
累加，最后转回权重 dtype。

训练脚本只改变 `accum_dtype=None` 与 `torch.float32`。
因此目前可以把新实验描述为“Liger 混合路径中的 dW 累加精度对照”，不能仅凭库名
写成“Triton 内部累加错误”。这是当前源码检查；没有拿当前文件摘要补造历史运行的
二进制或 kernel 身份。其他案例的实际 backend 以各自 capture 为准，缺项仍未知。

## 6. 可核查来源

以下结果全部保留原文件；本页新增的是连接和范围说明，不是新结果。

- **[L1]** [Liger 成因分析](../results/property/bias_formation_final/liger_formation_analysis.json)；
  [4096 直接测量](../results/property/declared_persistent_4096/liger_fused_ce.json)；
  [配对 loss](../results/property/paired_loss_4096/liger_fused_ce.json)。
- **[L2]** [训练脚本](../scripts/run_liger_single_boundary_collapse.py)；
  [普通设置](../results/property/single_point_collapse_v1/protocol.json)；
  [高压力设置](../results/property/single_point_collapse_v2/protocol.json)；
  [10000 步结果](../results/property/single_point_collapse_v2/full_10000_summary.json)；
  [实验说明](liger_single_boundary_collapse_experiment.md)。
- **[L3]** [事前协议](../results/property/training_numerical_analysis_v1/training_utility_protocol.json)；
  [第三条数据流汇总](../results/property/training_numerical_analysis_v1/training_utility_summary.json)。
- **[P1]** [五例原始汇总](../results/property/training_bias_profile_v2/five_case_summary.json)；
  [测量说明](five_case_training_bias_profile_v2.md)。
- **[P2]** [Phi 传播分解](../results/property/bias_formation_final/phi_transport_decomposition.json)；
  [配对干预](../results/property/bias_formation_final/intervention_results/phi_mm_transport_pairing.json)；
  [同 AdamW 随机舍入说明](phi_adamw_source_intervention.md)。
- **[N1]** [新采集](../results/property/training_numerical_analysis_v1/recapture/phi.json)；
  [实际写入复算](../results/property/training_numerical_analysis_v1/recomputed/phi_parameter_write.json)；
  [公式 update 复算](../results/property/training_numerical_analysis_v1/recomputed/phi_proposed_update.json)。
- **[N2]** [normalization 新采集](../results/property/training_numerical_analysis_v1/recapture/deepseek8b_seq256_backward_1714_in_out_ptr0.json)；
  [实际写入复算](../results/property/training_numerical_analysis_v1/recomputed/deepseek_norm_parameter_write.json)；
  [attention projection 新采集](../results/property/training_numerical_analysis_v1/recapture/deepseek8b_seq128_backward_1256_out_ptr0.json)；
  [实际写入复算](../results/property/training_numerical_analysis_v1/recomputed/deepseek_attn_parameter_write.json)。
- **[N3]** [Liger 新采集](../results/property/training_numerical_analysis_v1/recapture/liger.json)；
  [实际写入复算](../results/property/training_numerical_analysis_v1/recomputed/liger_parameter_write.json)；
  [公式 update 复算](../results/property/training_numerical_analysis_v1/recomputed/liger_proposed_update.json)。
- **[S1]** [按来源修改的推导与协议](source_aligned_repair.md)。
- **[R1]** [奇偶推导](effective_antithetic_symmetry.md)；
  [响应分解](../results/property/joint_bias_formation_v1/mu_parity_decomposition.json)；
  [saved-P 逐状态响应](../results/property/extended_unified_profiles_v1/saved_p_response.json)；
  [SiLU 逐状态响应](../results/property/extended_unified_profiles_v1/silu_response.json)。
- **[A1]** [attention 推导](l23_qproj_tile.md)；
  [区域干预](../results/property/bias_formation_final/intervention_results/qwen_l23_attention_state.json)。
- **[D1]** [新案例第一批](../results/property/training_bias_profile_v2/prospective_batch_1/summary.json)；
  [第二批](../results/property/training_bias_profile_v2/prospective_batch_2/summary.json)；
  [状态对照](../results/property/optimizer_condition_benchmark_v1/summary.json)。
- **[D2]** [四条输入流后果](../results/property/independent_consequence_v1/summary.json)。
- **[O1]** [FP32 顺序实验](../results/property/liger_fp32_chunk_order_v1/summary.json)。
- **[M1]** [参考坐标确认](../results/property/bias_oracle_recovery/confirmation/result.json)；
  [三阶段补测](three_mechanism_profiles.md)。
- **[B1]** [16 项原始汇总](../results/property/generalization_benchmark_v1/summary.json)；
  [等价 v2 重新分析](../results/property/generalization_benchmark_v1/equivalence_v2.json)；
  [Gemma 接入](../results/property/generalization_benchmark_v1/gemma4_method_bridge_result.json)。
- **[C1]** [覆盖与分母](coverage_table_v1.md)。
- **[H1]** [历史长程机器审计](../results/property/declared_persistent_4096/all_bias_case_audit.json)。
- **[H2]** [Liger/SiLU 历史长测说明](liger_silu_long_horizon_recheck.md)。
- **[H3]** [历史案例登记](../cases_flash_style.md)。
- **[H4]** [旧短筛](direct_persistence_screen.md)；
  [多重比较结果](../results/property/direct_persistence_v4/multiplicity.json)。
