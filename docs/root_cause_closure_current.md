# 当前根因闭环账本

本表按科学问题组去重。它把‘已经知道差异来自哪里’、‘已经证明形成了 bias’和‘已经看到训练后果’分开记录。模型位置、层号和训练状态是同一问题组内的证据，不重复计数。

当前共 12 个科学问题组；其中 1 个连到声明的 material loss improvement，6 个至少完成了局部来源闭环，2 个是阴性控制；另保留 2 个测量覆盖集合。其余开放分支明确列出，不用‘有差异’代替根因。

| 问题组 | 当前根因结论 | 训练后果 | 还缺什么 |
|---|---|---|---|
| `adamw8bit_moment_quantization` | END_TO_END_CAUSAL_CHAIN_UNDER_DECLARED_PROTOCOL；blockwise quantization discards a state residual which otherwise enters the next moment recurrence | two eight-pair 1024-step comparisons support a material loss improvement for correct residual readback; coordinate permutation causes constructed numerical failure | external model or checkpoint confirmation; not another intervention chosen on the same data |
| `liger_fused_linear_ce_dw_accumulation` | LOCAL_ARITHMETIC_ROOT_AND_DECLARED_UNIT_MEAN_CLOSED_LOSS_CAUSAL_SUFFICIENCY_OPEN；finite-precision addition order and accumulation precision of identical dW chunk products | paired trajectories separate; 4096/10000-step records do not support monotone degradation or collapse | only needed for a stronger claim: preserve original-coordinate writes or run a predeclared longer paired loss comparison |
| `mm_gemm_output_and_accumulation` | CONDITIONAL_SOURCE_ISOLATION_NOT_UNIVERSAL_MM_ROOT；case-specific finite-precision sources: output rounding only in Qwen128, kernel plus output rounding in Qwen64/Mamba, and kernel arithmetic in Phi | historical trajectories separate, but robust cross-state direction is case-dependent | same-operands factorial separating accumulation arithmetic, final cast, and their interaction on an independent state bank |
| `softmax_saved_state_backward` | LOCAL_ARITHMETIC_ROOT_CLOSED_AND_DECLARED_TRAJECTORY_NON_IDENTITY_NATURAL_BIAS_OPEN；same-call evidence isolates an inconsistency: BF16 saved scaled scores are combined with FP32 maximum and denominator from the pre-materialized calculation | a new 1024-step declared warm-state trajectory has nonzero direct update effect and finite parameter/loss non-identity; loss gaps change sign and no monotone degradation is shown | independent state-bank sampling with a predeclared loss endpoint, then restore only the consistent saved state before measuring gradient and update direction |
| `silu_backward_evaluation` | SOURCE_CHOICE_RESPONSE_CLOSED_NATURAL_BIAS_OPEN；finite-precision source choice in the sigmoid/SiLU derivative expression at a generated Triton gate-gradient endpoint | direct persistence is weak; later separation is feedback-sustained and the loss difference is small | only needed for a stronger natural-bias claim: separate sigmoid approximation, expression order and final cast on an independent state bank |
| `attention_state_to_q_projection_region` | ONE_LOCAL_ARITHMETIC_ROOT_CLOSED_WHOLE_REGION_MULTI_SOURCE；the S_bwd input to bmm_76 carries the regional difference; one isolated contributor is fusion-delayed BF16 materialization in layer-23 key RMSNorm plus RoPE | a paired trajectory exists under the historical protocol | treat the closed materialization contributor as a completed subcase; do not claim a unique root for the whole region unless remaining upstream-logit contributors are separately intervened |
| `fused_rope_position_scaling` | STATE_RESPONSE_ESTABLISHED_SOURCE_ROOT_OPEN；same-input implementation evaluation difference; position scaling alone is ruled out as the unique explanation | not measured | same-operands cast/materialization factorial, plus a state comparison holding step counter fixed while changing moments |
| `gemma_gelu_backward_evaluation` | SOURCE_CHOICE_RESPONSE_CLOSED_NATURAL_BIAS_OPEN；tanh evaluation and expression spelling in a generated Triton tanh-GELU backward product | not measured as an independent full training result | only needed if a natural GELU bias claim is desired: predeclared independent source distribution and training endpoint |
| `gemma_rms_feature_reduction_order` | NEGATIVE_SOURCE_CONTROL_NO_ROOT_CAUSE；FP32 feature-reduction order | not measured | none for this negative control; a different RMS source would require a new predeclared intervention |
| `granite_router_topk_selection` | NEGATIVE_CONTROL_FIXED_SUITE_IDENTITY；selection tie-order variation with unchanged selected set | no difference observed; no population or quality claim | none unless a different selection semantic (changed selected set, NaN, or non-tie scores) is explicitly studied |
| `granite_moe_expert_contribution_order` | SOURCE_CLOSED_FIXED_SUITE_NATURAL_BIAS_OPEN；FP32 expert-contribution accumulation order | not measured | if promoted beyond a coverage confirmation, isolate expert accumulation from routing and run an independent state-bank confirmation |
| `gemma_bound_square_sum` | EXECUTION_SOURCE_MISMATCH_UNRESOLVED；intended bound square-sum endpoint was not the actual executed kernel source | not measured | bind the actual executed endpoint and re-run the declared square-sum comparison before making any root-cause claim |

## 本轮新增的可核对结论

Liger 的同精度加法顺序来源已经有互斥的长度 64 和 256 确认：确认半区分别为 14/16 和 11/16 沿预先声明方向，参数梯度与零矩 AdamW 首步分支也有记录支持。它仍使用摘要更新量，1024 步配对 loss 差异回到零，因此根因来源闭合，但训练质量后果没有闭合。

SiLU 的 source-factorial run3 在同一调用前输入和同一 gate-gradient 输出上，显式指数求值相对候选改变了 local、gradient、moment、update 和 write profile；实际执行位置经过函数 AST 与调用前输入核对。该结果闭合一个局部求值来源，但不代表整个 SiLU 家族或自然输入总体存在平均 bias。

Gemma GELU 的同一位置、相同输入和独立确认状态上，显式指数重构会改变 gradient/update profile，而 native tanh 与融合乘加表达式在记录的 profile 中一致。这把 GELU 的来源缩小到求值选择的响应差异，但没有证明自然输入总体存在稳定 mean bias。

Gemma RMS 的 endpoint-by-endpoint 归约顺序干预说明，早期同时替换造成的较大差异来自上游传播，不能归因给第二个 RMS endpoint。Granite Top-k 的 24 状态结果则是完整阴性控制：合法的相等分数索引排列没有改变选中集合、值、梯度、写入或 loss。

## 使用边界

所有固定集合结果只覆盖各自声明的状态、参数和实现边界。‘源已闭合’不自动表示自然总体 bias 已闭合；‘训练轨迹不相同’不自动表示质量持续恶化。后续若要把开放分支升级，必须新增能区分表中竞争解释的观测，而不是从已有聚合量反推。

逐问题组的离线耗尽审计见 `results/property/root_cause_closure_v1/exhaustion_audit.json`；其中每个问题组都明确列出当前可推出的结论和仍缺少的区分性观测。

## 全量来源记录审计

当前账本同时对仓库保留的 866 条来源记录做自动归属：551 条有效测量位置、301 条历史矩阵记录、8 条旧案例复审和 6 条角色记录。它们按算子族和审查状态保留在机器字段 `source_record_inventory` 中；其中只有具备独立同输入来源干预的记录才进入上面的根因问题组。其余记录仍可作为覆盖、阴性、历史后果或未决证据，不能把非零范数直接称为已知根因。

机器结果：`results/property/case_causal_audit_v1/root_cause_closure_current.json`。

## 冻结 benchmark 与算子族的逐项根因边界

冻结 benchmark 共 16 个案例；这些案例均有测量结果，但当前没有任何一个仅凭 benchmark 的 AOT endpoint 替换就升级为新的根因闭环。逐项机器记录见 `generalization_benchmark_frontier`。

| benchmark 案例 | 模型 | 算子族 | update 结果 | 根因状态 |
|---|---|---|---|---|
| `deepseek8b_seq128_backward_1665_in_out_ptr0` | deepseek8b | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `deepseek8b_seq128_backward_664_in_out_ptr0` | deepseek8b | LOSS_HEAD_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `deepseek8b_seq256_backward_1309_out_ptr0` | deepseek8b | NORMALIZATION_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `deepseek8b_seq256_backward_659_output_0` | deepseek8b | LOSS_CE_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `mamba_seq128_backward_10615_in_out_ptr0` | mamba | NORMALIZATION_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `mamba_seq128_backward_9491_in_out_ptr0` | mamba | LOSS_HEAD_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `mamba_seq256_backward_18748_out_ptr0` | mamba | LOSS_CE_BACKWARD | ABSTAIN | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `mamba_seq64_backward_8402_out_ptr0` | mamba | STATE_SPACE_RECURRENT_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `phi4_seq128_backward_1032_output_0` | phi4 | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `phi4_seq128_backward_1133_out_ptr0` | phi4 | ATTENTION_PROJECTION_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `phi4_seq256_backward_613_in_out_ptr0` | phi4 | NORMALIZATION_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `qwen_seq128_backward_995_out_ptr0` | qwen | ATTENTION_PROJECTION_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `qwen_seq256_backward_1338_in_out_ptr0` | qwen | ATTENTION_STATE_OR_TRANSPORT_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `qwen_seq256_backward_1492_out_ptr1` | qwen | LOSS_CE_BACKWARD | NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `qwen_seq64_backward_519_in_out_ptr0` | qwen | LOSS_HEAD_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |
| `qwen_seq64_backward_540_out_ptr1` | qwen | NORMALIZATION_BACKWARD | CONFIRMED_TRAINING_UPDATE_EFFECT | MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION |

算子目录共 17 个族、目录记录约 173736 个位置。下表只给出当前根因证据等级，不把目录族数量当成独立 bias 数：

| 算子族 | 目录位置 | 当前根因证据 |
|---|---:|---|
| `LINEAR` | 0 | CASE_SPECIFIC_SOURCE_EVIDENCE_ONLY；MM/GEMM sources are isolated only in named cases; no universal linear root |
| `NORMALIZATION` | 1302 | MIXED_MEASUREMENT_AND_LOCAL_CONTROLS；some RMS/normalization controls exist, but the family is too broad for one root |
| `SOFTMAX` | 120 | LOCAL_ROOT_IN_ONE_SAVED_STATE_CASE；saved-state inconsistency is closed for one declared endpoint; natural family bias remains open |
| `CROSS_ENTROPY` | 1 | MEASUREMENT_ONLY_OR_CASE_BOUND；loss endpoints may be mixed with MM or NLL regions; no family-wide source claim |
| `SILU_GATING` | 10944 | LOCAL_SOURCE_CLOSED_IN_SELECTED_ENDPOINT；explicit source-choice intervention closes one endpoint, not the whole family |
| `SOFTPLUS` | 70 | MEASUREMENT_ONLY；no retained source-isolating intervention for this family |
| `RECURRENCE` | 10593 | MEASUREMENT_ONLY；no retained source-isolating intervention for the recurrent family |
| `ROTARY` | 32 | STATE_RESPONSE_SOURCE_OPEN；state dependence is measured, but arithmetic source and state components are not separated |
| `REDUCTION` | 36 | MULTIPLE_CASE_SPECIFIC_CONTROLS；Liger/expert/RMS results have different boundaries and cannot be merged |
| `INDEXED_ACCUMULATION` | 1 | MEASUREMENT_ONLY；no retained source-isolating intervention for this family |
| `GELU` | 34 | LOCAL_SOURCE_CLOSED_IN_SELECTED_ENDPOINT；explicit tanh source choice changes one endpoint; natural family bias remains open |
| `CONVOLUTION` | 24 | MEASUREMENT_ONLY；no retained source-isolating intervention for this family |
| `EMBEDDING` | 1 | MEASUREMENT_ONLY；embedding appears as a parameter carrier in several cases, not as an isolated embedding root |
| `SELECTION` | 0 | NEGATIVE_CONTROL_ONLY；equal-score tie-order control was identity on the declared suite |
| `OPTIMIZER_UPDATE` | 0 | END_TO_END_IN_ADAMW8BIT_CASE；moment residual propagation and targeted compensation are closed only for the declared optimizer case |
| `FUSED_ATTENTION` | 0 | PARTIAL_MULTI_SOURCE_REGION；one materialization contributor is isolated; the complete region has multiple sources |
| `ELEMENTWISE_BIAS` | 0 | MEASUREMENT_ONLY；no retained source-isolating intervention for the catalog family |

## 覆盖集合（不计入科学问题组）

原审计中的 99 个同输入 SiLU/RMS 位置和 31 个 reference-graph 区域保留为覆盖集合。它们证明统一流程可以运行，但没有逐项完成根因隔离，因此不计入上表的独立问题组数量。
