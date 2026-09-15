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

机器结果：`results/property/case_causal_audit_v1/root_cause_closure_current.json`。

## 覆盖集合（不计入科学问题组）

原审计中的 99 个同输入 SiLU/RMS 位置和 31 个 reference-graph 区域保留为覆盖集合。它们证明统一流程可以运行，但没有逐项完成根因隔离，因此不计入上表的独立问题组数量。
