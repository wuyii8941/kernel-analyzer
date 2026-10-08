#!/usr/bin/env python3
"""Build the current, conservative root-cause closure ledger.

The ledger is deliberately problem-oriented: model positions and repeated
conditions are evidence for a problem group, not additional root causes.  All
numeric fields for the newly incorporated GELU and selection controls are
recomputed from the retained raw records.  This report does not promote a
nonzero norm to a nonzero mean bias, and it does not turn an unresolved source
into a negative result.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# KA_OUTPUT_DIR (tests): write every output there instead of the tracked files, so a test run never rewrites them
_OUTPUT_DIR = __import__("os").environ.get("KA_OUTPUT_DIR")
OUT_JSON = (Path(_OUTPUT_DIR) / "root_cause_closure_current.json" if _OUTPUT_DIR
            else ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json")
OUT_MD = Path(_OUTPUT_DIR) / "root_cause_closure_current.md" if _OUTPUT_DIR else ROOT / "docs/root_cause_closure_current.md"

PROBLEM_NAMES = {
    "adamw8bit_moment_quantization": "AdamW8bit 保存的 moment",
    "liger_fused_linear_ce_dw_accumulation": "Liger fused linear CE 的 dW 累加",
    "liger_fused_linear_jsd_distillation": "Liger fused linear JSD 蒸馏损失",
    "mm_gemm_output_and_accumulation": "MM/GEMM 输出与累加（四个具体位置）",
    "bert_fused_addmm_bias_materialization": "BERT fused addmm 与拆分 bias 物化",
    "bert_nll_loss_evaluation": "BERT masked-LM NLL loss 求值与显式 log-softmax/gather",
    "softmax_saved_state_backward": "Softmax backward 保存状态",
    "bert_attention_softmax_materialization": "BERT attention softmax 求值与物化",
    "bert_attention_score_materialization": "BERT attention QK score 求值与物化",
    "bert_attention_value_materialization": "BERT attention value contraction 求值与物化",
    "silu_backward_evaluation": "SiLU backward 求值",
    "attention_state_to_q_projection_region": "attention 状态到 q-projection 区域",
    "fused_rope_position_scaling": "融合 RoPE 与位置缩放",
    "gemma_gelu_backward_evaluation": "GELU backward 求值",
    "gptneo_gelu_native_fp32_materialization": "GPT-Neo native GELU FP32 物化",
    "rmsnorm_cast_materialization": "RMSNorm 中间物化与权重乘法顺序",
    "bert_layernorm_compiled_materialization": "BERT LayerNorm 编译与 eager 物化路径",
    "bert_embedding_sum_materialization": "BERT word/position/token-type embedding 求和物化",
    "bert_pooler_tanh_materialization": "BERT pooler tanh 求值与物化",
    "granite_residual_addition_materialization": "Granite MoE attention residual 加法物化",
    "granite_moe_gate_product_materialization": "Granite MoE gate-times-up 门控乘法物化",
    "granite_moe_output_gate_materialization": "Granite MoE expert-output routing-gate 乘法物化",
    "granite_moe_router_score_materialization": "Granite MoE router score/weight 物化",
    "gemma3_vision_patch_convolution": "Gemma-3 视觉 patch convolution 累加",
    "qwen3vl_position_interpolation": "Qwen3-VL 学习位置表插值物化",
    "mamba_causal_conv_accumulation": "Mamba causal depthwise convolution 累加",
    "mamba_state_output_contraction_materialization": "Mamba recurrent state-to-output contraction 物化",
    "mamba_d_skip_materialization": "Mamba recurrent D skip-product 物化",
    "mamba_z_gate_materialization": "Mamba selective-scan z-gate 乘法物化",
    "mamba_fused_selective_scan_reassociation": "Mamba fused selective-scan 重结合",
    "deepseek_embedding_backward_accumulation": "DeepSeek/Qwen3 embedding backward 重复 token 累加",
    "deepseek_embedding_gradient_materialization": "DeepSeek embedding gradient BF16 物化",
    "deepseek_fused_embedding_nll_partial_materialization": "DeepSeek fused embedding/NLL backward partial 物化",
    "deepseek_fused_embedding_nll_reduction_order": "DeepSeek fused embedding/NLL backward FP32 归约顺序",
    "gemma4_causal_nll_loss_evaluation": "Gemma-4 causal NLL loss 求值与显式 log-softmax/gather",
    "gemma4_final_logit_softcap_materialization": "Gemma-4 final-logit soft-cap 物化",
    "gemma4_audio_output_projection_materialization": "Gemma-4 音频输出投影物化",
    "gemma4_audio_attention_softcap_materialization": "Gemma-4 音频 attention logit softcap 物化",
    "gemma4_audio_attention_softmax_probability_materialization": "Gemma-4 音频 attention probability BF16 物化",
    "gemma4_audio_lightconv_glu_product_materialization": "Gemma-4 音频 LightConv GLU 门控乘法物化",
    "gemma4_audio_lightconv_depthwise_conv_backward_accumulation": "Gemma-4 音频 LightConv depthwise convolution backward 累加",
    "gemma4_audio_subsampling_convolution_materialization": "Gemma-4 音频下采样 Conv2d 物化",
    "gemma4_rms_row_reduction_order": "Gemma-4 RMSNorm 行平方和归约顺序",
    "qwen3_attention_sdpa_eager_backend": "Qwen3 eager 与 SDPA attention backend",
    "mamba_softplus_materialization": "Mamba 递归 softplus 物化精度",
    "mamba_discrete_transition_materialization": "Mamba 离散状态转移 exp(A·delta) 物化",
    "rwkv_time_decay_materialization": "RWKV time-mix 递归 time-decay 指数物化",
    "rwkv_receptance_sigmoid_materialization": "RWKV attention receptance sigmoid 物化",
    "gemma_rms_feature_reduction_order": "Gemma RMS 归约顺序对照",
    "granite_router_topk_selection": "Granite router Top-k 对照",
    "granite_moe_expert_contribution_order": "Granite MoE expert 累加顺序",
    "olmoe_router_expert_accumulation": "OLMoE router 重复目标 expert 累加物化",
    "deberta_disentangled_relative_attention_materialization": "DeBERTa disentangled relative-attention 分数物化",
    "bloom_alibi_attention_materialization": "Bloom ALiBi attention 分数物化",
}

ROOT_CAUSE_LEVEL = {
    "adamw8bit_moment_quantization": "已定位：保存时丢失的分块量化残差进入下一步 moment 递推",
    "liger_fused_linear_ce_dw_accumulation": "已定位：相同 dW 分块乘积的 FP32 加法顺序；历史精度对照另列",
    "liger_fused_linear_jsd_distillation": "已定位：JSD/CE 分块蒸馏损失的 BF16 fused 求值与 FP32 显式参考之间的损失边界",
    "mm_gemm_output_and_accumulation": "条件定位：Qwen128 输出舍入；Qwen64/Mamba 算术加输出舍入；Phi 算术",
    "bert_fused_addmm_bias_materialization": "已定位：BERT native BF16 fused addmm 与拆分 matmul+bias 物化路径的融合边界；因素对照排除了单独 bias 加法精度",
    "bert_nll_loss_evaluation": "已定位：同一 BERT-tiny compiled logits 上 native cross-entropy 与显式 FP32 log-softmax/gather 的 NLL 求值边界",
    "softmax_saved_state_backward": "局部已定位：BF16 保存 scores 与物化前 FP32 统计量不一致",
    "bert_attention_softmax_materialization": "已定位：BERT CUDA attention softmax 的 native BF16 求值与同输入 FP32 softmax 后 BF16 写回之间的物化边界",
    "bert_attention_score_materialization": "已定位：BERT CUDA FP16 attention score QK 乘法的 native FP16 求值与同输入 FP32 求值后 FP16 写回之间的物化边界",
    "bert_attention_value_materialization": "已定位：BERT CUDA FP16 attention probability/value 乘法的 native FP16 求值与同输入 FP32 求值后 FP16 写回之间的物化边界",
    "silu_backward_evaluation": "已定位：生成 SiLU backward 导数路径中，product 与 derivative-factor 两个 FP32 中间物化边界的联合替换可逐状态复现 native/reference 写入差异；自然总体 mean bias 仍另行判断",
    "attention_state_to_q_projection_region": "区域已定位到 S_bwd；生成的 softmax-backward 输出、U、pre-softmax score、max/normalizer statistics 与 bmm_75 右侧 key carrier 均已路径保持地干预并改变 q_proj 梯度，整体上游仍为多来源",
    "fused_rope_position_scaling": "已定位：相同真实 vision-RoPE operands 上，FP32 multiply/add 后 BF16 写回单独复现 native→FP32 旋转差异；仅提高三角函数精度而保留 BF16 旋转为精确零差异",
    "gemma_gelu_backward_evaluation": "已定位：生成 Triton GELU backward 的 derivative 与 product-factor 联合 FP32 中间物化边界；自然总体 mean bias 仍另行判断",
    "gptneo_gelu_native_fp32_materialization": "已定位：GPT-Neo native NewGELUActivation 的 BF16 求值与同输入 FP32 tanh-GELU、原 dtype 写回之间的物化边界",
    "rmsnorm_cast_materialization": "已定位：真实 OLMoE、Qwen3 与 Gemma4 训练图中的 RMSNorm 物化边界改变了归一化结果与 BF16 权重相乘的顺序；FP32 权重乘法后一次写回是路径保持的对照",
    "bert_layernorm_compiled_materialization": "已定位到 LayerNorm 的 compiled/eager 求值与物化边界：compiled native 路径与显式 FP32 公式逐位一致，而 eager native 路径产生差异；更底层编译指令尚未单独归因",
    "bert_embedding_sum_materialization": "已定位到 BERT embedding block 的 word、position 与 token-type embedding 求和边界：native dtype 连续相加与同输入 FP32 三项求和、原 dtype 写回之间的物化差异",
    "bert_pooler_tanh_materialization": "已定位到 BERT pooler 的 tanh 求值/物化边界：native model-dtype tanh 与同输入 FP32 tanh、原 dtype 写回之间的差异",
    "granite_residual_addition_materialization": "已定位到 Granite MoE decoder attention residual 的加法边界：native dtype 的 residual + scaled attention output 与同输入 FP32 加法、原 dtype 写回之间的物化差异",
    "granite_moe_gate_product_materialization": "已定位到 Granite MoE expert 内部 activation(gate)×up 的门控乘法边界：native dtype 乘法与同输入 FP32 乘法、原 dtype 写回之间的物化差异；routing、expert projection 和 combine 保持不变",
    "granite_moe_output_gate_materialization": "已定位到 Granite MoE expert output×routing gate 的逐元素乘法边界：native dtype 乘法与同输入 FP32 乘法、原 dtype 写回之间的物化差异；routing、expert projection 和 combine 保持不变",
    "granite_moe_router_score_materialization": "已定位到 Granite MoE router score projection 的精度边界：native model-dtype linear score 与 FP32 score 在相同 native top-k 集合上产生不同 routing weights；expert selection 与 expert body 保持不变",
    "gemma3_vision_patch_convolution": "已定位：视觉 patch convolution 的 native BF16 累加与同输入 FP32 累加/原 dtype 写回之间的精度边界",
    "qwen3vl_position_interpolation": "已定位：Qwen3-VL 视觉学习位置表四点加权插值的 native dtype 求和与 FP32 求和/原 dtype 写回之间的物化边界",
    "mamba_causal_conv_accumulation": "已定位到 Mamba causal depthwise convolution 的 native BF16 求值/物化与同输入 FP32 求值、原 dtype 激活前写回之间的边界；底层卷积指令的内部累加类型仍未单独证明",
    "mamba_state_output_contraction_materialization": "已定位到 Mamba sequential selective scan 的 state-to-C 输出收缩边界：native dtype state/C 点积与同输入 FP32 点积、原 dtype 写回之间的物化差异；底层 kernel 实现类型仍未单独证明",
    "mamba_d_skip_materialization": "已定位到 Mamba sequential selective scan 的 residual D*hidden_states 乘法物化边界：native dtype 乘法与同输入 FP32 乘法、原 dtype 写回之间的差异；底层 kernel 实现类型仍未单独证明",
    "mamba_z_gate_materialization": "已定位到 Mamba sequential selective scan 输出与 SiLU(gate) 的乘法物化边界：native dtype 乘法与同输入 FP32 乘法、原 dtype 写回之间的差异；底层 kernel 实现类型仍未单独证明",
    "mamba_fused_selective_scan_reassociation": "已定位到实际 mamba_ssm fused selective-scan 与同层显式递推的重结合边界：保留同层 causal convolution、投影和下游路径，只将 scan kernel 切换为显式递推即可复现 fused 的 signed aligned-write component；完整 effect vector 仍有残差",
    "deepseek_embedding_backward_accumulation": "已定位：真实 Qwen3 embedding backward 的 native 重复 token 累加与同输入 FP32 index-add 累加之间的精度边界",
    "deepseek_embedding_gradient_materialization": "已定位：同一真实 embedding cotangent 的 FP32 梯度在写回 BF16 前的物化边界；重复 token 累加保持相同",
    "deepseek_fused_embedding_nll_partial_materialization": "已定位：真实 DeepSeek 生成 fused embedding/NLL/normalization backward 区域中 partial term 在归约前的 BF16 物化边界",
    "deepseek_fused_embedding_nll_reduction_order": "已定位：同一真实 DeepSeek 生成 fused embedding/NLL/normalization backward 边界内的 FP32 归约元素顺序边界",
    "gemma4_causal_nll_loss_evaluation": "已定位：同一 logits 上 native cross-entropy fused loss 与显式 FP32 log-softmax/gather 的 NLL 求值边界；不等同于 fused linear CE 的 dW 分块累加",
    "gemma4_final_logit_softcap_materialization": "已定位：同一 hidden/logit 输入上 native BF16 divide/tanh/multiply 与 FP32 求值后 BF16 写回的 final-logit soft-cap 边界",
    "gemma4_audio_output_projection_materialization": "已定位：Gemma-4 音频塔 output_proj 的 native BF16 乘加与同一音频特征上的 FP32 乘加、BF16 写回边界",
    "gemma4_audio_attention_softcap_materialization": "已定位：Gemma-4 音频 attention logits 的 softcap 除法/tanh/乘法边界；native FP32 求值与同层 BF16 或 FP64 单变量求值改变 q_proj 写入",
    "gemma4_audio_attention_softmax_probability_materialization": "已定位：Gemma-4 音频 attention 的 native FP32 softmax 结果在 value contraction 前写回 BF16；只保留 FP32 probabilities 的单变量边界在 layer-0 产生稳定 aligned write bias",
    "gemma4_audio_lightconv_glu_product_materialization": "已定位：Gemma-4 音频 LightConv1d 中 native-dtype GLU gate-times-value 乘法与 FP32 乘法后原 dtype 写回边界",
    "gemma4_audio_lightconv_depthwise_conv_backward_accumulation": "已定位：Gemma-4 音频 LightConv1d depthwise causal Conv1d 的 native backward 权重梯度累加与同输入显式 FP32 窗口乘加路径之间的边界；前向窗口语义已单独核对",
    "gemma4_audio_subsampling_convolution_materialization": "已定位：Gemma-4 音频下采样 Conv2d 的 native BF16 累加与同输入 FP32 累加、原 dtype 写回之间的物化边界；layer-1 前向边界已单独核对",
    "gemma4_rms_row_reduction_order": "已定位：同一 Gemma-4 RMSNorm 输入上的 FP32 行平方和归约顺序",
    "qwen3_attention_sdpa_eager_backend": "已闭合到语义 backend 边界：同一 attention 输入下仅替换目标层 eager 与 SDPA 接口就改变 q_proj 写入；更低层算术仅作为未细分项保留",
    "mamba_softplus_materialization": "已定位：同一 Mamba 顺序递归中，delta softplus 在输入 dtype 求值与 FP32 求值后写回之间的物化边界改变 dt_proj 写入",
    "mamba_discrete_transition_materialization": "已定位：同一 Mamba 顺序递归中，离散状态转移 exp(A·delta) 的指数参数在模型 dtype 与 FP32 中间路径之间的物化边界改变 A_log 写入",
    "rwkv_time_decay_materialization": "已定位：RWKV sequential WKV recurrence 中 time_decay 指数的模型 dtype 与 FP32 物化边界",
    "rwkv_receptance_sigmoid_materialization": "已定位：RWKV attention receptance sigmoid 的模型 dtype 与 FP32 求值、原 dtype 写回之间的物化边界",
    "gemma_rms_feature_reduction_order": "阴性：所测归约顺序不是早期区域差异的来源",
    "granite_router_topk_selection": "阴性：所测 tie-order 不改变选中集合或后续数值",
    "granite_moe_expert_contribution_order": "已定位：同为 FP32 的 expert contribution 累加顺序",
    "olmoe_router_expert_accumulation": "已定位：OLMoE router 的重复目标 expert contribution 在 BF16 中累加，与 FP32 累加后一次写回不同",
    "deberta_disentangled_relative_attention_materialization": "已定位：DeBERTa c2p/p2c disentangled relative-position score products 的 native BF16 求值与同输入 FP32 求值后 native dtype 返回之间的物化边界",
    "bloom_alibi_attention_materialization": "已定位：Bloom ALiBi attention 的 native baddbmm QK/ALiBi score 求值与同输入 FP32 QK 乘积及 ALiBi 加法后 native dtype 返回之间的物化边界",
}

# These are deliberately explicit blockers, rather than a generic "more data
# needed" label.  They describe why the retained artifacts cannot identify a
# stronger root cause without a new intervention or a new observable.
OPEN_ROOT_CAUSE_BLOCKERS = {
    "gemma_gelu_backward_evaluation": {
        "current_limit": "逐分量自然 probe 复现了候选 tanh-GELU 语义表达式；按生成 kernel 的临时量顺序重做的 32-state 输出级中介显示路径保持的 tanh、polynomial、derivative-factor 和 final-derivative 变体能解释约 0.597 的候选/reference write-effect norm，但方向余弦约 0.461、残差约 0.898，仍不是完整复现，因此这些 arithmetic boundary 不是充分根因",
        "why_offline_stops": "reference source-choice 的响应差异不能唯一证明自然 candidate 使用的某条 tanh/FMA/derivative 指令造成了观测差异；完整 32-state 复跑仍留下较大 residual，且只有 4/32 状态在所选参数上有非零目标 effect，不能仅由下游写入聚合量唯一选择其他中间量。原始 backward:952 contract 还要求 packed multiplier storage；当前 fresh same-family compile 只在动态发现的 endpoint 上可做路径保持替换，其他同名 kernel 不能静默合并",
        "next_observation": "在原始 runtime release 或一个显式验证过的 endpoint adapter 上恢复 backward:952 的 packed layout，再对生成指令级融合或中间物化做单变量干预并测 local→gradient→write；自然总体 mean bias 仍需独立状态库",
    },
}

TRAINING_RESULT = {
    "adamw8bit_moment_quantization": "两批各 8 对 1024 步训练支持声明幅度的 loss 改善",
    "liger_fused_linear_ce_dw_accumulation": "历史精度对照有轨迹分叉；纯 FP32 顺序的 1024 步对照末端 loss 差回到零",
    "liger_fused_linear_jsd_distillation": "真实 BERT-tiny 文本蒸馏边界的一步参数写入 aligned bias；未测多步质量后果",
    "mm_gemm_output_and_accumulation": "历史轨迹按位置分叉；未形成跨四位置共同质量结论",
    "bert_fused_addmm_bias_materialization": "BERT-tiny CPU 真实文本边界的一步 LayerNorm 参数写入存在稳定负 aligned scaling；未测 CUDA 或多步质量后果",
    "bert_nll_loss_evaluation": "BERT-tiny 128 个真实文档窗口的 lm_head 单步 AdamW 写入存在稳定正 aligned scaling；loss 区间跨零，未形成质量结论",
    "softmax_saved_state_backward": "1024 步参数轨迹分离；loss 差多次变号",
    "bert_attention_softmax_materialization": "BERT-tiny CUDA 单步 query 投影写入存在稳定负 aligned scaling；loss 完全相同，未测独立训练 loss",
    "bert_attention_score_materialization": "BERT-tiny CUDA FP16 单步 query 投影写入存在稳定负 aligned scaling；未测独立训练 loss",
    "bert_attention_value_materialization": "BERT-tiny CUDA FP16 单步 query 投影写入存在稳定负 aligned scaling；未测独立训练 loss",
    "silu_backward_evaluation": "小幅反馈维持分离；没有稳定质量损害",
    "attention_state_to_q_projection_region": "历史配对轨迹存在；未证单一来源导致质量变化",
    "fused_rope_position_scaling": "未测 loss",
    "gemma_gelu_backward_evaluation": "未测独立训练 loss",
    "gptneo_gelu_native_fp32_materialization": "32 个真实文本状态的单步参数写入有稳定负 aligned scaling；loss 区间跨零，未作多步质量结论",
    "rmsnorm_cast_materialization": "三种真实模型的单步写入差异已确认；loss 区间跨零，未形成质量结论",
    "bert_layernorm_compiled_materialization": "BERT-tiny 单步 AdamW 写入存在稳定的 aligned 差异；loss 区间跨零，未形成质量结论",
    "bert_embedding_sum_materialization": "BERT-tiny embedding.weight 的单步写入约 6.2% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "bert_pooler_tanh_materialization": "BERT-tiny pooler tanh 在 embedding carrier 上产生约 0.81% 梯度差异，held-out additive direction 区间为正；pooler dense carrier 未复现该方向，未测独立训练 loss",
    "granite_residual_addition_materialization": "Granite layer-0 attention o_proj.weight 的单步写入约 39.8% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "granite_moe_gate_product_materialization": "Granite layer-0 MoE input_linear.weight 的单步写入约 40.4% RMS、aligned scaling 稳定为负，held-out additive projection 区间为正；未测独立训练 loss",
    "granite_moe_output_gate_materialization": "Granite layer-0 MoE input_linear.weight 的单步写入约 42.3% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "granite_moe_router_score_materialization": "Granite layer-0 router.weight 在冻结 native top-k 后的单步写入约 192.7% RMS、aligned scaling 稳定为负；未测独立训练 loss",
    "gemma3_vision_patch_convolution": "Gemma-3 图文输入 bank 的单步 AdamW 写入存在约 106.6% RMS、稳定负 aligned scaling；loss 区间跨零，未形成质量结论",
    "qwen3vl_position_interpolation": "Qwen3-VL 真实图文 loss 下的单步位置表写入 aligned bias；未形成长期质量结论",
    "mamba_causal_conv_accumulation": "Mamba layer-0 causal conv1d.weight 的单步写入约 28.6% RMS、确认方向为负且 aligned scaling 稳定；未测独立训练 loss",
    "mamba_state_output_contraction_materialization": "Mamba layer-0 A_log 的单步写入约 30.6% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "mamba_d_skip_materialization": "Mamba layer-0 D 的单步写入约 29.2% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "mamba_z_gate_materialization": "Mamba layer-0 out_proj.weight 的 Wikitext 单步写入约 27.4% RMS、aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "mamba_fused_selective_scan_reassociation": "Mamba layer-3 实际 fused selective-scan 相对全顺序参考的单步写入约 31.5% RMS；16 状态确认 aligned 区间稳定为负，source-cut 复现该 signed aligned component（平均 effect-vector cosine 约 0.587，故不声称完整向量相同）；未测多步训练 loss",
    "deepseek_embedding_backward_accumulation": "DeepSeek/Qwen3 真实文本 bank 的单步 AdamW 写入存在约 1.05% RMS、稳定负 aligned scaling；前向 loss 完全相同，不代表质量结论",
    "deepseek_embedding_gradient_materialization": "DeepSeek 16 个真实文本状态中，单独改变 embedding gradient 的 FP32→BF16 物化使写入 RMS 约 3.0e-8，aligned 区间严格为负；效应极小，不作质量结论",
    "deepseek_fused_embedding_nll_partial_materialization": "DeepSeek 16 个真实文本状态中，fused embedding/NLL/normalization backward 的 partial BF16 物化使 embedding 写入 RMS 约 1.70%，aligned 区间严格为负；loss 完全相同，未作总体或长期质量结论",
    "deepseek_fused_embedding_nll_reduction_order": "DeepSeek 16 个真实文本状态中，同一 fused backward 边界只反转 FP32 归约顺序产生约 1.61e-6 写入 RMS，aligned 区间严格为正；效应极小，不作质量结论",
    "gemma4_causal_nll_loss_evaluation": "Gemma-4 E2B 文本 bank 的 lm_head 单步 AdamW 写入存在约 0.576% RMS、稳定负 aligned scaling；loss 区间跨零，未形成质量结论",
    "gemma4_final_logit_softcap_materialization": "Gemma-4 E2B 文本 bank 的 lm_head 单步 AdamW 写入存在约 2.22% RMS；26 状态确认半区 aligned 区间为负，未测独立训练 loss",
    "gemma4_audio_output_projection_materialization": "Gemma-4 E2B 真实语音的完整音频语言 loss 下，audio_tower.output_proj 单步写入差异约 88.8% RMS；16 状态确认 aligned 区间为负，未作长期质量结论",
    "gemma4_audio_attention_softcap_materialization": "Gemma-4 E2B 真实语音的完整音频语言 loss 下，layer-0 attention softcap 精度干预使 q_proj 写入 RMS 平均约 193.1%（BF16）或 215.3%（FP64）；固定 16 状态总 aligned 区间为负，确认半区区间跨零，未作总体或长期质量结论",
    "gemma4_audio_attention_softmax_probability_materialization": "Gemma-4 E2B layer-0 真实语音 16 状态中，仅保留 FP32 attention probabilities（softmax arithmetic、logits、softcap 和下游路径不变）使 q_proj 写入 RMS 平均约 1.77%，aligned-write 区间严格为负；loss 完全相同，未作总体或长期质量结论",
    "gemma4_audio_lightconv_glu_product_materialization": "Gemma-4 E2B 真实语音的完整音频语言 loss 下，LightConv GLU product 单步写入约 328.2% RMS；8 状态确认 aligned 区间稳定为负，未作长期质量结论",
    "gemma4_audio_lightconv_depthwise_conv_backward_accumulation": "Gemma-4 E2B 真实语音的完整音频语言 loss 下，LightConv depthwise Conv1d 权重写入 RMS 平均约 31.4%；8 状态确认 ratio-of-sums aligned scaling 为 +1.975%，但逐状态方向混合，未作总体或长期质量结论",
    "gemma4_audio_subsampling_convolution_materialization": "Gemma-4 E2B 真实语音的下采样 Conv2d layer-1 写入 RMS 平均约 125.5%；8 状态确认 ratio-of-sums aligned scaling 约为 -38.0%，未作总体或长期质量结论",
    "gemma4_rms_row_reduction_order": "Gemma-4 layer-0 input_layernorm.weight 单步 AdamW 的固定集合 aligned update bias；loss 区间跨零，未测多步训练",
    "qwen3_attention_sdpa_eager_backend": "Qwen3 layer-13 q_proj 的 eager/SDPA 单步写入约 36.1% RMS、稳定负 aligned scaling；loss 区间跨零，未形成质量结论",
    "mamba_softplus_materialization": "Mamba layer-0 dt_proj.weight 的单步写入约 9.5% RMS、稳定负 aligned scaling；未测独立训练 loss",
    "mamba_discrete_transition_materialization": "Mamba layer-0 A_log 的单步写入约 26.1% RMS、确认 aligned scaling 稳定为负；held-out additive direction 未确认，未测独立训练 loss",
    "rwkv_time_decay_materialization": "RWKV-4-world block-0 真实文本单步参数写入约 29.25% RMS；未测多步训练 loss",
    "rwkv_receptance_sigmoid_materialization": "RWKV-4-world block-0 真实文本单步参数写入约 7.66% RMS；确认半区 aligned scaling 稳定为负，未测多步训练 loss",
    "gemma_rms_feature_reduction_order": "未测训练 loss",
    "granite_router_topk_selection": "固定集合无差异",
    "granite_moe_expert_contribution_order": "未测训练 loss",
    "olmoe_router_expert_accumulation": "固定真实 OLMoE 状态中的 cold-start AdamW aligned update bias；未测多步训练 loss",
    "deberta_disentangled_relative_attention_materialization": "DeBERTa-v3-small 两层真实文本边界均有稳定负 aligned write scaling；MLM 头为本地探针中新初始化，未作训练质量结论",
    "bloom_alibi_attention_materialization": "Bloom-560m 两层真实文本边界均有稳定负 aligned write scaling；未测独立训练 loss",
}

PRIMARY_EVIDENCE = {
    "adamw8bit_moment_quantization": "results/property/result_analysis_v4/iid_training_confirmation/verification.json",
    "liger_fused_linear_ce_dw_accumulation": "results/property/liger_fp32_chunk_order_v1/length64_population_mean_v1.json",
    "liger_fused_linear_jsd_distillation": "results/property/case_causal_audit_v1/liger_jsd_natural_training_boundary.json",
    "mm_gemm_output_and_accumulation": "results/property/case_causal_audit_v1/mm_conditional_sources.json",
    "bert_fused_addmm_bias_materialization": "results/property/case_causal_audit_v1/bert_fused_addmm_bias_materialization_boundary.json",
    "bert_nll_loss_evaluation": "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json",
    "softmax_saved_state_backward": "results/property/root_cause_closure_v1/softmax_saved_state.json",
    "bert_attention_softmax_materialization": "results/property/case_causal_audit_v1/bert_attention_softmax_materialization_boundary.json",
    "bert_attention_score_materialization": "results/property/case_causal_audit_v1/bert_attention_score_materialization_boundary.json",
    "bert_attention_value_materialization": "results/property/case_causal_audit_v1/bert_attention_value_materialization_boundary.json",
    "silu_backward_evaluation": "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json",
    "attention_state_to_q_projection_region": "results/property/bias_formation_final/intervention_results/qwen_l23_attention_state.json",
    "fused_rope_position_scaling": "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v4.json",
    "gemma_gelu_backward_evaluation": "results/property/root_cause_closure_v1/gelu_source_factorial_v1.json",
    "gptneo_gelu_native_fp32_materialization": "results/property/root_cause_closure_v1/gptneo_gelu_natural_32_20260920.json",
    "rmsnorm_cast_materialization": "results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json",
    "bert_layernorm_compiled_materialization": "results/property/new_problem_group_search_v1/bert_tiny_layernorm_eager_natural_32_20260918.json",
    "bert_embedding_sum_materialization": "results/property/new_problem_group_search_v1/bert_embedding_sum_materialization_natural_32_20260918.json",
    "bert_pooler_tanh_materialization": "results/property/new_problem_group_search_v1/bert_pooler_tanh_natural_cpu32_20260919.json",
    "granite_residual_addition_materialization": "results/property/new_problem_group_search_v1/granite_residual_addition_materialization_natural_32_20260918.json",
    "granite_moe_gate_product_materialization": "results/property/new_problem_group_search_v1/granite_moe_gate_product_materialization_natural_32_20260918.json",
    "granite_moe_output_gate_materialization": "results/property/new_problem_group_search_v1/granite_moe_output_gate_materialization_natural_32_20260918.json",
    "granite_moe_router_score_materialization": "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json",
    "gemma3_vision_patch_convolution": "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json",
    "qwen3vl_position_interpolation": "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_16_20260918.json",
    "mamba_causal_conv_accumulation": "results/property/new_problem_group_search_v1/mamba_causal_conv_materialization_natural_32_20260918.json",
    "mamba_state_output_contraction_materialization": "results/property/new_problem_group_search_v1/mamba_state_output_contraction_natural_32_20260918.json",
    "mamba_d_skip_materialization": "results/property/new_problem_group_search_v1/mamba_d_skip_materialization_natural_32_20260918.json",
    "mamba_z_gate_materialization": "results/property/new_problem_group_search_v1/mamba_z_gate_materialization_wikitext_32_20260918.json",
    "mamba_fused_selective_scan_reassociation": "results/property/new_problem_group_search_v1/mamba_scan_source_cut_layer3_16_20260920.json",
    "deepseek_embedding_backward_accumulation": "results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json",
    "deepseek_embedding_gradient_materialization": "results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json",
    "deepseek_fused_embedding_nll_partial_materialization": "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json",
    "deepseek_fused_embedding_nll_reduction_order": "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_reverse_v5_20260920.json",
    "gemma4_causal_nll_loss_evaluation": "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json",
    "gemma4_final_logit_softcap_materialization": "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json",
    "gemma4_audio_output_projection_materialization": "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_16_20260919.json",
    "gemma4_audio_attention_softcap_materialization": "results/property/new_problem_group_search_v1/gemma4_audio_attention_softcap_layer0_natural_16_20260920.json",
    "gemma4_audio_attention_softmax_probability_materialization": "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_natural_16_20260920.json",
    "gemma4_audio_lightconv_glu_product_materialization": "results/property/new_problem_group_search_v1/gemma4_audio_glu_layer0_natural_16_20260919.json",
    "gemma4_audio_lightconv_depthwise_conv_backward_accumulation": "results/property/new_problem_group_search_v1/gemma4_audio_depthwise_conv_layer0_natural_16_20260920.json",
    "gemma4_audio_subsampling_convolution_materialization": "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_raw_16_20260920.json",
    "gemma4_rms_row_reduction_order": "results/property/new_problem_group_search_v1/gemma4_rms_row_reduction_natural_26_20260919.json",
    "qwen3_attention_sdpa_eager_backend": "results/property/new_problem_group_search_v1/qwen3_attention_backend_isolated_layer13_qproj_16_20260918.json",
    "mamba_softplus_materialization": "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_32_20260918.json",
    "mamba_discrete_transition_materialization": "results/property/new_problem_group_search_v1/mamba_discrete_transition_materialization_natural_cpu16_20260919.json",
    "rwkv_time_decay_materialization": "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json",
    "rwkv_receptance_sigmoid_materialization": "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_natural_32_20260920.json",
    "gemma_rms_feature_reduction_order": "results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json",
    "granite_router_topk_selection": "results/property/numerical_coverage_v1/granite_selection_suite24_v1/summary.json",
    "granite_moe_expert_contribution_order": "results/property/granite_expert_order_population_v1/result.json",
    "olmoe_router_expert_accumulation": "results/property/new_problem_group_search_v1/olmoe_router_accum_natural_16_20260919.json",
    "deberta_disentangled_relative_attention_materialization": "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer0_seed0_32_20260919.json",
    "bloom_alibi_attention_materialization": "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_16_20260919.json",
}

# These records are kept in the single ledger so exploratory measurements do
# not get mistaken for additional scientific problem groups.  They are not
# active rows and therefore do not affect the deduplicated case count.
NOT_COUNTED_CANDIDATES = [
    {
        "candidate": "selected_nll_logsoftmax_backward",
        "status": "NOT_PROMOTED_RUNTIME_BINDING_INCOMPLETE",
            "reason": (
            "The Qwen selected-NLL endpoint has a nonzero local/gradient/write profile on "
            "a 32-state record, but the saved execution contract marks runtime binding as "
            "incomplete and the reference as a partial bound-input formula. It is therefore "
            "not a closed natural training source case."
        ),
        "evidence": [
            "results/property/numerical_coverage_v1/qwen128_nll_capture_v1/raw/mapped_backward_515_in_out_ptr0.json",
            "results/property/numerical_coverage_v1/qwen128_nll_capture_v1/raw/family_execution_protocol.json",
            "results/property/numerical_coverage_v1/qwen128_nll_capture_v1/completion_verification.json",
        ],
    },
    {
        "candidate": "gemma_softcapped_nll_backward",
        "status": "PROMOTED_TO_ACTIVE_GROUP_WITH_CLEAN_SAME_LOGITS_BOUNDARY",
        "reason": (
            "The earlier Gemma softcapped-NLL capture had incomplete runtime binding and "
            "is retained as historical evidence only. A new same-logits Gemma-4 E2B probe "
            "now isolates the native cross-entropy versus explicit FP32 log-softmax/gather "
            "boundary and is promoted separately as gemma4_causal_nll_loss_evaluation."
        ),
        "evidence": [
            "results/property/numerical_coverage_v1/gemma_softcap_capture_v4/raw/softcap_backward_669_in_out_ptr0.json",
            "results/property/numerical_coverage_v1/gemma_softcap_capture_v4/raw/family_execution_protocol.json",
            "results/property/numerical_coverage_v1/gemma_softcap_capture_v4/family_completion_v2.json",
            "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json",
            "scripts/run_gemma4_softcapped_nll_natural_probe.py",
        ],
    },
    {
        "candidate": "triton_signature_nonisolated_region_pool",
        "status": "NON_ISOLATED_REGION_CANDIDATE_POOL",
        "reason": (
            "The frozen Triton signature campaigns contain 31 valid real-model endpoint "
            "measurements across the v4, v5 and v9 batches, including several large "
            "parameter-write effects. Their comparisons are AOT region substitutions "
            "with possible upstream differences, so they identify a high-value natural "
            "candidate pool but do not identify an individual kernel root or a distinct "
            "problem group. A source review of the highest-priority embedding-dense region "
            "also found that its visible arithmetic already loads BF16 values into FP32 and "
            "mixes NLL/normalization terms. A same-local Qwen embedding-backward probe on the "
            "same model family gives only a 0.0497% mean write RMS, far below the 51% fused-region "
            "screen, so the large region effect cannot be assigned to embedding accumulation alone. "
            "A same compiled-call-boundary DeepSeek probe then changed only the visible fused reduction: "
            "BF16 partial materialization produced about 1.68% write RMS over eight real states, while "
            "an FP64 reduction changed writes by about 1.5e-6%; both are far below the region screen. "
            "A full FP32 expression replay at the same compiled call boundary also changed writes by only "
            "about 1.7e-6 (about 0.00017%), with a confirmation aligned interval crossing zero. This rules out the reviewed "
            "visible fused arithmetic and reduction schedule as the sufficient source of the large region "
            "effect, but does not isolate the remaining upstream producer or indexed-update path. "
            "A subsequent 16-state same-boundary probe replaced each of the three visible upstream "
            "BF16 MM outputs with same-operand FP32 products: the three single replacements and the "
            "all-three replacement all had negative aligned intervals (approximately -88.8%, -89.4%, "
            "-96.7% and -99.2% means, respectively). This identifies a sufficient upstream MM-output "
            "materialisation contribution and is merged with the existing MM/GEMM problem group; it "
            "does not create a second group for the same MM root or explain every effect in the larger "
            "non-isolated region. "
            "These candidates must be source-isolated before promotion."
        ),
        "evidence": [
            "results/property/numerical_coverage_v1/triton_signature_batches_v4/summary_final.json",
            "results/property/numerical_coverage_v1/triton_signature_batches_v5/summary_final.json",
            "results/property/numerical_coverage_v1/triton_signature_batches_v9/summary_final.json",
            "results/property/new_problem_group_search_v1/deepseek_embedding_dense_source_screen_20260918.json",
            "results/property/new_problem_group_search_v1/qwen_embedding_backward_natural_16_20260920.json",
            "results/property/root_cause_closure_v1/deepseek_embedding_backward_seq128_8_20260920.json",
            "results/property/root_cause_closure_v1/deepseek_nll_same_logits_seq128_8_20260920.json",
            "results/property/root_cause_closure_v1/deepseek_embedding_gradient_cast_seq128_8_20260920.json",
            "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_8_v3_20260920.json",
            "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_fp32_replay_20260920.json",
            "results/property/new_problem_group_search_v1/deepseek_fused_mm_source_probe_16_20260920.json",
            "scripts/run_deepseek_fused_mm_source_probe.py",
        ],
    },
    {
        "candidate": "gemma_rms_forward_normalized",
        "status": "MERGED_WITH_EXISTING_RMSNORM_GROUP_LEGACY_BINDING_INCOMPLETE",
        "reason": (
            "Two source-checked Gemma RMS forward-normalized endpoint records show fixed-suite "
            "parameter-write RMS of about 6.7% and 8.2% on the same declared carrier. The "
            "family execution contract does not establish complete runtime binding, so those "
            "legacy endpoints are not promoted independently. A new 16-state Gemma-4 model-level "
            "probe does establish the same cast-order RMSNorm boundary and is merged into the "
            "existing rmsnorm_cast_materialization group."
        ),
        "evidence": [
            "results/property/numerical_coverage_v1/gemma_rms_forward_capture_v1/family_completion_v1.json",
            "results/property/numerical_coverage_v1/gemma_rms_forward_capture_v1/raw/rms_forward_normalized_forward_116_out_ptr0-rms-forward-normalized-common-input.json",
            "results/property/numerical_coverage_v1/gemma_rms_forward_capture_v1/raw/rms_forward_normalized_forward_200_out_ptr0-rms-forward-normalized-common-input.json",
            "results/property/new_problem_group_search_v1/gemma4_rmsnorm_natural_16_20260918.json",
            "scripts/run_gemma4_rmsnorm_natural_probe.py",
        ],
    },
    {
        "candidate": "mamba_recurrence_fused_scan",
        "status": "SCREENED_NOT_PROMOTED_MEAN_DIRECTION_UNCONFIRMED",
        "reason": (
            "The official fused Mamba path is now executable in the compatible Transformers "
            "environment and was compared with the sequential reference on all 32 real text "
            "windows. The 24-state held-out projection has 11 positive and 13 negative values, "
            "mean -8.05e-5 with an approximate 95% interval [-4.32e-4, 2.71e-4], despite a "
            "large first-step write RMS. Earlier A_log, x_proj.weight, dt_proj.bias, and the "
            "new out_proj.weight replay likewise retained mixed signs or intervals crossing zero "
            "(the out_proj mean is -0.00256 with interval [-0.00695, 0.00184]). This "
            "closes the execution blocker but not the signed-bias criterion. A further 64-state "
            "real-bank replay has 15 positive and 17 negative held-out projections, with mean "
            "-1.84e-3 and approximate 95% interval crossing zero. The latest 126-state pilot-bank "
            "confirmation has 67 positive and 59 negative projections, mean -1.32e-4, and an "
            "aligned-gradient interval [-1.536e-2, 1.256e-3], which still crosses zero. The "
            "alternative mambapy execution on 32 states also has 18 positive and 14 negative "
            "projections with aligned-gradient interval [-1.681e-3, 7.151e-4]. The fused "
            "recurrence is therefore a real, large-magnitude candidate, not a confirmed new "
            "mean- or aligned-bias group. The latest layer-0 x_proj replay on 24 real states "
            "also has about 30.5% mean write RMS, but its aligned-gradient interval "
            "[-1.543e-2, 1.177e-2] crosses zero (11 positive and 5 negative projections); "
            "it is retained as evidence against promoting the whole fused region, not as a "
            "closed x_proj problem group. A new layer-3 local probe now isolates one actual fused "
            "selective-scan boundary and is promoted separately as mamba_fused_selective_scan_reassociation; "
            "the full multi-layer fused region remains unpromoted because its other layers and carriers "
            "still have mixed directions."
        ),
        "evidence": [
            "results/mamba_scan/fused_confirm_seq256.json",
            "results/mamba_scan/fused_confirm_seq1024.json",
            "results/mamba_scan/family_confirmation.json",
            "results/property/new_problem_group_search_v1/mamba_A_log_natural_bank16_16_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_x_proj_natural_bank16_16_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_dt_proj_bias_natural_bank16_16_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_A_log_seq1024_4_16_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_x_proj_seq1024_4_20_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_A_log_layer23_seq1024_4_20_20260918_summary.json",
            "results/property/new_problem_group_search_v1/mamba_dt_proj_bias_layer23_seq1024_4_20_20260918_summary.json",
            "results/property/new_problem_group_search_v1/mamba_x_proj_layer23_seq1024_4_20_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_fused_official_natural_20_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_fused_official_natural_32_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_fused_official_natural_pilot128_20260919.json",
            "results/property/new_problem_group_search_v1/mamba_mambapy_natural_pilot32_20260919.json",
            "results/property/new_problem_group_search_v1/mamba_fused_out_proj_aligned_cuda32_20260918.json",
            "results/property/new_problem_group_search_v1/mamba_fused_official_natural_bank05_64_20260919.json",
            "results/property/new_problem_group_search_v1/mamba_fused_x_proj_layer0_natural_24_20260920.json",
            "results/property/new_problem_group_search_v1/mamba_official_local_scan_layer3_16_20260920.json",
            "results/property/new_problem_group_search_v1/mamba_scan_source_cut_layer3_16_20260920.json",
            "scripts/run_mamba_official_local_scan_probe.py",
            "scripts/run_mamba_scan_source_cut.py",
        ],
    },
    {
        "candidate": "mainstream_generated_operator_scans",
        "status": "NO_NEW_DISTINCT_ROOT_CAUSE",
        "reason": (
            "The generated-input mainstream scan contains four confirmed endpoint instances, "
            "but its own deduplication audit assigns zero new distinct root-cause groups; the "
            "instances reconfirm fused RoPE/GELU families or remain family-level unresolved. "
            "The additional five-model screen likewise confirmed only the already represented "
            "RMSNorm and rotary families; activation, projection, softmax, SDPA and Softplus "
            "did not pass the directional screen, while Gemma-4 remained unresolved at this "
            "route. A second six-model screen again confirmed only rotary instances and no "
            "new semantic family; one model remained unresolved at this route."
            " The remaining 13-model screen likewise found no new semantic family: the only "
            "confirmed endpoints were already represented RMSNorm and rotary families; the "
            "LayerNorm, activation, projection, softmax and attention checks were not confirmed, "
            "and the Ministral route remained unresolved."
        ),
        "evidence": [
            "results/property/mainstream_model_bias_scan_v1/deduplicated_case_audit_20260918.json",
            "results/property/new_problem_group_search_v1/mainstream_candidate_screen_20260918.json",
            "results/property/new_problem_group_search_v1/mainstream_candidate_screen_extra_20260918.json",
            "results/property/new_problem_group_search_v1/mainstream_candidate_screen_remaining_20260918.json",
        ],
    },
    {
        "candidate": "mamba_softplus_backward",
        "status": "NOT_PROMOTED_POOLED_DIRECTION_UNCONFIRMED",
        "reason": (
            "Source-checked Softplus backward records cover the same Mamba dt_proj.bias "
            "carriers at sequence lengths 64, 128 and 256. Pooling the three fixed suites "
            "gives nonzero but tiny write effects whose 95% aligned intervals cross zero; "
            "the family is therefore not a confirmed natural mean-bias group. A fresh "
            "16-calibration/16-confirmation run on the existing 32-state bank produced "
            "a 0.0931 mean write RMS but an 8-positive/8-negative frozen-direction split "
            "and a near-zero projected mean, so the large magnitude is not enough to promote it."
        ),
        "evidence": [
            "results/property/numerical_coverage_v1/mamba64_softplus_bias_batch000_v1",
            "results/property/numerical_coverage_v1/mamba128_softplus_device1_batch000_v1",
            "results/property/numerical_coverage_v1/mamba256_softplus_device2_batch000_v1",
            "results/property/new_problem_group_search_v1/mamba_dt_proj_bias_natural_bank16_16_20260918.json",
        ],
    },
    {
        "candidate": "external_liger_loss_and_execution_candidates",
        "status": "TRIAGED_NO_NEW_CLOSED_GROUP",
        "reason": (
            "An independent audit of the external Liger/DFuzz candidates found no new closed "
            "natural problem group. The apparent ORPO discrepancy omitted the beta weight in "
            "the reference formula; the TVD discrepancy mixed mean with batchmean reduction; "
            "SimPO matches the corrected explicit formula; GeGLU compared exact GELU with the "
            "implementation's tanh-GELU; SwiGLU, torch.compile and fused Adam remain composite "
            "or source-unisolated candidates. They are retained as a frontier, not counted as "
            "new roots."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/external_liger_candidate_triage_20260919.json",
            "results/property/new_problem_group_search_v1/liger_loss_recheck_20260918/single_step_results.json",
        ],
    },
    {
        "candidate": "indexed_accumulation_reduction_order",
        "status": "CONDITIONAL_OPERATOR_ONLY_NOT_NATURAL",
        "reason": (
            "The repeated-destination finite-precision reduction-order probe is a closed "
            "operator-level result, but it has no natural LLM training boundary, parameter "
            "write, or declared training-state population and is kept outside the active count."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/indexed_accumulation_root_cause_summary_20260918.json",
        ],
    },
    {
        "candidate": "gemma4_per_layer_input_gate_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real Gemma-4 text-bank component probe compared the per-layer input "
            "gate-times-input product in native BF16 with a float32 product followed by "
            "BF16 writeback. Eight states produced exact zero loss, gradient, and parameter "
            "write differences on the declared layer-0 carrier. The path therefore adds no "
            "natural problem group; its surrounding projection and product arithmetic also "
            "overlap existing GEMM/materialization families."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gemma4_per_layer_input_probe_20260919.json",
            "scripts/run_gemma4_per_layer_input_probe.py",
        ],
    },
    {
        "candidate": "deepseek_gqa_repeat_kv_backward",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "A real DeepSeek/Qwen3 checkpoint was replayed for both k_proj and v_proj on "
            "the declared 16-state text bank. Replacing native expand+reshape repeat_kv "
            "with repeat_interleave produced exact zero gradient, write, and loss effects "
            "in the selected layers. A direct Qwen3 layer-25 v_proj replay on the same kind "
            "of 16-state bank is also exact zero, so this specific GQA repeat boundary is "
            "not a new natural problem group. The larger non-isolated layout-region records "
            "remain open because they contain other fused operations and possible upstream changes."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/deepseek_gqa_repeat_kv_kproj_16_20260918.json",
            "results/property/new_problem_group_search_v1/deepseek_gqa_repeat_kv_vproj_16_20260918.json",
            "results/property/new_problem_group_search_v1/qwen3_repeat_kv_layer25_vproj_16_20260918.json",
            "scripts/run_deepseek_gqa_repeat_kv_natural_probe.py",
        ],
    },
    {
        "candidate": "qwen3_causal_mask_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "A real Qwen3-1.7B layer-13 attention boundary was replayed on 16 declared text states, "
            "holding model weights, token ids, eager attention and the target parameter fixed while "
            "comparing the native implicit causal mask with an explicit additive upper-triangular mask. "
            "Gradient, AdamW-write and loss effects were exactly zero on every observed state, so this "
            "mask representation is not a new natural problem group under the tested path."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/qwen3_causal_mask_layer13_qproj_16_20260918.json",
            "scripts/run_qwen3_causal_mask_natural_probe.py",
        ],
    },
    {
        "candidate": "qwen3_attention_contraction_expression",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "A real Qwen3-1.7B layer-13 attention boundary was replayed on 16 declared text states, "
            "holding the eager attention path fixed while replacing both score/value torch.matmul "
            "contractions with equivalent einsum expressions. Gradient, AdamW-write and loss effects "
            "were exactly zero on every observed state. This expression-level reassociation is therefore "
            "not a new natural problem group under the tested backend and bank."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/qwen3_attention_contraction_layer13_qproj_16_20260918.json",
            "scripts/run_qwen3_attention_contraction_natural_probe.py",
        ],
    },
    {
        "candidate": "qwen3_attention_output_layout_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "A real Qwen3-1.7B layer-13 attention boundary was replayed on 16 declared text states, "
            "holding the attention values and all arithmetic fixed while replacing transpose(1,2).contiguous() "
            "with an equivalent permute(...).reshape output materialization. Gradient, AdamW-write and loss "
            "effects were exactly zero on every observed state for q_proj; v_proj and k_proj replays on the "
            "same 16-state bank are also exact zero. This layout spelling is therefore not a new natural "
            "problem group under the tested paths."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/qwen3_attention_output_layout_layer13_qproj_16_20260918.json",
            "results/property/new_problem_group_search_v1/qwen3_attention_output_layout_layer13_vproj_16_20260918.json",
            "results/property/new_problem_group_search_v1/qwen3_attention_output_layout_layer13_kproj_16_20260920.json",
            "results/property/root_cause_closure_v1/qwen3_attention_output_layout_qproj_16_20260920.json",
            "scripts/run_qwen3_attention_output_layout_natural_probe.py",
        ],
    },
    {
        "candidate": "deepseek_attention_output_layout_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "The highest-priority DeepSeek layout/transpose region candidates were replayed "
            "on the declared DeepSeek-Qwen3 checkpoint and text bank. Holding attention "
            "values and arithmetic fixed, replacing transpose(1,2).contiguous() with the "
            "equivalent permute(...).reshape materialization produced exact zero gradient, "
            "AdamW-write, and loss effects for both a layer-25 v_proj carrier and a layer-13 "
            "q_proj carrier. This specific layout spelling is therefore not a new natural "
            "problem group; the larger AOT region records remain non-isolated because they "
            "contain other fused operations and possible upstream differences. The retained "
            "generated source for the layer-25 endpoint also shows a four-head BF16 input "
            "reduction accumulated in FP32 before one BF16 store, so that visible kernel "
            "body does not by itself expose a new low-precision reduction root."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/deepseek_layout_vproj_natural_8_20260919.json",
            "results/property/new_problem_group_search_v1/deepseek_layout_qproj_natural_8_20260919.json",
            "scripts/run_qwen3_attention_output_layout_natural_probe.py",
        ],
    },
    {
        "candidate": "bert_embedding_backward",
        "status": "SCREENED_NOT_PROMOTED_WRITE_EFFECT_NEGLIGIBLE",
        "reason": (
            "A real BERT-tiny masked-LM graph with an untied decoder was compared at the word "
            "embedding boundary against explicit FP32 index-add from the captured upstream "
            "gradient. Repeated-token accumulation produced a small gradient difference, but "
            "the parameter-write effect was about 5.46e-8 and all directional intervals crossed zero."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/bert_tiny_embedding_natural_32_20260918.json",
            "scripts/run_bert_tiny_embedding_natural_probe.py",
        ],
    },
    {
        "candidate": "ministral3_patch_merger_layout",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY",
        "reason": (
            "The real Ministral-3 vision patch-merger boundary was replayed on a trained "
            "checkpoint and a real image. The released F.unfold patch grouping was compared "
            "with an explicit spatial unfold/permute grouping using the same vision features, "
            "the same merger weights, and the same downstream quadratic probe. Blocks, merger "
            "outputs, and merger-weight gradients were exactly identical, so this layout "
            "spelling is not a new natural problem group under the tested path. A genuinely "
            "different patch-merger arithmetic would be a different semantic intervention."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/ministral_patch_merger_probe_20260918.json",
            "scripts/run_ministral_patch_merger_probe.py",
        ],
    },
    {
        "candidate": "bert_residual_addition_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_CPU",
        "reason": (
            "A real BERT-tiny masked-LM attention residual-addition boundary was replayed on "
            "32 text windows on both the available CPU and CUDA paths. Replacing only the "
            "native residual addition by an FP32 addition followed by the original write-back "
            "produced exact zero gradient, write and loss differences on both paths. It is "
            "therefore retained as a negative screen rather than counted as a new problem "
            "group; this does not rule out a different fused residual implementation."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/bert_residual_addition_materialization_natural_32_20260918.json",
            "results/property/new_problem_group_search_v1/bert_residual_addition_materialization_cuda32_20260919.json",
            "scripts/run_bert_residual_addition_materialization_natural_probe.py",
        ],
    },
    {
        "candidate": "gemma3_vision_embedding_addition_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real Gemma-3 image/text replay over 16 states replaced only the SigLIP "
            "patch-plus-position embedding addition with FP32 addition followed by the "
            "original BF16 write-back. The candidate and reference produced exact zero "
            "gradient, parameter-write and loss differences on this path, so the vision "
            "embedding addition is retained as a negative screen rather than promoted to a "
            "new problem group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gemma3_vision_embedding_addition_natural_16_20260918.json",
            "scripts/run_gemma3_vision_embedding_addition_natural_probe.py",
        ],
    },
    {
        "candidate": "gemma3_vision_projector_pool_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real Gemma-3 image/text replay over 16 states replaced only the 4x4 average "
            "pool in the multimodal projector by FP32 average pooling followed by the original "
            "vision-dtype write-back. The candidate and reference produced exact zero gradient, "
            "parameter-write and loss differences on this path, so this projector boundary is "
            "retained as a negative screen rather than promoted to a new problem group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gemma3_vision_pool_materialization_natural_16_20260918.json",
            "scripts/run_gemma3_vision_pool_materialization_natural_probe.py",
        ],
    },
    {
        "candidate": "gemma3_vision_projector_pool_materialization_repeat",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A second real Gemma-3 image/text replay over 16 states repeated the 4x4 "
            "average-pool boundary with FP32 accumulation followed by the original "
            "vision-dtype write-back.  Gradient, parameter-write, direction and loss "
            "effects were exactly zero on every observed state.  This is an explicitly "
            "recorded repeat of the earlier negative screen, not an additional problem group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gemma3_vision_pool_materialization_natural_16_20260920.json",
            "scripts/run_gemma3_vision_pool_materialization_natural_probe.py",
        ],
    },
    {
        "candidate": "gemma4_audio_attention_softmax_materialization",
        "status": "SCREENED_NOT_PROMOTED_NO_DIRECTIONAL_BIAS",
        "reason": (
            "Two real Gemma-4 audio-language attention layers were replayed on 16 real "
            "waveforms with the same logits, softcap, downstream path and q_proj carrier. "
            "Changing only the attention softmax evaluation from FP32 to FP64 produced a "
            "large write effect (about 99%--100% RMS), but the layer-0 aligned-write "
            "interval [-52.7%, +18.1%], the layer-1 interval [-53.8%, +8.7%], and both "
            "held-out direction intervals crossed zero.  This is a high-variance candidate, "
            "not a confirmed signed natural bias or a new final problem group under the "
            "strict root-cause rule."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gemma4_audio_attention_softmax_natural_8_20260920.json",
            "results/property/new_problem_group_search_v1/gemma4_audio_attention_softmax_natural_16_20260920.json",
            "results/property/new_problem_group_search_v1/gemma4_audio_attention_softmax_layer1_natural_16_20260920.json",
            "scripts/run_gemma4_audio_attention_softmax_natural_probe.py",
        ],
    },
    {
        "candidate": "bert_softmax_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_CPU",
        "reason": (
            "A real BERT-tiny attention path was replayed on 16 text windows while holding "
            "the model, attention scores and downstream path fixed. Replacing only the "
            "attention softmax with FP32 evaluation followed by the original dtype write-back "
            "produced exact zero gradient, parameter-write and loss effects on the available "
            "CPU path. It is retained as a CPU negative screen and does not rule out a CUDA-specific "
            "softmax implementation difference."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/bert_softmax_materialization_natural_cpu16_20260919.json",
        ],
    },
    {
        "candidate": "gptneo_local_attention_mask_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real GPT-Neo-125M local-attention path was replayed on 16 text windows while "
            "holding model weights, Q/K/V, the causal window and the downstream path fixed; "
            "a second run uses 512-token windows to activate the local-window boundary. "
            "Replacing the native boolean local mask materialization with an explicit additive "
            "masked-fill produced exact zero gradient, parameter-write and loss effects on every "
            "observed state. This is a negative control, not a new natural problem group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gptneo_local_attention_materialization_natural_16_20260920.json",
            "results/property/new_problem_group_search_v1/gptneo_local_attention_materialization_natural_16_len512_20260920.json",
            "scripts/run_gptneo_local_attention_materialization_natural_probe.py",
        ],
    },
    {
        "candidate": "olmoe_router_score_materialization",
        "status": "MERGED_WITH_EXISTING_ROUTER_SCORE_GROUP",
        "reason": (
            "A real OLMoE router boundary was replayed with the actual native top-k indices "
            "frozen, changing only the router-score materialization from native dtype to "
            "FP32 before the native write-back. The write RMS is substantial (about 11.5%--12.3% "
            "across the 16, 32 and 64-state banks), but the held-out aligned-write intervals "
            "cross zero in all three smaller banks (the 64-state interval is [-1.27%, +0.84%]). "
            "A 128-state replay now gives a negative aligned-write interval "
            "[-1.503%, -0.279%], confirming the same router-score/weight materialization source "
            "already closed for Granite. It is therefore merged as cross-model confirmation, "
            "not counted as a second problem group; the corrected probe supersedes the "
            "discarded invalid embedding-based top-k screen."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/olmoe_router_score_natural_16_20260920.json",
            "results/property/new_problem_group_search_v1/olmoe_router_score_natural_32_20260920.json",
            "results/property/new_problem_group_search_v1/olmoe_router_score_natural_64_20260920.json",
            "results/property/new_problem_group_search_v1/olmoe_router_score_natural_128_20260920.json",
            "scripts/run_olmoe_router_score_natural_probe.py",
        ],
    },
    {
        "candidate": "granite_router_gate_materialization",
        "status": "SCREENED_NOT_PROMOTED_NO_DIRECTIONAL_BIAS",
        "reason": (
            "A new same-input Granite MoE probe held router logits, top-k indices, "
            "expert computations and repeated-destination accumulation fixed while "
            "changing only the top-k routing-weight cast/multiplication boundary. "
            "The write effect is substantial (15.9%, 15.5% and 13.9% RMS on 16, 64 "
            "and 128 natural states), but the held-out aligned-write intervals cross "
            "zero in every run ([-0.75%, +0.70%], [-2.57%, +0.86%] and "
            "[-1.23%, +0.23%]). The effect is therefore a high-variance routing-weight "
            "candidate, not a confirmed signed natural bias or a new final problem group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/granite_router_gate_materialization_natural_16_20260920.json",
            "results/property/new_problem_group_search_v1/granite_router_gate_materialization_natural_64_20260920.json",
            "results/property/new_problem_group_search_v1/granite_router_gate_materialization_natural_128_20260920.json",
            "results/property/new_problem_group_search_v1/granite_router_gate_materialization_natural_64_v2_20260920.json",
            "results/property/new_problem_group_search_v1/granite_router_gate_materialization_shakespeare_128_20260920.json",
            "scripts/run_granite_router_gate_materialization_natural_probe.py",
        ],
    },
    {
        "candidate": "t5_relative_attention_bias_addition",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real T5-small encoder self-attention boundary was replayed on 16 real "
            "README-text states. Replacing only the score-plus-relative-position bias "
            "addition by FP32 evaluation followed by the original BF16 write-back produced "
            "exact zero gradient, parameter-write, and loss effects on every observed state. "
            "The tested T5 relative-bias addition is therefore a negative screen, not a new "
            "natural problem group; a different relative-bias lookup or attention backend "
            "would be a separate intervention."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/t5_relative_attention_bias_natural_16_20260920.json",
            "scripts/run_t5_relative_attention_bias_natural_probe.py",
        ],
    },
    {
        "candidate": "gptneo_embedding_position_addition_materialization",
        "status": "SCREENED_NOT_PROMOTED_EXACT_IDENTITY_BF16_PATH",
        "reason": (
            "A real GPT-Neo-125M checkpoint was replayed on 32 declared text windows. "
            "Replacing only the token-plus-position embedding addition by FP32 evaluation "
            "followed by the original BF16 write-back produced exact zero gradient, "
            "parameter-write and loss effects on every observed state. The tested embedding "
            "addition is therefore a negative screen, not a new natural problem group; a "
            "different embedding lookup or fused materialization boundary would be a separate "
            "intervention."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/gptneo_embedding_position_addition_natural_32_20260920.json",
            "scripts/run_gpt2_embedding_position_addition_natural_probe.py",
        ],
    },
    {
        "candidate": "rwkv_channel_mix_square_relu_materialization",
        "status": "SCREENED_NOT_PROMOTED_SPARSE_EFFECT_NO_DIRECTIONAL_BIAS",
        "reason": (
            "A real RWKV-4-world checkpoint was replayed on 64 declared text windows while "
            "holding the channel-mix path fixed and changing only the square/ReLU evaluation "
            "boundary. Only one of 64 states produced a nonzero write difference; the remaining "
            "states were exact, and the resulting aligned interval crossed zero. The probe is "
            "therefore a sparse arithmetic screen rather than a confirmed natural bias group."
        ),
        "evidence": [
            "results/property/new_problem_group_search_v1/rwkv_channel_mix_natural_16_20260920.json",
            "results/property/new_problem_group_search_v1/rwkv_channel_mix_natural_64_20260920.json",
            "results/property/new_problem_group_search_v1/rwkv_channel_mix_method_64_20260920.json",
            "scripts/run_rwkv_channel_mix_natural_probe.py",
        ],
    },
]


def mean_bias_description(status: str) -> str:
    if status.startswith("SUPPORTED_PROJECTED_MEAN"):
        return "声明经验输入库内的冻结方向均值有支持；不外推自然训练总体"
    if status == "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK":
        return "声明经验轨迹库的有放回抽样 aligned 标量均值有支持；不外推自然训练总体或全向量均值"
    if status.startswith("SUPPORTED_ALIGNED_MEAN"):
        return "声明独立 history 中的 aligned 标量均值有支持；未证全向量均值"
    if status.startswith("NEGATIVE_CONTROL"):
        return "所测对照未确认方向；不证明所有实现无偏"
    if status.startswith("IDENTITY_ON_OBSERVED"):
        return "所测固定集合逐项相同；不外推其他输入"
    if status.startswith("FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE"):
        return "固定真实模型状态集合的 aligned 写入描述支持；没有总体保证"
    if status.startswith("FIXED_SUITE_ALIGNED"):
        return "仅固定集合的 aligned 描述；没有总体向量均值证据"
    if status.startswith("FIXED_SUITE"):
        return "仅固定集合描述；没有总体向量均值证据"
    return "自然训练状态总体的均值未确认"


def short_evidence_link(row: dict[str, Any]) -> str:
    primary = PRIMARY_EVIDENCE[row["problem_group"]]
    if (ROOT / primary).is_file():
        return f"[原始记录](../{primary})"
    for source in reversed(row["evidence"]):
        if (ROOT / source).exists() and source.startswith("results/"):
            return f"[原始记录](../{source})"
    return "见机器账本的 evidence 字段"


def read(path: str | Path) -> Any:
    return json.loads((ROOT / path if not Path(path).is_absolute() else Path(path)).read_text())


def confirmation_rows(raw: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    ids = set(map(str, raw.get("confirmation_state_ids", [])))
    return [row for sid, row in zip(raw["state_ids"], raw["original_coordinate_statistics"][stage])
            if str(sid) in ids]


def rms(raw: dict[str, Any], stage: str) -> float:
    rows = confirmation_rows(raw, stage)
    effect = sum(float(row["effect_energy"]) for row in rows)
    repair = sum(float(row["repair_energy"]) for row in rows)
    if repair <= 0:
        return 0.0 if effect == 0 else math.inf
    return math.sqrt(effect / repair)


def aligned(raw: dict[str, Any], stage: str) -> float:
    rows = confirmation_rows(raw, stage)
    effect = sum(float(row.get("effect_repair_inner_product", 0.0)) for row in rows)
    repair = sum(float(row["repair_energy"]) for row in rows)
    return effect / repair if repair else 0.0


def heldout_direction_from_gram(profile: dict[str, Any]) -> dict[str, Any]:
    """Recompute fixed-suite projections, retaining an unidentified direction."""
    suite = profile["suite"]
    gram = suite["joint_gram"]["effect_effect"]
    nc, nt = suite["calibration_state_count"], suite["confirmation_state_count"]
    if len(gram) != nc + nt or any(len(row) != len(gram) for row in gram):
        raise ValueError("expected calibration states followed by confirmation states")
    norm_sq = math.fsum(gram[i][j] for i in range(nc) for j in range(nc)) / nc**2
    norm = math.sqrt(max(0.0, norm_sq))
    projections = [math.fsum(gram[i][:nc]) / (nc * norm)
                   for i in range(nc, nc + nt)] if norm > 0 else None
    return {
        "scope": "DESCRIPTIVE_FIXED_SUITE_ONLY",
        "calibration_count": nc,
        "confirmation_count": nt,
        "direction_status": "IDENTIFIED" if norm > 0 else "NOT_IDENTIFIABLE",
        "calibration_mean_norm": norm,
        "confirmation_projections": projections,
        "confirmation_projection_mean": math.fsum(projections) / nt if projections is not None else None,
        "confirmation_positive_count": sum(x > 0 for x in projections) if projections is not None else None,
        "confirmation_negative_count": sum(x < 0 for x in projections) if projections is not None else None,
        "confirmation_zero_count": sum(x == 0 for x in projections) if projections is not None else None,
        "confirmation_mean_energy": math.fsum(gram[i][j] for i in range(nc, nc + nt)
                                              for j in range(nc, nc + nt)) / nt**2,
        "confirmation_nonzero_cross_state_inner_products": sum(
            gram[i][j] != 0 for i in range(nc, nc + nt) for j in range(i + 1, nc + nt)),
        "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
        "zero_population_mean_proven": False,
    }


def gelu_evidence() -> dict[str, Any]:
    base = "results/property/numerical_coverage_v1"
    dirs = {
        "natural_reference": "gemma_gelu_capture_v2",
        "explicit_exponential_tanh": "gemma_gelu_tanh_exp_confirmation_v3",
        "native_tanh": "gemma_gelu_tanh_native_confirmation_v3",
        "fused_multiply_add": "gemma_gelu_fma_confirmation_v1",
    }
    records: dict[str, Any] = {}
    for label, directory in dirs.items():
        raw = read(f"{base}/{directory}/raw/gelu_backward_952_out_ptr0.json")
        records[label] = {
            "state_count": len(raw["state_ids"]),
            "confirmation_count": len(raw["confirmation_state_ids"]),
            "local_rms": rms(raw, "LOCAL"),
            "gradient_rms": rms(raw, "PARAMETER_GRADIENT"),
            "update_rms": rms(raw, "ADAMW_UPDATE"),
            "write_rms": rms(raw, "PARAMETER_WRITE"),
            "update_aligned_ratio": aligned(raw, "ADAMW_UPDATE"),
            "reference_variant": raw["reference_comparison_scope"]["reference_variant"],
            "same_local_operands": raw["reference_comparison_scope"]["same_local_operands"],
        }
    profile_fields = ("local_rms", "gradient_rms", "update_rms", "write_rms", "update_aligned_ratio")
    records["native_tanh_vs_fma_exact_in_recorded_profile"] = all(
        records["native_tanh"][field] == records["fused_multiply_add"][field]
        for field in profile_fields
    )
    factorial_path = ROOT / "results/property/root_cause_closure_v1/gelu_source_factorial_v1.json"
    if factorial_path.is_file():
        factorial = json.loads(factorial_path.read_text())
        records["same_bank_source_factorial"] = factorial
    natural_path = ROOT / "results/property/root_cause_closure_v1/gelu_natural_intermediates_v1.json"
    if natural_path.is_file():
        records["natural_intermediate_probe"] = json.loads(natural_path.read_text())
    component_path = ROOT / "results/property/root_cause_closure_v1/gelu_componentwise_intermediates_v1.json"
    if component_path.is_file():
        records["componentwise_intermediate_probe"] = json.loads(component_path.read_text())
    arithmetic_path = ROOT / "results/property/root_cause_closure_v1/gelu_arithmetic_interventions_4_20260920.json"
    if arithmetic_path.is_file():
        records["path_preserving_arithmetic_interventions"] = json.loads(arithmetic_path.read_text())
    mediation_path = ROOT / "results/property/root_cause_closure_v1/gelu_single_source_mediation_v6_32_20260921.json"
    if mediation_path.is_file():
        mediation = json.loads(mediation_path.read_text())
        records["path_preserving_joint_mediation"] = mediation
        joint = mediation.get("mediation_summary", {}).get("reference_derivative_product", {})
        records["joint_product_derivative_source_closed"] = bool(
            mediation.get("status") == "COMPLETE_PATH_PRESERVING_GELU_MEDIATION_INTERVENTIONS_V4"
            and joint.get("exact_write_count") == mediation.get("state_count")
            and float(joint.get("residual_norm_ratio_aggregate", 1.0)) <= 1e-9
        )
    else:
        records["joint_product_derivative_source_closed"] = False
    gelu_probe_path = ROOT / "results/property/root_cause_closure_v1/gelu_mean_probe_empirical_bank_128_20260922.json"
    if gelu_probe_path.is_file():
        probe = json.loads(gelu_probe_path.read_text())
        records["declared_empirical_bank_mean_probe"] = {
            "source": str(gelu_probe_path.relative_to(ROOT)),
            "summary": probe.get("summary", {}).get("torch_reference"),
            "sampling": probe.get("sampling"),
            "claim_boundary": (
                "The native-minus-reference aligned parameter-write mean is tested on a "
                "with-replacement draw from the declared Gemma-4 trajectory bank; it is not "
                "an unrestricted natural-training population claim."
            ),
        }
    source_closed = records["joint_product_derivative_source_closed"]
    return {
        "case_id": "gemma_gelu_backward_952",
        "source": "generated Triton tanh-GELU backward product; same operands and one declared trainable parameter",
        "status": (
            "SOURCE_CLOSED_JOINT_DERIVATIVE_PRODUCT_BOUNDARY_NATURAL_MEAN_BIAS_NOT_CONFIRMED"
            if source_closed
            else "SOURCE_CHOICE_RESPONSE_CONFIRMED_NATURAL_MEAN_BIAS_NOT_CONFIRMED"
        ),
        "evidence": records,
        "interpretation": (
            "The explicit exponential tanh choice changes the measured gradient/update profile, "
            "whereas native tanh and its fused multiply-add spelling agree on the disjoint confirmation. "
            "The path-preserving joint derivative/product-factor override now reproduces the native/reference "
            "parameter-write effect exactly on all 32 retained states; tanh-only, factor-only and product-only "
            "overrides do not provide the same sufficient boundary. This closes the reviewed source at the "
            "joint intermediate-materialization boundary, not at a unique instruction decomposition or a "
            "natural-population mean-bias claim."
        ),
    }


def gptneo_gelu_evidence() -> dict[str, Any]:
    """Summarize the independently closed native GPT-Neo GELU boundary."""
    path = ROOT / "results/property/root_cause_closure_v1/gptneo_gelu_natural_32_20260920.json"
    data = read(path)
    summary = data["summary"]
    return {
        "case_id": "gptneo_gelu_natural_32_20260920",
        "source": data["operator"],
        "reference": data["reference"],
        "model": data["model"],
        "layer": data["layer"],
        "state_count": summary["state_count"],
        "write_effect_rms_mean": summary["write_effect_rms_mean"],
        "aligned_write_mean": summary["aligned_write_mean"],
        "aligned_write_interval_normal_95": summary["aligned_write_interval_normal_95"],
        "gradient_effect_rms_mean": summary["gradient_effect_rms_mean"],
        "loss_difference_interval_normal_95": summary["loss_difference_interval_normal_95"],
        "source_native_formula_bf16_relative_l2_mean": summary[
            "source_native_formula_bf16_relative_l2_mean"
        ],
        "source_native_vs_fp32_formula_relative_l2_mean": summary[
            "source_native_vs_fp32_formula_relative_l2_mean"
        ],
        "scope": data["claim_boundary"],
        "semantic_family_note": (
            "This is the same broad GELU semantic family as the Gemma generated-Triton case, "
            "whose joint derivative/product-factor source boundary is now closed; it remains "
            "a separate native GPT-Neo implementation boundary, and Gemma's natural mean-bias "
            "question remains separate."
        ),
    }


def liger_order_confirmation_evidence() -> dict[str, Any]:
    result = {}
    for length in (64, 256):
        data = read(f"results/property/liger_fp32_chunk_order_v1/length{length}_confirmation.json")
        result[str(length)] = {
            "state_count": len(data["state_ids"]),
            "confirmation_positive_count": data["source_prediction"]["confirmation_positive_count"],
            "confirmation_count": data["source_prediction"]["confirmation_count"],
            "source_prediction_role": "DESCRIPTIVE_FIXED_SUITE_ONLY",
            "gradient_direction": heldout_direction_from_gram(data["profiles"]["PARAMETER_GRADIENT"]),
            "proposed_update_direction": heldout_direction_from_gram(data["profiles"]["ADAMW_UPDATE"]),
            "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
            "legacy_population_labels_accepted": False,
            "claim_boundary": data["claim_boundary"],
        }
    training = read("results/property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json")
    result["training_1024"] = {
        "validation_loss_difference": training["validation_loss_difference"],
        "claim_boundary": training["claim_boundary"],
    }
    write_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_parameter_write_confirmation_v1.json"
    if write_path.exists():
        write_data = json.loads(write_path.read_text())
        write_profile = write_data["profiles"]["PARAMETER_WRITE"]
        result["length64_actual_parameter_write_confirmation"] = {
            "state_count": len(write_data["state_ids"]),
            "fixed_suite_total_write_rms": write_data["fixed_suite_total_write_rms"],
            "fixed_suite_total_write_effect_energy": write_data["fixed_suite_total_write_effect_energy"],
            "fixed_suite_total_reference_write_energy": write_data["fixed_suite_total_reference_write_energy"],
            "write_direction_diagnostic": heldout_direction_from_gram(write_profile),
            "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
            "legacy_population_labels_accepted": False,
            "legacy_gradient_profile_status": "INVALID_REFERENCE_USED_HIDDEN_GRADIENT_INSTEAD_OF_WEIGHT_GRADIENT",
            "gradient_to_write_l2_ratio_mean": math.fsum(
                row["effect_l2"] / row["gradient_effect_l2"]
                for row in write_data["rows"] if row["gradient_effect_l2"] > 0
            ) / sum(row["gradient_effect_l2"] > 0 for row in write_data["rows"]),
            "gradient_to_write_l2_ratio_range": [
                min(row["effect_l2"] / row["gradient_effect_l2"] for row in write_data["rows"] if row["gradient_effect_l2"] > 0),
                max(row["effect_l2"] / row["gradient_effect_l2"] for row in write_data["rows"] if row["gradient_effect_l2"] > 0),
            ],
            "claim_boundary": write_data["claim_boundary"],
        }
    population_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_population_mean_v1.json"
    if population_path.exists():
        population = json.loads(population_path.read_text())
        result["length64_population_mean_confirmation"] = {
            "source": str(population_path.relative_to(ROOT)),
            "calibration_count": population["calibration_count"],
            "confirmation_count": population["confirmation_count"],
            "confirmation": population["confirmation"],
            "scope": population["population"],
            "claim_boundary": population["claim_boundary"],
        }
    population256_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length256_population_mean_v1.json"
    if population256_path.exists():
        population256 = json.loads(population256_path.read_text())
        result["length256_population_mean_confirmation"] = {
            "source": str(population256_path.relative_to(ROOT)),
            "calibration_count": population256["calibration_count"],
            "confirmation_count": population256["confirmation_count"],
            "confirmation": population256["confirmation"],
            "scope": population256["population"],
            "claim_boundary": population256["claim_boundary"],
        }
    condition_path = ROOT / "results/property/liger_fp32_chunk_order_v1/input_conditions_parameter_write_v1.json"
    if condition_path.exists():
        condition_data = json.loads(condition_path.read_text())
        result["input_condition_probe"] = {
            "conditions": condition_data["condition_policy"]["conditions"],
            "summaries": condition_data["summaries"],
            "interpretation": (
                "On the fixed hidden-state bank, reversing or collapsing labels changes the "
                "write RMS only modestly (about 1.38e-6 to 1.50e-6) and leaves a positive but "
                "descriptive sketch direction. These conditions do not identify a unique natural "
                "input structure that prevents cancellation."
            ),
            "claim_boundary": condition_data["claim_boundary"],
        }
    kahan_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_kahan_intervention_new.json"
    if kahan_path.exists():
        kahan = json.loads(kahan_path.read_text())
        variants = {}
        for name, value in kahan.get("profiles", {}).items():
            variants[name] = {
                "gradient_total_effect_rms": value["PARAMETER_GRADIENT"]["suite"]["total_effect_rms"],
                "update_total_effect_rms": value["ADAMW_UPDATE"]["suite"]["total_effect_rms"],
                "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
            }
        ratios = [float(row["kahan_to_reverse_gradient_l2_ratio"]) for row in kahan["rows"]]
        result["length64_kahan_intervention"] = {
            "state_count": len(kahan["rows"]),
            "variants": variants,
            "kahan_to_reverse_gradient_l2_ratio_mean": math.fsum(ratios) / len(ratios),
            "kahan_to_reverse_gradient_l2_ratio_range": [min(ratios), max(ratios)],
            "claim_boundary": kahan["claim_boundary"],
        }
    return result


def silu_factorial_evidence() -> dict[str, Any]:
    raw = read(
        "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/"
        "mapped_backward_667_in_out_ptr0-silu-common-input.json"
    )
    result = {
        "case_id": raw["case_id"],
        "source_task_id": raw["source_task_id"],
        "state_count": len(raw["state_ids"]),
        "confirmation_count": len(raw["confirmation_state_ids"]),
        "same_local_operands": raw["reference_comparison_scope"]["same_local_operands"],
        "source_attribution": raw["reference_comparison_scope"]["single_kernel_source_attribution"],
        "stages": {},
    }
    for stage in ("LOCAL", "PARAMETER_GRADIENT", "ADAMW_MOMENT1_WRITE", "ADAMW_MOMENT2_WRITE", "ADAMW_UPDATE", "PARAMETER_WRITE", "NEXT_STEP_COMMON_GRADIENT_UPDATE"):
        profiles = []
        for value in raw["stages"][stage].values():
            suite = value["profile"]["suite"]
            profiles.append({
                "total_effect_rms": suite["total_effect_rms"],
                "mean_effect_over_repair_rms": suite["mean_effect_over_repair_rms"],
                "repair_aligned_effect": suite["repair_aligned_effect"],
                "residual_direction_heldout_effect": suite.get("residual_direction_heldout_effect"),
            })
        result["stages"][stage] = {
            "total_effect_rms_range": [min(x["total_effect_rms"] for x in profiles), max(x["total_effect_rms"] for x in profiles)],
            "mean_effect_over_repair_rms_range": [min(x["mean_effect_over_repair_rms"] for x in profiles), max(x["mean_effect_over_repair_rms"] for x in profiles)],
            "repair_aligned_effect_range": [min(x["repair_aligned_effect"] for x in profiles), max(x["repair_aligned_effect"] for x in profiles)],
        }
    factorial_path = ROOT / "results/property/root_cause_closure_v1/silu_source_factorial_v1.json"
    if factorial_path.is_file():
        result["same_bank_source_factorial"] = json.loads(factorial_path.read_text())
    intermediate_path = ROOT / "results/property/root_cause_closure_v1/silu_intermediate_probe_v1.json"
    if intermediate_path.is_file():
        result["intermediate_source_probe"] = json.loads(intermediate_path.read_text())
    natural_path = ROOT / "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json"
    if natural_path.is_file():
        result["natural_intermediate_probe"] = json.loads(natural_path.read_text())
    arithmetic_path = ROOT / "results/property/root_cause_closure_v1/silu_arithmetic_interventions_v4_4_20260920.json"
    if arithmetic_path.is_file():
        result["path_preserving_arithmetic_interventions"] = json.loads(arithmetic_path.read_text())
    mediation_path = ROOT / "results/property/root_cause_closure_v1/silu_single_source_mediation_v7_32_20260921.json"
    if mediation_path.is_file():
        mediation = json.loads(mediation_path.read_text())
        result["joint_product_factor_mediation"] = {
            "source": str(mediation_path.relative_to(ROOT)),
            "schema": mediation.get("schema"),
            "status": mediation.get("status"),
            "state_count": mediation.get("state_count"),
            "summary": mediation.get("mediation_summary", {}).get("reference_product_factor"),
            "factor_only_summary": mediation.get("mediation_summary", {}).get("reference_factor"),
            "product_only_summary": mediation.get("mediation_summary", {}).get("reference_product"),
            "claim_boundary": (
                "The generated derivative product and derivative-factor intermediates are jointly "
                "replaced by same-input FP32 reference values while preserving the downstream write path."
            ),
        }
    mean_probe_path = ROOT / "results/property/root_cause_closure_v1/silu_mean_probe_empirical_bank_128_20260921.json"
    if mean_probe_path.is_file():
        result["declared_empirical_bank_mean_probe"] = {
            "source": str(mean_probe_path.relative_to(ROOT)),
            "summary": json.loads(mean_probe_path.read_text()).get("summary", {}).get("torch_reference"),
            "sampling": json.loads(mean_probe_path.read_text()).get("sampling"),
            "claim_boundary": (
                "The native-minus-reference aligned parameter-write mean is tested on a "
                "with-replacement draw from the declared DeepSeek trajectory bank; it is not "
                "an unrestricted natural-training population claim."
            ),
        }
    return result


def mm_source_evidence() -> dict[str, Any]:
    """Summarize the case-specific MM source decompositions without merging them."""
    source_records = read("results/property/case_causal_audit_v1/mm_conditional_sources.json")
    decompositions = {
        "qwen_seq128_forward_8_output": "results/coverage/cases/qwen128_vproj_precision_decomposition.json",
        "qwen_seq64_forward_8_output": "results/coverage/cases/qwen64_vproj_precision_decomposition.json",
        "mamba_seq64_forward_1_output": "results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json",
        "phi4_seq64_backward_497_output": "results/coverage/cases/phi4_seq64_lmhead_dx_precision_decomposition.json",
    }
    conditional_paths = {
        "qwen_seq128_forward_8_output": "results/property/conditional_debias/qwen128_vproj.json",
        "qwen_seq64_forward_8_output": "results/property/conditional_debias/qwen64_vproj.json",
        "mamba_seq64_forward_1_output": "results/property/conditional_debias/mamba_seq64_input_proj.json",
    }
    conditional_by_case = {row["case_id"]: row for row in source_records["cases"]}
    cases = {}
    conditional_summary = {}
    for case_id, path in decompositions.items():
        decomposition = read(path)
        row = conditional_by_case.get(case_id)
        conditional = read(conditional_paths[case_id]) if case_id in conditional_paths else {}
        checks = row.get("checks", {}) if row else {}
        preservation = {
            name: {
                "repeat_count": len(value.get("records", [])),
                "nonzero_preservation_repeats": sum(
                    record["preservation_error"]["nonzero"] > 0
                    for record in value.get("records", [])
                ),
            }
            for name, value in checks.items()
        }
        aggregate = {}
        for arm_name, arm in conditional.get("arms", {}).items():
            aggregate_payload = arm.get("aggregate", {})
            roles = aggregate_payload.get("roles", {})
            aggregate[arm_name] = {
                "status": aggregate_payload.get("status"),
                "condition_count": aggregate_payload.get("condition_count"),
                "role_status_counts": {
                    role: value.get("status_counts", {})
                    for role, value in roles.items()
                },
                "all_conditions_candidate_local_biased": roles.get(
                    "candidate_local_effect_removed", {}
                ).get("all_conditions_biased", False),
                "all_conditions_candidate_zero_moment_update_biased": roles.get(
                    "candidate_adamw_zero_update_effect_removed", {}
                ).get("all_conditions_biased", False),
                "repair_residual_centered": roles.get(
                    "repair_local_residual", {}
                ).get("all_conditions_centered", False),
                "absolute_downstream_reference": aggregate_payload.get(
                    "absolute_downstream_repair_bias"
                ),
            }
        cases[case_id] = {
            "coherent_sources": decomposition["coherent_sources"],
            "decomposition_status": decomposition["status"],
            "decomposition_scope": decomposition["claim_boundary"],
            "conditional_checks": preservation,
            "conditional_debias": aggregate,
        }
        conditional_summary[case_id] = aggregate
    fused_mm = read("results/property/new_problem_group_search_v1/deepseek_fused_mm_source_probe_16_20260920.json")
    fused_mm_summary = fused_mm.get("summary", {})
    fused_mm_variants = fused_mm_summary.get("variants", {})
    return {
        "problem_group": "mm_gemm_output_and_accumulation",
        "case_specific_sources": cases,
        "conditional_downstream_summary": conditional_summary,
        "deepseek_fused_boundary_upstream_mm": {
            "state_count": fused_mm_summary.get("state_count"),
            "confirmation_count": fused_mm_summary.get("confirmation_count"),
            "variants": fused_mm_variants,
            "interpretation": (
                "At the real DeepSeek fused embedding/NLL backward boundary, replacing any one "
                "of the three visible BF16 MM outputs by the same-operand FP32 product changes "
                "the downstream embedding write in the same negative aligned direction across "
                "all 16 states. Replacing all three gives the same direction. This closes an "
                "upstream MM-output-materialization contribution, but it is merged into the "
                "existing MM/GEMM root group rather than counted as a new operator problem."
            ),
        },
        "interpretation": (
            "The local source is case-specific: output rounding only for Qwen128, "
            "kernel plus output rounding for Qwen64 and Mamba, and kernel arithmetic "
            "for Phi. The retained same-input conditional-debias ensembles additionally "
            "show that, for the Qwen cases and the local/zero-moment Mamba branches, "
            "the candidate-minus-debiased downstream effect remains directionally biased "
            "while the repair residual is centered. The DeepSeek fused-boundary probe adds "
            "a same-input upstream MM-output materialization confirmation, still within the "
            "same MM/GEMM root family. These are conditional case-level F+B results, not "
            "one universal MM root, an absolute high-precision reference, or a common "
            "natural-population mean bias."
        ),
    }


def rms_order_evidence() -> dict[str, Any]:
    data = read("results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json")
    cases = {}
    for task, row in data["cases"].items():
        cases[task] = {
            "native_endpoint_rms": row.get("variants", {}).get("FP32_NATIVE", {}).get("endpoint", {}).get("total_effect_rms"),
            "reverse_endpoint_rms": row.get("variants", {}).get("FP32_REVERSE_FEATURE_ORDER", {}).get("endpoint", {}).get("total_effect_rms"),
            "direct_endpoint_rms": row.get("source_intervention", {}).get("endpoint_total_rms_absolute_difference"),
            "direct_gradient_rms": row.get("source_intervention", {}).get("reference_parameter_gradient_difference", {}).get("total_effect_rms"),
            "direction_status": row.get("direction_status"),
        }
    return {
        "case_id": "gemma_rms_feature_reduction_order",
        "source": "FP32 feature-reduction order in a real compiled Gemma graph",
        "status": "SOURCE_CONTROL_NEGATIVE_NO_ROOT_CAUSE_FOUND",
        "state_count": data["state_count"],
        "evidence": cases,
        "interpretation": (
            "Isolating the two endpoints shows one tiny changed endpoint without a confirmed direction "
            "and one exact identity. The earlier combined replacement was an upstream region effect, "
            "so no RMS endpoint is promoted as a root cause."
        ),
    }


def selection_evidence() -> dict[str, Any]:
    data = read("results/property/numerical_coverage_v1/granite_selection_suite24_v1/summary.json")
    records = data["records"]
    return {
        "case_id": "granite_router_topk_sort",
        "source": "ATen Top-k/sort selection path with deterministic tie-order variant",
        "status": "NO_DIFFERENCE_OBSERVED_SELECTION_VARIANT_NOT_A_BIAS_CASE",
        "state_count": len(records),
        "all_scores_equal": all(row.get("same_scores") for row in records),
        "all_gradients_zero_difference": all(
            row["original_coordinate_statistics"]["PARAMETER_GRADIENT"]["effect_energy"] == 0
            for row in records
        ),
        "all_writes_zero_difference": all(
            row["original_coordinate_statistics"]["PARAMETER_WRITE"]["effect_energy"] == 0
            for row in records
        ),
        "selected_set_unchanged": all(row.get("selected_expert_set_equal", True) for row in records),
        "interpretation": (
            "Index order can change for equal scores while the selected expert set, values, gradients, "
            "writes and loss remain equal in the fixed suite. This is a useful semantic negative control, "
            "not a discovered bias."
        ),
    }


def granite_expert_evidence() -> dict[str, Any]:
    data = read(
        "results/property/training_numerical_analysis_v2/"
        "granite_expert_order_confirmation_v2/recomputed.json"
    )
    bias = data["bias_analysis"]
    raw = read(
        "results/property/training_numerical_analysis_v2/"
        "granite_expert_order_confirmation_v2/raw.json"
    )
    write_rows = confirmation_rows(raw, "PARAMETER_WRITE")
    nonzero_rows = [row for row in write_rows if row["effect_energy"] > 0]
    inner_products = [float(row.get("effect_repair_inner_product", 0.0)) for row in nonzero_rows]
    result = {
        "case_id": data["case_id"],
        "contrast_id": data["contrast_id"],
        "state_count": len(bias["confirmation_state_ids"]),
        "fixed_suite_total_rms": bias["fixed_suite_total_rms"],
        "fixed_suite_aligned_ratio_of_sums": bias["fixed_suite_aligned_ratio_of_sums"],
        "fixed_suite_gradient_rms": rms(raw, "PARAMETER_GRADIENT"),
        "fixed_suite_parameter_write_rms": rms(raw, "PARAMETER_WRITE"),
        "parameter_write_nonzero_confirmation_states": sum(
            row["effect_energy"] > 0 for row in confirmation_rows(raw, "PARAMETER_WRITE")
        ),
        "parameter_write_nonzero_inner_product_signs": [
            "positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in inner_products
        ],
        "parameter_write_direction_status": "CALIBRATION_DIRECTION_ORTHOGONAL_TO_ALL_CONFIRMATION_EFFECTS",
        "parameter_write_direction": heldout_direction_from_gram(raw["stages"]["PARAMETER_WRITE"]["EXACT"]["profile"]),
        "effect_repair_signs_role": "STATEWISE_ALIGNMENT_NOT_A_FIXED_DIRECTION_TEST",
        "equivalence_decision": data["equivalence_decision"],
        "training_outcome": data["training_outcome"],
    }
    population_path = ROOT / "results/property/granite_expert_order_population_v1/result.json"
    if population_path.is_file():
        population = read(str(population_path.relative_to(ROOT)))
        result["independent_population_confirmation"] = {
            "source": str(population_path.relative_to(ROOT)),
            "population": population.get("population"),
            "calibration_count": population.get("calibration_count"),
            "confirmation_count": population.get("confirmation_count"),
            "gradient": population.get("gradient"),
            "parameter_write": population.get("parameter_write"),
            "claim_boundary": population.get("claim_boundary"),
        }
    return result


def olmoe_router_accumulation_evidence() -> dict[str, Any]:
    data = read(
        "results/property/new_problem_group_search_v1/"
        "olmoe_router_accum_natural_16_20260919.json"
    )
    summaries = data["summaries"]
    return {
        "case_id": "olmoe_router_accum_natural_16_20260919",
        "contrast_id": "NATURAL_SAME_INPUT_ROUTER_COMBINE_ACCUMULATION",
        "state_count": data["state_count"],
        "model": data["model"],
        "carrier_scope": "layer-0 router gate parameters",
        "routing_and_weights_unchanged": all(
            row["routing_same"] and row["routing_weights_same"] for row in data["rows"]
        ),
        "gradient": summaries["gradient"],
        "parameter_write": summaries["write"],
        "claim_boundary": data["claim_boundary"],
        "interpretation": (
            "The real OLMoE router combine boundary is distinct from the Granite FP32 order "
            "control: it changes repeated-destination accumulation dtype while routing, expert "
            "selection and routing weights remain identical. The fixed 16-state confirmation has "
            "a negative aligned-write interval, while the held-out calibration-direction projection "
            "crosses zero; this supports a scoped aligned update bias, not a general additive mean "
            "or long-run loss claim."
        ),
    }


def rmsnorm_cast_materialization_evidence() -> dict[str, Any]:
    """Summarize the scoped natural RMSNorm cast-order intervention."""
    path = ROOT / "results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json"
    data = read(str(path.relative_to(ROOT)))
    gemma_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_rmsnorm_natural_16_20260918.json"
    gemma = read(str(gemma_path.relative_to(ROOT))) if gemma_path.is_file() else None
    qnorm_records = []
    for name in ("qwen3_qnorm_natural_16_20260918.json", "qwen3_knorm_natural_16_20260918.json"):
        probe_path = ROOT / "results/property/new_problem_group_search_v1" / name
        if probe_path.is_file():
            probe = read(str(probe_path.relative_to(ROOT)))
            qnorm_records.append(
                {
                    "source": str(probe_path.relative_to(ROOT)),
                    "parameter": probe["parameter"],
                    "states": probe["summary"]["state_count"],
                    "summary": probe["summary"],
                    "claim_boundary": probe["claim_boundary"],
                }
            )
    return {
        "status": data["status"],
        "source": data["source"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "models": data["models"],
        "claim_boundary": data["claim_boundary"],
        "interpretation": data["interpretation"],
        "qk_norm_confirmations": qnorm_records,
        "negative_control": data["negative_control"],
        "gemma4_confirmation": (
            {
                "source": str(gemma_path.relative_to(ROOT)),
                "states": len(gemma["rows"]),
                "summary": gemma["summaries"],
                "claim_boundary": gemma["claim_boundary"],
                "interpretation": (
                    "Gemma-4 native RMSNorm has the same FP32 weight-multiply/final-cast path; "
                    "the cast-before-weight reference yields a large scoped aligned write effect. "
                    "This is a third model confirmation of the existing materialization group, "
                    "not a distinct Gemma-only root."
                ),
            }
            if gemma is not None
            else None
        ),
    }


def bert_layernorm_compiled_materialization_evidence() -> dict[str, Any]:
    """Summarize the scoped real-checkpoint compiled/eager LayerNorm probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    eager_path = root / "bert_tiny_layernorm_eager_natural_32_20260918.json"
    fp32_path = root / "bert_tiny_layernorm_fp32_reference_20260918.json"
    shorter_path = root / "bert_tiny_layernorm_eager_natural_20260918.json"
    eager = read(str(eager_path.relative_to(ROOT)))
    fp32 = read(str(fp32_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": eager["status"],
        "model": eager["model"],
        "operator": eager["operator"],
        "candidate": eager["candidate"],
        "eager_reference": eager["reference"],
        "fp32_reference": fp32["reference"],
        "input_source": eager["input_source"],
        "claim_boundary": eager["claim_boundary"],
        "eager_32_state_summary": eager["summary"],
        "eager_16_state_summary": shorter["summary"] if shorter else None,
        "fp32_control_summary": fp32["summary"],
        "source_interpretation": (
            "The compiled native path agrees with the explicit FP32 reduction/materialisation "
            "control on the declared bank, while the eager native path differs. This closes "
            "the observed compiled-versus-eager LayerNorm boundary, not a universal compiler "
            "instruction-level cause."
        ),
    }


def bert_embedding_sum_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural BERT embedding-sum materialization probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "bert_embedding_sum_materialization_natural_32_20260918.json"
    shorter_path = root / "bert_embedding_sum_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The native BERT embedding block adds word, position and token-type embeddings "
            "in the native dtype, while the comparison path performs the same three-term sum "
            "in FP32 and casts once before the existing LayerNorm. On the real BERT-tiny text "
            "bank this isolated materialization boundary gives a stable negative aligned write "
            "effect, while the held-out additive direction crosses zero. This is a distinct "
            "embedding-sum boundary, not a second LayerNorm or embedding-backward case."
        ),
    }


def bert_pooler_tanh_materialization_evidence() -> dict[str, Any]:
    """Summarize the expanded natural BERT pooler-tanh screen."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    embedding_path = root / "bert_pooler_tanh_natural_cpu32_20260919.json"
    dense_path = root / "bert_pooler_tanh_dense_natural_cpu32_20260919.json"
    embedding = read(str(embedding_path.relative_to(ROOT)))
    dense = read(str(dense_path.relative_to(ROOT)))
    return {
        "status": embedding["status"],
        "model": embedding["model"],
        "operator": embedding["operator"],
        "embedding_carrier": embedding["parameter"],
        "dense_carrier": dense["parameter"],
        "candidate": embedding["candidate"],
        "reference": embedding["reference"],
        "comparison_scope": embedding["comparison_scope"],
        "embedding_state_count": embedding["state_count"],
        "dense_state_count": dense["state_count"],
        "embedding_summary": embedding["summary"],
        "dense_summary": dense["summary"],
        "claim_boundary": embedding["claim_boundary"],
        "source_interpretation": (
            "On 32 real document-derived BERT-tiny text windows, replacing only the pooler "
            "tanh boundary with FP32 tanh followed by the original BF16 write produces a "
            "positive held-out additive gradient direction at the embedding carrier. The "
            "same intervention does not produce a confirmed direction at pooler.dense.weight, "
            "so the result is a scoped carrier-specific tanh materialization bias, not a claim "
            "that every downstream parameter shares the effect or that it changes loss."
        ),
    }


def granite_residual_addition_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Granite MoE residual-addition probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "granite_residual_addition_materialization_natural_32_20260918.json"
    shorter_path = root / "granite_residual_addition_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "layer": primary["layer"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The Granite MoE decoder's native attention residual addition differs from an "
            "otherwise identical FP32 residual-plus-scaled-output addition followed by one "
            "native-dtype cast. On the real text bank the isolated boundary produces stable "
            "negative aligned scaling of the o_proj write, while the held-out additive direction "
            "crosses zero. This is distinct from the Granite expert-contribution accumulation "
            "order group because the expert routing, expert outputs and MoE combine are unchanged."
        ),
    }


def granite_moe_gate_product_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Granite MoE gate-times-up probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "granite_moe_gate_product_materialization_natural_32_20260918.json"
    shorter_path = root / "granite_moe_gate_product_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "layer": primary["layer"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The Granite MoE expert gate-times-up product differs when the product alone is "
            "evaluated in FP32 and cast once, while routing, activation, expert projections, "
            "output gating and repeated-destination combine remain unchanged. The 32-state bank "
            "shows stable negative aligned write scaling and a positive held-out calibration-direction "
            "projection interval. This is distinct from expert-contribution accumulation order and "
            "from the attention residual-addition boundary."
        ),
    }


def granite_moe_output_gate_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Granite MoE output-gate product probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "granite_moe_output_gate_materialization_natural_32_20260918.json"
    shorter_path = root / "granite_moe_output_gate_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "layer": primary["layer"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The Granite MoE expert output times routing-gate product differs when this "
            "product alone is evaluated in FP32 and cast once, while routing, expert projections, "
            "gate-times-up product and repeated-destination combine remain unchanged. The 32-state "
            "bank shows stable negative aligned write scaling, but the held-out additive direction "
            "crosses zero. This is distinct from the gate-times-up product and expert accumulation "
            "order boundaries."
        ),
    }


def granite_router_score_materialization_evidence() -> dict[str, Any]:
    """Summarize the fixed-selection Granite router-score probe.

    The ordinary router comparison changes both score precision and, on many
    states, the selected expert set.  This record freezes the native top-k
    indices, so the retained effect is the score/softmax-weight materialization
    boundary rather than a routing-set change.
    """
    path = ROOT / "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json"
    data = read(str(path.relative_to(ROOT)))
    values = [float(x) for x in data["summary"]["confirmation_aligned_write_values"]]
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (n - 1))
    from scipy.stats import t
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "layer": data["layer"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "summary": data["summary"],
        "confirmation_aligned_mean": mean,
        "confirmation_aligned_interval_normal_95": [mean - half, mean + half],
        "confirmation_positive_count": sum(x > 0.0 for x in values),
        "confirmation_negative_count": sum(x < 0.0 for x in values),
        "source_interpretation": (
            "The native Granite router computes its score projection in model dtype before "
            "the float conversion used by routing.  With the native top-k indices frozen, "
            "recomputing only the score projection and softmax weights in FP32 leaves the "
            "selected expert set and expert body unchanged, yet produces a negative aligned "
            "router-write effect on 15 of 16 confirmation states.  This isolates a router "
            "score/weight materialization bias from the separate Granite expert-contribution "
            "order group and from the natural top-k selection changes seen without freezing."
        ),
        "claim_boundary": (
            "Fixed native top-k selection, one Granite checkpoint, 32 real text windows, "
            "and the declared layer-0 router carrier; no external-population or loss claim."
        ),
    }


def gemma3_vision_patch_convolution_evidence() -> dict[str, Any]:
    """Summarize the clean same-input Gemma vision convolution probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "module": data["module"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "source_interpretation": (
            "The same real Gemma-3 model and image/text inputs were run with the native BF16 "
            "patch convolution and with only that convolution replaced by FP32 accumulation "
            "followed by native-dtype write-back. The confirmation half has a negative aligned "
            "write interval, while the held-out additive direction remains non-confirmed; this "
            "supports an operator-local aligned scaling result, not an additive mean or quality claim."
        ),
    }


def qwen3vl_position_interpolation_evidence() -> dict[str, Any]:
    """Summarize the real-loss Qwen3-VL learned-position interpolation probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_16_20260918.json"
    data = read(str(path.relative_to(ROOT)))
    ratios = [float(row["aligned_ratio"]) for row in data["rows"]]
    mean = sum(ratios) / len(ratios)
    variance = sum((value - mean) ** 2 for value in ratios) / (len(ratios) - 1)
    half_width = 2.145 * math.sqrt(variance / len(ratios))
    return {
        "status": data["status"],
        "model": data["model"],
        "target_parameter": data["target_parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "image_count": data["image_count"],
        "calibration_count": data["calibration_count"],
        "confirmation_count": data["confirmation_count"],
        "fixed_suite_mean_write_rms": data["fixed_suite_mean_write_rms"],
        "aligned_ratio_mean": mean,
        "aligned_ratio_normal_95_interval": [mean - half_width, mean + half_width],
        "confirmation_projection_mean": sum(data["confirmation_projections"]) / len(data["confirmation_projections"]),
        "confirmation_positive_count": data["confirmation_positive_count"],
        "confirmation_negative_count": data["confirmation_negative_count"],
        "claim_boundary": (
            "The source is isolated on the declared Qwen3-VL vision/language loss path and "
            "16 real image states: native-dtype four-tap position interpolation versus FP32 "
            "weighted summation followed by the original dtype write. The aligned write interval "
            "is descriptive for this fixed image bank; the additive direction is mixed and no "
            "long-run quality or broader population claim is made."
        ),
        "source_interpretation": (
            "The nonzero effect begins in the arithmetic used to materialize the learned 2-D "
            "position embedding, before the unchanged visual blocks and language loss. This is "
            "distinct from rotary position arithmetic, patch convolution, GELU, and patch-merger "
            "layout: the latter was screened to exact identity on a real Ministral checkpoint."
        ),
    }


def mamba_causal_conv_accumulation_evidence() -> dict[str, Any]:
    """Summarize the natural Mamba causal-convolution evaluation probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "mamba_causal_conv_materialization_natural_32_20260918.json"
    shorter_path = root / "mamba_causal_conv_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The native Mamba causal depthwise convolution and an otherwise identical FP32 "
            "convolution reference differ at a single convolution boundary. On the 32-state "
            "document bank, the held-out write direction is negative and the aligned interval "
            "is strictly negative. This is a natural causal-convolution evaluation/materialization case; it "
            "shares the broad accumulation-precision mechanism with the Gemma patch convolution "
            "but is a distinct operator family and execution boundary."
        ),
    }


def mamba_state_output_contraction_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Mamba state-to-output contraction probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "mamba_state_output_contraction_natural_32_20260918.json"
    shorter_path = root / "mamba_state_output_contraction_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The native sequential Mamba scan and an otherwise identical scan that keeps only "
            "the state-to-C output contraction in FP32 differ at one materialization boundary. "
            "On the 32-state document bank the confirmation aligned-write interval is strictly "
            "negative, while the held-out additive direction interval crosses zero. This supports "
            "a scoped aligned state-output contraction result, not an additive population or loss claim."
        ),
    }


def mamba_d_skip_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Mamba residual D skip-product probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "mamba_d_skip_materialization_natural_32_20260918.json"
    shorter_path = root / "mamba_d_skip_materialization_natural_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The native sequential Mamba scan and an otherwise identical scan that keeps only "
            "the residual D*hidden_states product in FP32 differ at one output materialization "
            "boundary. On the 32-state document bank the confirmation aligned-write interval is "
            "strictly negative, while the held-out additive direction interval crosses zero. This "
            "supports a scoped aligned skip-product result, not an additive population or loss claim."
        ),
    }


def mamba_z_gate_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Mamba z-gate product probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary_path = root / "mamba_z_gate_materialization_wikitext_32_20260918.json"
    shorter_path = root / "mamba_z_gate_materialization_wikitext_16_20260918.json"
    primary = read(str(primary_path.relative_to(ROOT)))
    shorter = read(str(shorter_path.relative_to(ROOT))) if shorter_path.is_file() else None
    return {
        "status": primary["status"],
        "model": primary["model"],
        "operator": primary["operator"],
        "layer": primary["layer"],
        "parameter": primary["parameter"],
        "candidate": primary["candidate"],
        "reference": primary["reference"],
        "input_source": primary["input_source"],
        "claim_boundary": primary["claim_boundary"],
        "summary": primary["summary"],
        "shorter_bank_summary": shorter["summary"] if shorter else None,
        "source_interpretation": (
            "The Mamba sequential scan's final scan-output times SiLU(z) gate differs when "
            "that product alone is evaluated in FP32 and cast once, while the recurrence, state/C "
            "contraction and D skip path remain unchanged. The real text bank shows stable negative "
            "aligned write scaling on a real Wikitext token bank, while the held-out additive "
            "direction crosses zero. This is a distinct z-gate boundary from the existing "
            "state-output, D-skip, softplus and causal-convolution groups."
        ),
    }


def deepseek_embedding_backward_accumulation_evidence() -> dict[str, Any]:
    """Summarize the real-checkpoint embedding backward accumulation probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "summary": data["summary"],
        "comparison_scope": data["comparison_scope"],
        "source_interpretation": (
            "The native embedding backward and same-input FP32 index-add reference have identical "
            "forward loss, while repeated-token gradient accumulation produces a stable negative "
            "aligned parameter-write scaling on the declared DeepSeek/Qwen3 bank. The result is "
            "a fixed-suite natural aligned-bias finding, not a loss-quality or population claim."
        ),
    }


def deepseek_embedding_gradient_materialization_evidence() -> dict[str, Any]:
    """Summarize the same-cotangent embedding-gradient materialization probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "source_interpretation": (
            "The embedding cotangent is captured once from the real DeepSeek/Qwen3 model and "
            "the repeated-index accumulation is held identical. Only the final FP32 gradient "
            "retention versus BF16 materialization changes. The confirmation aligned-write "
            "interval is strictly negative, but the write RMS is about 3e-8, so this is a "
            "source-closed micro-bias and not a practically important quality result."
        ),
    }


def deepseek_fused_embedding_nll_partial_materialization_evidence() -> dict[str, Any]:
    """Summarize the generated fused-boundary partial-materialization probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    term_path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_terms_v7_20260920.json"
    term_data = read(str(term_path.relative_to(ROOT))) if term_path.is_file() else None
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference_variants": data["reference_variants"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "term_boundary_summary": term_data["summary"] if term_data else None,
        "full_fp32_replay_summary": (
            read("results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_fp32_replay_20260920.json")["summary"]
            if (ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_fp32_replay_20260920.json").is_file()
            else None
        ),
        "source_interpretation": (
            "The real generated fused embedding/NLL/normalization backward call boundary is "
            "held fixed. The original partial-term probe changes only pre-reduction materialization, "
            "while the FP64-reduction control is nearly identity. The follow-up term probe also "
            "changes only the base-term or final-value materialization; both remain strictly negative "
            "over 16 real text states (about 2.61% and 1.68% RMS). These boundaries are therefore "
            "subcomponents of one fused intermediate-materialization problem family, not extra model "
            "or operator groups. The full FP32 expression replay is also near identity, so the visible "
            "fused arithmetic and reduction schedule are not sufficient explanations of the much larger "
            "region screen. The evidence still does not assign every source of the larger fused region to "
            "this family."
        ),
    }


def deepseek_fused_embedding_nll_reduction_order_evidence() -> dict[str, Any]:
    """Summarize the same-boundary FP32 reverse-order probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_reverse_v5_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference_variants": data["reference_variants"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "source_interpretation": (
            "The compiled fused call and all operands are held fixed. The reverse-order variant "
            "keeps FP32 term arithmetic and changes only the reduction element order; its small "
            "positive aligned interval is distinct from the BF16-partial variant, while the FP64 "
            "control is near identity. This closes a same-precision non-associative reduction "
            "boundary at the declared fused call, without a quality or population claim."
        ),
    }


def gemma4_causal_nll_loss_evaluation_evidence() -> dict[str, Any]:
    """Summarize the same-logits Gemma-4 causal-NLL boundary probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    data = read(str((root / "gemma4_softcapped_nll_24_20260918.json").relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "source_interpretation": (
            "The model forward and logits are held fixed. The candidate uses the native "
            "cross-entropy loss path, while the reference uses explicit FP32 shifted "
            "log-softmax/gather. A stable aligned write effect therefore identifies the "
            "NLL-loss-evaluation boundary, not the upstream soft-capped-logit computation "
            "and not the Liger fused-linear dW accumulation boundary."
        ),
    }


def bert_nll_loss_evaluation_evidence() -> dict[str, Any]:
    """Summarize the expanded BERT-tiny masked-LM NLL boundary probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    summary = data["summary"]
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["claim_boundary"],
        "summary": summary,
        "source_interpretation": (
            "The BERT-tiny compiled logits, model parameters, and optimizer state are held fixed. "
            "Only native cross-entropy evaluation is replaced by explicit FP32 shifted "
            "log-softmax/gather on the identical logits. Across 128 real document windows, "
            "the confirmation aligned-write interval is strictly positive while the held-out "
            "additive direction and loss intervals cross zero. This closes a scoped NLL-evaluation "
            "source for the BERT masked-LM boundary, without claiming a population additive mean, "
            "a loss-quality consequence, or a lower-level kernel instruction cause. The earlier "
            "32-window README screen remains an inconclusive development result and is superseded "
            "for promotion by this larger declared document bank."
        ),
    }


def gemma4_final_logit_softcap_materialization_evidence() -> dict[str, Any]:
    """Summarize the expanded Gemma-4 final-logit parameter probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    data = read(str((root / "gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json").relative_to(ROOT)))
    logits_path = root / "gemma4_final_logit_softcap_logits_26_20260919.json"
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "logit_gradient_summary": read(str(logits_path.relative_to(ROOT)))["summary"] if logits_path.is_file() else None,
        "source_interpretation": (
            "The hidden state and lm_head are held fixed while only the final-logit soft-cap "
            "arithmetic changes. Across all 26 real text states, the logit-gradient endpoint "
            "has positive aligned effect in every state, while the 13-state confirmation half "
            "has 12 negative and one positive aligned write values, with a descriptive normal "
            "95% interval entirely below zero. This closes a fixed-suite natural aligned-bias "
            "result at the final-logit materialization boundary; it is distinct from the downstream "
            "Gemma-4 NLL evaluation group and does not claim a population or loss consequence."
        ),
    }


def gemma4_audio_output_projection_materialization_evidence() -> dict[str, Any]:
    """Summarize the real-audio full-language-loss projection probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_16_20260919.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "source_interpretation": (
            "The real Gemma-4 audio encoder, text decoder and labels are shared. Only the "
            "audio output projection changes from native BF16 linear arithmetic to FP32 "
            "accumulation followed by BF16 writeback. The resulting nonzero loss and projection "
            "gradient/write differences on real speech make this a natural audio-language "
            "boundary result. The eight-state confirmation aligned-write interval is entirely "
            "negative, while the held-out additive projection interval crosses zero; this is "
            "not a population or long-run quality claim."
        ),
    }


def gemma4_audio_attention_softcap_materialization_evidence() -> dict[str, Any]:
    """Summarize the single-layer natural audio-attention softcap probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_softcap_layer0_natural_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    rows = data["rows"]
    split = int(data["summary"]["calibration_count"])
    confirmation = rows[split:]
    confirm_bf16 = [float(row["write_aligned_bf16"]) for row in confirmation]
    confirm_fp64 = [float(row["write_aligned_fp64"]) for row in confirmation]
    def descriptive_interval(values: list[float]) -> list[float]:
        if len(values) < 2:
            return [values[0], values[0]]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
        return [mean - half, mean + half]
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "layer": data["layer"],
        "target_parameter": data["target_parameter"],
        "candidate": data["candidate"],
        "reference_variants": data["reference_variants"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": {
            **data["summary"],
            "confirmation_bf16_aligned_interval": descriptive_interval(confirm_bf16),
            "confirmation_fp64_aligned_interval": descriptive_interval(confirm_fp64),
            "confirmation_bf16_positive": sum(value > 0 for value in confirm_bf16),
            "confirmation_bf16_negative": sum(value < 0 for value in confirm_bf16),
            "confirmation_fp64_positive": sum(value > 0 for value in confirm_fp64),
            "confirmation_fp64_negative": sum(value < 0 for value in confirm_fp64),
        },
        "source_interpretation": (
            "The real Gemma-4 audio-language graph, waveforms, text and labels are shared. "
            "Only the softcap arithmetic in audio_tower.layers.0.self_attn is changed: the "
            "native path evaluates (logits / 50) -> tanh -> *50 in FP32, while the BF16 "
            "variant rounds the softcap branch to BF16 and the FP64 variant evaluates that "
            "branch in FP64 before returning to FP32. Both variants change the actual q_proj "
            "gradient and first-step write on every retained state, with large fixed-suite total "
            "effects. The full 16-state descriptive aligned intervals are negative, but the held-out "
            "eight-state intervals cross zero; therefore this closes the numerical source and "
            "parameter reachability, not a population mean-bias or quality claim. This is a new "
            "audio-attention softcap boundary, distinct from final-logit softcap and audio output projection."
        ),
    }


def gemma4_audio_attention_softmax_probability_materialization_evidence() -> dict[str, Any]:
    """Summarize the layer-scoped real-audio probability-materialization probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_natural_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    layer1_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_layer1_natural_16_20260920.json"
    layer1 = read(str(layer1_path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "layer": data["layer"],
        "target_parameter": data["target_parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": data["summary"],
        "layer1_heterogeneity": {
            "layer": layer1["layer"],
            "summary": layer1["summary"],
            "claim_boundary": layer1["claim_boundary"],
        },
        "source_interpretation": (
            "The real Gemma-4 audio-language graph, logits, softcap, text, labels and downstream "
            "value contraction are shared. The only changed boundary is whether the native FP32 "
            "attention probabilities are retained in FP32 or materialized to BF16 before the value "
            "contraction. On layer 0 this single intervention produces a strictly negative aligned "
            "q_proj write interval on the 16-state fixed suite, while layer 1 crosses zero; the "
            "result is therefore a layer-0-scoped source-closed aligned-bias case, not a universal "
            "audio-attention or population claim."
        ),
    }


def gemma4_audio_lightconv_glu_product_materialization_evidence() -> dict[str, Any]:
    """Summarize the real-audio LightConv GLU product probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    paths = [
        root / "gemma4_audio_glu_layer0_natural_8_20260919.json",
        root / "gemma4_audio_glu_layer0_natural_16_20260919.json",
    ]
    data = [read(str(path.relative_to(ROOT))) for path in paths]
    confirm = data[-1]["summary"]
    return {
        "status": data[-1]["status"],
        "model": data[-1]["model"],
        "operator_family": data[-1]["operator_family"],
        "parameter": data[-1]["parameter"],
        "candidate": data[-1]["candidate"],
        "reference": data[-1]["reference"],
        "input_source": data[-1]["input_source"],
        "comparison_scope": data[-1]["comparison_scope"],
        "claim_boundary": data[-1]["claim_boundary"],
        "summary": confirm,
        "source_interpretation": (
            "The real Gemma-4 audio-language graph and speech waveforms are shared. Only the "
            "LightConv1d GLU gate-times-value product changes: the candidate multiplies in the "
            "native activation dtype, while the reference performs the product in FP32 and writes "
            "back once in the original dtype. Across 16 real-speech states the first-step write "
            "effect is large and the eight-state confirmation aligned interval is entirely negative. "
            "The held-out additive projection interval crosses zero, so this closes a fixed-suite "
            "aligned-scaling result at a distinct GLU product boundary, not a population mean-bias "
            "or long-run loss claim."
        ),
    }


def gemma4_audio_lightconv_depthwise_conv_backward_accumulation_evidence() -> dict[str, Any]:
    """Summarize the natural audio depthwise-convolution backward probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    path = root / "gemma4_audio_depthwise_conv_layer0_natural_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    rows = data["rows"]
    split = int(data["summary"]["calibration_count"])
    confirmation = rows[split:]
    numerator = sum(float(row["aligned_write_numerator"]) for row in confirmation)
    denominator = sum(float(row["aligned_write_denominator"]) for row in confirmation)
    leave_one_out = [
        (numerator - float(row["aligned_write_numerator"]))
        / max(denominator - float(row["aligned_write_denominator"]), 1e-30)
        for row in confirmation
    ]
    boundary_path = root / "gemma4_audio_depthwise_conv_boundary_check_20260920.json"
    boundary = read(str(boundary_path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": {
            **data["summary"],
            "confirmation_aligned_write_ratio_of_sums_recomputed": numerator / max(denominator, 1e-30),
            "confirmation_leave_one_out_ratio_of_sums_min": min(leave_one_out),
            "confirmation_leave_one_out_ratio_of_sums_max": max(leave_one_out),
        },
        "boundary_check": boundary,
        "source_interpretation": (
            "The real Gemma-4 audio-language graph and speech bank are shared. Only the "
            "LightConv1d depthwise causal-convolution boundary changes: the candidate uses "
            "the native Conv1d backward, while the reference evaluates the same padded windows "
            "with explicit FP32 multiply-accumulate and writes back in the original activation "
            "dtype. The boundary check shows matching window semantics and nearly identical "
            "forward outputs; the large state-dependent effects therefore arise primarily in "
            "the weight-gradient accumulation path. Under the protocol's ratio-of-sums aligned "
            "estimand, the eight-state confirmation is +1.975%, and deleting any one confirmation "
            "state leaves it positive. The statewise ratios are heterogeneous, so this is a "
            "fixed-suite aligned-bias result, not a population mean or long-run quality claim."
        ),
    }


def gemma4_audio_subsampling_convolution_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural audio subsampling Conv2d boundary probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    path = root / "gemma4_audio_subsample_conv_layer1_natural_raw_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    rows = data["rows"]
    split = int(data["summary"]["calibration_count"])
    confirmation = rows[split:]
    numerator = sum(float(row["aligned_write_numerator"]) for row in confirmation)
    denominator = sum(float(row["aligned_write_denominator"]) for row in confirmation)
    leave_one_out = [
        (numerator - float(row["aligned_write_numerator"]))
        / max(denominator - float(row["aligned_write_denominator"]), 1e-30)
        for row in confirmation
    ]
    boundary_path = root / "gemma4_audio_subsample_conv_layer1_boundary_check_20260920.json"
    boundary = read(str(boundary_path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "comparison_scope": data["comparison_scope"],
        "claim_boundary": data["claim_boundary"],
        "summary": {
            **data["summary"],
            "confirmation_aligned_write_ratio_of_sums_recomputed": numerator / max(denominator, 1e-30),
            "confirmation_leave_one_out_ratio_of_sums_min": min(leave_one_out),
            "confirmation_leave_one_out_ratio_of_sums_max": max(leave_one_out),
        },
        "boundary_check": boundary,
        "source_interpretation": (
            "The real Gemma-4 audio-language graph and speech bank are shared. Only the layer-1 "
            "audio subsampling Conv2d materialization boundary changes: the candidate uses native "
            "BF16 convolution while the reference evaluates the same captured window in FP32 and "
            "writes back in BF16. The three-state captured-input boundary check shows identical "
            "window shapes and native-vs-explicit forward relative RMS below 2.4e-5. The large "
            "parameter-write effect is therefore not an input or padding mismatch. The eight-state "
            "confirmation ratio-of-sums aligned scaling is about -38.0%, and deleting any one state "
            "leaves it negative. Statewise values are heterogeneous, so this is a fixed-suite "
            "ratio-of-sums aligned-bias result, not a population mean or quality claim."
        ),
    }


def deberta_disentangled_relative_attention_materialization_evidence() -> dict[str, Any]:
    """Summarize the DeBERTa c2p/p2c relative-attention probes."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    paths = [
        root / "deberta_disentangled_attention_materialization_natural_layer0_seed0_32_20260919.json",
        root / "deberta_disentangled_attention_materialization_natural_layer5_32_20260919.json",
    ]
    data = [read(str(path.relative_to(ROOT))) for path in paths]
    return {
        "status": "COMPLETE",
        "model": data[0]["model"],
        "layers": [item["layer"] for item in data],
        "operator": data[0]["operator"],
        "candidate": data[0]["candidate"],
        "reference": data[0]["reference"],
        "input_source": data[0]["input_source"],
        "claim_boundary": data[0]["claim_boundary"],
        "layer_summaries": [item["summary"] for item in data],
        "source_interpretation": (
            "The same DeBERTa-v3-small backbone and real-text windows are evaluated with native "
            "c2p/p2c disentangled relative-position score products or FP32 score products followed "
            "by native-dtype return. Both an early and the final encoder layer show a negative "
            "confirmation aligned-write interval. The local checkpoint lacks a pretrained MLM head, "
            "so the result is a declared backbone/auxiliary-head probe and not a pretrained MLM-quality claim."
        ),
    }


def bloom_alibi_attention_materialization_evidence() -> dict[str, Any]:
    """Summarize the Bloom ALiBi attention score probes."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    paths = [
        root / "bloom_alibi_attention_materialization_natural_16_20260919.json",
        root / "bloom_alibi_attention_materialization_natural_layer11_16_20260919.json",
    ]
    data = [read(str(path.relative_to(ROOT))) for path in paths]
    return {
        "status": "COMPLETE",
        "model": data[0]["model"],
        "layers": [item["layer"] for item in data],
        "operator": data[0]["operator"],
        "candidate": data[0]["candidate"],
        "reference": data[0]["reference"],
        "input_source": data[0]["input_source"],
        "claim_boundary": data[0]["claim_boundary"],
        "layer_summaries": [item["summary"] for item in data],
        "source_interpretation": (
            "The same Bloom-560m checkpoint, ALiBi tensor, causal mask and real-text windows are "
            "evaluated with native baddbmm score materialization or explicit FP32 QK multiplication "
            "and ALiBi addition before native-dtype return. Two layers reproduce a negative "
            "confirmation aligned-write interval; this is an ALiBi score-boundary result, not a "
            "population or long-run loss claim."
        ),
    }


def gemma4_rms_row_reduction_order_evidence() -> dict[str, Any]:
    """Summarize the Gemma-4 same-input row-reduction-order probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    data = read(str((root / "gemma4_rms_row_reduction_natural_26_20260919.json").relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "layer": data["layer"],
        "target": data["target"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_bank": data["input_bank"],
        "optimizer": data["optimizer"],
        "summary": data["summary"],
        "claim_boundary": data["claim_boundary"],
        "source_interpretation": (
            "The real Gemma-4 text graph and target are shared. Only the FP32 row square-sum "
            "reduction order changes. Across 26 states the confirmation aligned-write values "
            "are all negative with an interval entirely below zero. This is distinct from the "
            "existing RMSNorm cast-before-weight multiplication group; it is a declared "
            "model-level reduction-order variant, not proof of a released kernel's internal order."
        ),
    }


def mamba_softplus_materialization_evidence() -> dict[str, Any]:
    """Summarize the path-preserving Mamba softplus precision probe."""
    root = ROOT / "results/property/new_problem_group_search_v1"
    primary = root / "mamba_softplus_materialization_natural_32_20260918.json"
    shorter = root / "mamba_softplus_materialization_natural_16_20260918.json"
    data = read(str(primary.relative_to(ROOT)))
    short = read(str(shorter.relative_to(ROOT))) if shorter.is_file() else None
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "claim_boundary": data["claim_boundary"],
        "state_summary": data["summary"],
        "short_bank_summary": short["summary"] if short else None,
        "source_interpretation": (
            "The same sequential Mamba recurrence and real model states were used on both sides; "
            "only delta softplus evaluation was changed from input-dtype arithmetic to FP32 "
            "followed by the original dtype cast. The confirmation aligned-write interval is "
            "negative on both the 16-state and 32-state banks, closing this scoped softplus "
            "materialisation source. The held-out vector-direction interval crosses zero, so "
            "this is an aligned scaling result rather than a full additive mean-bias claim."
        ),
    }


def mamba_discrete_transition_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural Mamba exp(A*delta) transition probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/mamba_discrete_transition_materialization_natural_cpu16_20260919.json"
    data = read(str(path.relative_to(ROOT)))
    return {
        "status": data["status"],
        "model": data["model"],
        "operator": data["operator"],
        "parameter": data["parameter"],
        "candidate": data["candidate"],
        "reference": data["reference"],
        "input_source": data["input_source"],
        "claim_boundary": data["claim_boundary"],
        "state_summary": data["summary"],
        "source_interpretation": (
            "The sequential Mamba model, real text windows and all downstream recurrence "
            "operations are shared. Only the exp(A*delta) transition materialization changes: "
            "the reference rounds the exponent argument to the model dtype before exp and then "
            "returns to FP32 for the recurrence. The 16-state confirmation has a strictly "
            "negative aligned-write interval, while the held-out additive direction crosses "
            "zero. This closes a scoped aligned transition-materialization source, not a full "
            "additive mean-bias or loss-quality claim."
        ),
    }


def rwkv_time_decay_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural RWKV time-mix decay materialization probe."""
    path = ROOT / "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json"
    data = read(str(path.relative_to(ROOT)))
    modes = data["summary"]["modes"]
    return {
        "status": data["status"],
        "model": data["model"],
        "operator_family": data["operator_family"],
        "candidate": data["candidate"],
        "target_parameter": data["target_parameter"],
        "input_source": data["input_source"],
        "state_count": data["summary"]["state_count"],
        "decay_fp32": modes["decay_fp32"],
        "value_fp32": modes["value_fp32"],
        "all_fp32": modes["all_fp32"],
        "claim_boundary": data["claim_boundary"],
        "source_interpretation": (
            "The RWKV-4-world block-0 time-mix recurrence is replayed on the same real text "
            "states with the recurrent boundary held fixed. Changing only the time_decay "
            "exponential to FP32 reproduces the all-FP32 write effect on all 16 states, while "
            "changing only recurrent value materialization does not. The source is therefore "
            "closed at the time_decay exponential/materialization boundary for this declared "
            "checkpoint and bank. The write result is a fixed-suite aligned effect; no population "
            "or multi-step loss claim is made."
        ),
    }


def rwkv_receptance_sigmoid_materialization_evidence() -> dict[str, Any]:
    """Summarize the natural RWKV receptance-sigmoid probe."""
    paths = [
        "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_natural_32_20260920.json",
        "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_method_32_20260920.json",
    ]
    records = [read(path) for path in paths if (ROOT / path).is_file()]
    if len(records) != 2:
        raise AssertionError("both RWKV receptance-sigmoid confirmation banks are required")
    return {
        "status": records[0]["status"],
        "model": records[0]["model"],
        "operator_family": records[0]["operator_family"],
        "candidate": records[0]["candidate"],
        "reference": records[0]["reference"],
        "target_parameter": records[0]["target_parameter"],
        "banks": [
            {
                "input_source": record["input_source"],
                "summary": record["summary"],
                "claim_boundary": record["claim_boundary"],
            }
            for record in records
        ],
        "source_interpretation": (
            "The same RWKV attention block, model weights and real-text states are replayed while "
            "only receptance sigmoid evaluation changes from native model dtype to FP32 followed by "
            "the original dtype write-back. Both independent 32-state text banks have all 16 held-out "
            "aligned values negative, with confirmation intervals strictly below zero. This closes "
            "the sigmoid materialization source for the declared block and banks; it is a fixed-suite "
            "aligned update result and does not claim a population or multi-step loss effect."
        ),
    }


def rotary_arithmetic_probe_evidence() -> dict[str, Any]:
    """Summarize the same-operand RoPE arithmetic source probe.

    The probe deliberately uses synthetic operands.  It can exclude a
    trigonometric implementation difference on the tested device, but cannot
    replace a natural-model capture or prove a training-population claim.
    """
    probe_paths = [
        "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v4.json",
        "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v3.json",
    ]
    probe_path = next(path for path in probe_paths if (ROOT / path).is_file())
    data = read(probe_path)
    source_contract = read(
        "results/property/root_cause_closure_v1/rotary_source_contract_v1.json"
    )
    natural_path = ROOT / "results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json"
    natural = read(str(natural_path.relative_to(ROOT))) if natural_path.is_file() else None
    qwen3_path = ROOT / "results/property/new_problem_group_search_v1/qwen3_rope_layer13_kproj_16_20260918.json"
    qwen3 = read(str(qwen3_path.relative_to(ROOT))) if qwen3_path.is_file() else None
    vision_rope_path = ROOT / "results/property/new_problem_group_search_v1/ministral_vision_rope_materialization_natural_cpu8_20260919.json"
    vision_rope = read(str(vision_rope_path.relative_to(ROOT))) if vision_rope_path.is_file() else None
    factorial_path = ROOT / "results/property/root_cause_closure_v1/ministral_vision_rope_source_factorial_natural_8_20260920.json"
    factorial = read(str(factorial_path.relative_to(ROOT))) if factorial_path.is_file() else None
    rows = []
    for row in data["rows"]:
        rows.append({
            "position_scale": row["position_scale"],
            "tl_math_vs_libdevice_exact": row["tl_math_vs_libdevice"]["exact"],
            "tl_math_vs_fp32_relative_l2": row["tl_math_vs_fp32_formula"]["relative_l2"],
            "tl_math_vs_fp64_relative_l2": row["tl_math_vs_fp64_formula"]["relative_l2"],
            "scale_terms_order_relative_l2": row["tl_math_vs_scale_terms_order"]["relative_l2"],
            "rounded_sum_materialization_relative_l2": row["tl_math_vs_rounded_sum_materialization"]["relative_l2"],
        })
    result = {
        "probe_record": probe_path,
        "status": data["status"],
        "input_policy": data["input_policy"],
        "rows": rows,
        "all_trigonometric_variants_exact": all(
            row["tl_math_vs_libdevice_exact"] for row in rows
        ),
        "saved_source_contract": {
            "status": source_contract["status"],
            "symbol": source_contract["symbol"],
            "input_contract": source_contract["input_contract"],
            "operation_order": source_contract["operation_order"],
            "source_candidates_left_after_probe": source_contract["source_candidates_left_after_probe"],
            "excluded_on_tested_device": source_contract["excluded_on_tested_device"],
        },
        "interpretation": (
            "On the tested CUDA device and identical operands, tl_math and libdevice "
            "produce identical BF16 outputs; the remaining difference is consistent "
            "with finite-precision formula evaluation/materialization rather than a "
            "choice between those two trigonometric entry points."
        ),
    }
    if natural is not None:
        store_tolerance = natural.get(
            "final_store_tolerance", {"max_abs": 0.0078125, "relative_l2": 2.0e-5}
        )
        result["natural_intermediate_probe"] = {
            "source": str(natural_path.relative_to(ROOT)),
            "status": natural["status"],
            "state_count": natural["state_count"],
            "kernel_symbol": natural.get("kernel_symbol"),
            "final_store_tolerance": store_tolerance,
            "final_output0_within_tolerance": all(
                row["final_output0_alignment"]["max_abs"] <= store_tolerance["max_abs"]
                and row["final_output0_alignment"]["relative_l2"] <= store_tolerance["relative_l2"]
                for row in natural["rows"]
            ),
            "final_output1_within_tolerance": all(
                row["final_output1_alignment"]["max_abs"] <= store_tolerance["max_abs"]
                and row["final_output1_alignment"]["relative_l2"] <= store_tolerance["relative_l2"]
                for row in natural["rows"]
            ),
            "first_difference_candidates": natural["first_difference_candidates"],
            "claim_boundary": natural["claim_boundary"],
        }
        result["natural_source_interpretation"] = (
            "On four natural Ministral states, the reviewed fused expression reproduces both "
            "candidate BF16 stores within the declared BF16 write tolerance; phase, cosine, "
            "sine, rotated value and position scale all align before the final fused output "
            "multiplication. This closes the "
            "source boundary for the reviewed endpoint at the final fused evaluation/materialization "
            "boundary, but does not identify a single causal operation or machine instruction, nor "
            "establish a natural population mean bias."
        )
    if qwen3 is not None:
        result["qwen3_standard_rope_confirmation"] = {
            "source": str(qwen3_path.relative_to(ROOT)),
            "model": qwen3["model"],
            "layer": qwen3["layer"],
            "parameter": qwen3["parameter"],
            "summary": qwen3["summary"],
            "claim_boundary": qwen3["claim_boundary"],
            "interpretation": (
                "A Qwen3 native rotary helper versus explicit FP32 same-formula reference "
                "produces a fixed natural-bank aligned write effect. This is treated as a "
                "confirmation of the broader RoPE materialization family, not as a second "
                "problem group; the Ministral fused position-scaling source boundary remains "
                "separately unresolved."
            ),
        }
    if vision_rope is not None:
        result["ministral_vision_rope_confirmation"] = {
            "source": str(vision_rope_path.relative_to(ROOT)),
            "operator": vision_rope["operator"],
            "parameter": vision_rope["parameter"],
            "summary": vision_rope["summary"],
            "claim_boundary": vision_rope["claim_boundary"],
            "interpretation": (
                "A real Ministral multimodal loss replay changes only the vision two-dimensional "
                "RoPE intermediate rotation precision and produces a negative aligned first-step "
                "write interval on the patch-convolution carrier. This is an additional natural "
                "confirmation of the rotary materialization family, not a duplicate problem group."
            ),
        }
    if factorial is not None:
        result["natural_source_factorial"] = {
            "source": str(factorial_path.relative_to(ROOT)),
            "state_count": factorial.get("state_count"),
            "summary": factorial.get("summary"),
            "interpretation": (
                "On the real Ministral vision endpoint, increasing only trig precision while "
                "keeping BF16 rotation is exact identity, whereas FP32 multiply/add followed "
                "by BF16 write-back reproduces the native-to-reference aligned effect on all "
                "eight states. This is the decisive source-factorial result for the declared "
                "vision boundary; it does not by itself prove a population or text-path claim."
            ),
        }
    return result


def attention_region_source_evidence() -> dict[str, Any]:
    """Summarize the complete nested source decomposition for the q-projection region."""
    root = read("results/coverage/cases/l23_qproj_attention_state_region.json")
    go = read("results/final/l23_go_path_summary.json")
    residual = {
        layer: read(f"results/final/l23_residual_l{layer}_summary.json")
        for layer in (24, 25, 26, 27)
    }
    terminal = read("results/final/l23_terminal_summary.json")
    final_norm = read("results/final/l23_final_norm_summary.json")
    key = read("results/final/l23_key_actual_scale_materialization_32states.json")
    kernel_probe_path = ROOT / "results/property/root_cause_closure_v1/l23_softmax_kernel_output_probe_v1.json"
    kernel_probe = json.loads(kernel_probe_path.read_text()) if kernel_probe_path.is_file() else None
    input_probe_path = ROOT / "results/property/root_cause_closure_v1/l23_softmax_kernel_input_probe_v1.json"
    input_probe = json.loads(input_probe_path.read_text()) if input_probe_path.is_file() else None
    scores_probe_path = ROOT / "results/property/root_cause_closure_v1/l23_softmax_scores_probe_v1.json"
    scores_probe = json.loads(scores_probe_path.read_text()) if scores_probe_path.is_file() else None
    statistics_probe_path = ROOT / "results/property/root_cause_closure_v1/l23_softmax_statistics_probe_v1.json"
    statistics_probe = json.loads(statistics_probe_path.read_text()) if statistics_probe_path.is_file() else None
    carrier_probe_path = ROOT / "results/property/root_cause_closure_v1/l23_softmax_carrier_right_probe_v1.json"
    carrier_probe = json.loads(carrier_probe_path.read_text()) if carrier_probe_path.is_file() else None
    if root.get("status") != "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE":
        raise ValueError("attention semantic-region certificate is not complete")
    if not root["gates"]["s_bwd_only_repair_closes_direction"]:
        raise ValueError("S_bwd repair does not close the declared carrier")
    if not root["gates"]["matched_sham_exact"]:
        raise ValueError("attention semantic-region sham is not exact")
    return {
        "status": "COMPLETE_DECLARED_SEMANTIC_REGION_DECOMPOSITION",
        "root_boundary": (
            "generated softmax-backward output feeding bmm_75; the retained semantic-region "
            "record names the same S_bwd carrier at its historical bmm boundary"
            if kernel_probe is not None else root["causal_attribution"]["root_boundary"]
        ),
        "single_kernel_attribution": root["classification"]["single_kernel_property_eligible"],
        "direct_softmax_kernel_output_probe": (
            {
                "state_count": len(kernel_probe.get("rows", [])),
                "output_is_bmm_left_input": all(
                    row["candidate_kernel_output_equals_bmm_input"]
                    for row in kernel_probe.get("rows", [])
                ),
                "output_buffer_is_bmm_left_input": all(
                    row["candidate_kernel_output_ptr_equals_bmm_input"]
                    for row in kernel_probe.get("rows", [])
                ),
                "interpretation": (
                    "The generated softmax-backward output is the actual bmm_75 left-input "
                    "buffer on every probed state. Replacing that output with the eager "
                    "reference changes the q_proj gradient, so this component has a direct "
                    "path-preserving causal intervention; it does not claim that it explains "
                    "the entire multi-source attention region."
                ),
                "qproj_gradient_changed_all_states": all(
                    row["full_qproj_grad_repair_l2"] > 0.0
                    for row in kernel_probe.get("rows", [])
                ),
                "qproj_gradient_repair_l2": [
                    row["full_qproj_grad_repair_l2"] for row in kernel_probe.get("rows", [])
                ],
            }
            if kernel_probe is not None else None
        ),
        "direct_softmax_kernel_input_probe": (
            {
                "state_count": len(input_probe.get("rows", [])),
                "input_replacement_observed": all(
                    row["target_input_replaced"]
                    for row in input_probe.get("rows", [])
                ),
                "qproj_gradient_changed_all_states": all(
                    row["full_qproj_grad_repair_l2"] > 0.0
                    for row in input_probe.get("rows", [])
                ),
                "qproj_gradient_repair_l2": [
                    row["full_qproj_grad_repair_l2"] for row in input_probe.get("rows", [])
                ],
                "candidate_input_vs_eager_max_abs": [
                    row["candidate_softmax_input_vs_eager_u_max_abs"]
                    for row in input_probe.get("rows", [])
                ],
                "interpretation": (
                    "Replacing only the upstream gradient input of the generated "
                    "softmax-backward call changes the q_proj gradient on every "
                    "probed state. This establishes a second path-preserving "
                    "component of the attention-region effect, but does not identify "
                    "which earlier probability or key producer created that input."
                ),
            }
            if input_probe is not None else None
        ),
        "direct_softmax_scores_probe": (
            {
                "state_count": len(scores_probe.get("rows", [])),
                "scores_replacement_observed": all(
                    row["target_scores_replaced"]
                    for row in scores_probe.get("rows", [])
                ),
                "qproj_gradient_changed_all_states": all(
                    row["full_qproj_grad_repair_l2"] > 0.0
                    for row in scores_probe.get("rows", [])
                ),
                "qproj_gradient_repair_l2": [
                    row["full_qproj_grad_repair_l2"] for row in scores_probe.get("rows", [])
                ],
                "candidate_scores_error_max_abs": [
                    row["candidate_scores_vs_eager_max_abs"]
                    for row in scores_probe.get("rows", [])
                ],
                "intervention_operand_l2": [
                    row["intervention_operand_repair_l2"] for row in scores_probe.get("rows", [])
                ],
                "interpretation": (
                    "Replacing only the pre-softmax score buffer consumed by the target "
                    "softmax-backward call changes q_proj gradient on every probed "
                    "state. This identifies the saved score operand as a "
                    "causal component, without assigning its upstream producer."
                ),
            }
            if scores_probe is not None else None
        ),
        "direct_softmax_statistics_probe": (
            {
                "state_count": len(statistics_probe.get("rows", [])),
                "statistics_replacement_observed": all(
                    row["target_statistics_replaced"]
                    for row in statistics_probe.get("rows", [])
                ),
                "qproj_gradient_changed_all_states": all(
                    row["full_qproj_grad_repair_l2"] > 0.0
                    for row in statistics_probe.get("rows", [])
                ),
                "qproj_gradient_repair_l2": [
                    row["full_qproj_grad_repair_l2"] for row in statistics_probe.get("rows", [])
                ],
                "intervention_operand_l2": [
                    row["intervention_operand_repair_l2"] for row in statistics_probe.get("rows", [])
                ],
                "interpretation": (
                    "Replacing only the max and normalizer statistics consumed by "
                    "the target softmax-backward call changes q_proj gradient on "
                    "every probed state. This identifies the saved statistics as "
                    "a causal component, without assigning their producer."
                ),
            }
            if statistics_probe is not None else None
        ),
        "direct_bmm_key_carrier_probe": (
            {
                "state_count": len(carrier_probe.get("rows", [])),
                "carrier_replacement_observed": all(
                    row["target_carrier_right_replaced"]
                    for row in carrier_probe.get("rows", [])
                ),
                "qproj_gradient_changed_all_states": all(
                    row["full_qproj_grad_repair_l2"] > 0.0
                    for row in carrier_probe.get("rows", [])
                ),
                "qproj_gradient_repair_l2": [
                    row["full_qproj_grad_repair_l2"] for row in carrier_probe.get("rows", [])
                ],
                "intervention_operand_l2": [
                    row["intervention_operand_repair_l2"] for row in carrier_probe.get("rows", [])
                ],
                "interpretation": (
                    "Replacing only the right key carrier of bmm_75 changes q_proj "
                    "gradient on every probed state. This confirms a direct key-side "
                    "component after softmax-backward, while the upstream key producer "
                    "and its interaction with score formation remain separate questions."
                ),
            }
            if carrier_probe is not None else None
        ),
        "carrier": {
            "positive_states": root["causal_attribution"]["total_projection"]["positive_states"],
            "bootstrap_95": root["causal_attribution"]["total_projection"]["state_cluster_bootstrap_95"],
            "projection_strictly_increases_each_step": root["trajectory"]["projection_strictly_increases_each_step"],
        },
        "nested_source_cuts": {
            "downstream_go_path_fraction": go["ratios"]["go_path_removal_over_original_total"],
            "direct_residual_fraction": go["ratios"]["r_shapley_over_original_total"],
            "local_mlp_fraction_by_layer": {
                str(layer): residual[layer]["ratios"]["a_shapley_over_original_total"]
                for layer in residual
            },
            "attention_vjp_fraction_by_layer": {
                str(layer): residual[layer]["ratios"]["b_shapley_over_original_total"]
                for layer in residual
            },
            "terminal_upstream_logit_fraction": terminal["ratios_over_original_total"].get("upstream_logits"),
            "terminal_nll_vjp_fraction": terminal["ratios_over_original_total"].get("logits_vjp"),
            "terminal_lm_head_vjp_fraction": terminal["ratios_over_original_total"].get("lm_head_mm"),
            "terminal_final_rms_fraction": terminal["ratios_over_original_total"].get("final_norm"),
            "key_materialization_record": {
                "status": key.get("status"),
                "intervention_endpoint": key.get("local_vjp_intervention", {}).get("endpoint"),
                "key_reference_variant": key.get("local_vjp_intervention", {}).get("key_reference_variant"),
                "row_count": len(key.get("rows", [])),
            },
            "final_norm_fraction": final_norm["ratios_over_original_total"].get("joint"),
        },
        "interpretation": (
            "The declared semantic region has a closed causal boundary and complete retained "
            "nested source cuts. Four path-preserving interventions now separately show that "
            "the generated softmax-backward output, its upstream gradient input, the consumed "
            "probability buffer, the consumed max/normalizer statistics and the right key "
            "carrier of bmm_75 affect the q_proj gradient. Percentages are nested, not "
            "additive; their upstream query/key and residual-stream producers are not "
            "uniquely separated."
        ),
    }


def source_record_inventory() -> dict[str, Any]:
    """Account for all retained causal-audit records without upgrading them."""
    data = read("results/property/case_causal_audit_v1/exhaustive_source_records.json")
    rows = data["rows"]
    source_kind_counts = Counter(row["source_kind"] for row in rows)
    family_counts = Counter(row.get("operator_family") or "UNSPECIFIED" for row in rows)
    assessment_counts = Counter(row["assessment"] for row in rows)
    return {
        "record_count": len(rows),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "operator_family_counts": dict(sorted(family_counts.items())),
        "assessment_counts": dict(sorted(assessment_counts.items())),
        "interpretation": (
            "All retained source records are accounted for. A measured fixed-suite difference "
            "does not by itself establish a mean bias or a numerical source; only records with "
            "an independent same-input source check are promoted into the deduplicated ledger."
        ),
    }


def generalization_benchmark_frontier() -> dict[str, Any]:
    """List every frozen benchmark case without promoting measurement to a root."""
    data = read("results/property/generalization_benchmark_v1/summary.json")
    cases = []
    for case_id, row in sorted(data["cases"].items()):
        cases.append({
            "case_id": case_id,
            "model": row.get("model"),
            "family": row.get("family"),
            "primary_update_result": row.get("primary_update_result"),
            "confirmed_update_branches": row.get("confirmed_update_branches"),
            "root_cause_status": "MEASUREMENT_ONLY_NO_NEW_SOURCE_INTERVENTION",
            "missing_observation": (
                "same-input source intervention for this exact endpoint, followed by an independent "
                "state-bank confirmation; the benchmark result alone does not identify a root"
            ),
        })
    return {
        "case_count": len(cases),
        "selection_status": data.get("status"),
        "claim_boundary": data.get("claim_boundary"),
        "cases": cases,
        "interpretation": (
            "The frozen benchmark establishes measured update behavior at selected training locations. "
            "Its AOT endpoint substitutions do not by themselves isolate a low-level arithmetic source, "
            "so no benchmark case is promoted to a new root-cause closure without a case-specific source intervention."
        ),
    }


def operator_family_frontier() -> dict[str, Any]:
    """Summarize all catalog families and their current causal evidence level."""
    # The family-frontier artifact is the current deduplicated inventory.  The
    # older operator-family report predates the DATA_MOVEMENT_LAYOUT and
    # ELEMENTWISE campaigns, so using it here silently dropped two measured
    # families from the consolidated ledger.
    data = read("results/property/numerical_coverage_v1/unmeasured_triton_family_frontier_v4.json")
    legacy = read("results/property/numerical_coverage_v1/operator_family_report_v10.json")
    legacy_by_id = {row["family_id"]: row for row in legacy["families"]}
    labels = {
        "LINEAR": "矩阵乘法与线性层",
        "NORMALIZATION": "Normalization",
        "SOFTMAX": "Softmax / log-softmax",
        "CROSS_ENTROPY": "Cross-entropy / NLL loss",
        "SILU_GATING": "SiLU 与门控乘法",
        "GELU": "GELU",
        "SOFTPLUS": "Softplus",
        "RECURRENCE": "状态递推 / scan",
        "ROTARY": "Rotary / position scaling",
        "REDUCTION": "求和、均值与其他归约",
        "CONVOLUTION": "卷积",
        "EMBEDDING": "Embedding lookup / backward",
        "INDEXED_ACCUMULATION": "索引、scatter 与梯度累加",
        "MASK_POSITION_CONTROL": "位置、mask 与前缀控制",
        "DATA_MOVEMENT_LAYOUT": "view、copy、transpose 与布局变换",
        "ELEMENTWISE": "逐元素算术与类型转换",
        "FUSED_MIXED": "跨多个计算家族的融合 kernel",
        "FUSED_ATTENTION": "融合 attention",
        "OPTIMIZER_UPDATE": "Optimizer 参数与 moment 更新",
        "SELECTION": "Top-k / sort selection",
        "ELEMENTWISE_BIAS": "按通道偏置加法",
    }
    statuses = {
        "LINEAR": ("CASE_SPECIFIC_SOURCE_EVIDENCE_ONLY", "MM/GEMM sources are isolated only in named cases; no universal linear root"),
        "NORMALIZATION": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "RMSNorm cast/materialization is now a named natural training group across three models; the broader normalization family still contains unresolved and control-only paths"),
        "SOFTMAX": ("LOCAL_ROOT_IN_ONE_SAVED_STATE_CASE", "saved-state inconsistency is closed for one declared endpoint; natural family bias remains open"),
        "CROSS_ENTROPY": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "Gemma-4 native cross-entropy versus explicit FP32 log-softmax/gather is a named natural NLL boundary; other loss endpoints remain mixed or unresolved"),
        "SILU_GATING": ("SOURCE_CLOSED_JOINT_EXPRESSION_BOUNDARY", "a path-preserving 32-state intervention replacing the product and derivative-factor FP32 intermediates jointly reproduces the native/reference write effect; natural population mean bias remains a separate claim"),
        "SOFTPLUS": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "Mamba delta-softplus is a named natural source-boundary group; generic and fused recurrence variants remain separate or unresolved"),
        "RECURRENCE": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "Mamba component boundaries and the new RWKV time-decay boundary are closed natural groups; the full fused-recurrence regions remain unresolved"),
        "ROTARY": ("NATURAL_SOURCE_FACTORIAL_ROOT_CLOSED_MEAN_BIAS_SCOPE_OPEN", "a real vision-RoPE source factorial rules out trig precision alone and reproduces the effect by FP32 multiply/add followed by BF16 write-back"),
        "REDUCTION": ("MULTIPLE_CASE_SPECIFIC_CONTROLS", "Liger/expert/RMS results have different boundaries and cannot be merged"),
        "INDEXED_ACCUMULATION": ("CONDITIONAL_OPERATOR_BIAS_CONFIRMED", "repeated-destination finite-precision reduction-order probe is closed; no natural training boundary or training consequence yet"),
        "GELU": ("SOURCE_CLOSED_JOINT_INTERMEDIATE_BOUNDARY", "a path-preserving 32-state intervention jointly replacing derivative and product-factor intermediates reproduces the native/reference write effect; natural population mean bias remains a separate claim"),
        "CONVOLUTION": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "Gemma-3 patch convolution accumulation is a named natural training group; other convolution endpoints remain unisolated"),
        "EMBEDDING": ("NATURAL_SCOPED_ROOT_PLUS_OPEN_FAMILY", "DeepSeek/Qwen3 repeated-token embedding backward accumulation is a named natural group; other embedding endpoints remain screened or unresolved"),
        "SELECTION": ("NEGATIVE_CONTROL_ONLY", "equal-score tie-order control was identity on the declared suite"),
        "OPTIMIZER_UPDATE": ("END_TO_END_IN_ADAMW8BIT_CASE", "moment residual propagation and targeted compensation are closed only for the declared optimizer case"),
        "FUSED_ATTENTION": ("NATURAL_SCOPED_BACKEND_BOUNDARY_ROOT_CLOSED", "Qwen3 eager versus SDPA is a closed semantic backend-boundary root on the declared target layer; lower-level arithmetic is optional refinement, not a second bias group"),
        "ELEMENTWISE_BIAS": ("EXTERNAL_EVIDENCE_ONLY", "external evidence exists, but there is no current catalog position or unified measurement result"),
        "DATA_MOVEMENT_LAYOUT": ("MEASUREMENT_ONLY_FIXED_SUITE", "a fixed-suite endpoint measurement exists; no same-input source intervention isolates a numerical root"),
        "ELEMENTWISE": ("MEASUREMENT_ONLY_FIXED_SUITE", "a fixed-suite endpoint measurement exists; no same-input source intervention isolates a numerical root"),
        "FUSED_MIXED": ("MEASUREMENT_BLOCKED_EXECUTION_PATH", "the family has a recorded execution-path blockage; no valid family measurement is promoted"),
        "MASK_POSITION_CONTROL": ("SCREENED_IDENTITY", "implicit versus explicit causal-mask representation was exact identity on the tested Qwen3 boundary; no natural mask group is promoted"),
    }
    families = []
    for frontier_row in data["families"]:
        family_id = frontier_row["operator_family"]
        old_row = legacy_by_id.get(family_id, {})
        status, interpretation = statuses.get(
            family_id,
            ("MEASUREMENT_ONLY", "no retained source-isolating intervention for this family"),
        )
        catalogue = frontier_row.get("catalogue") or {}
        families.append({
            "family_id": family_id,
            "label": labels.get(family_id, old_row.get("label")),
            "classified_positions": catalogue.get("catalogue_position_count", 0) or 0,
            "support_stage_counts": catalogue.get("support_status_counts", {}),
            "recorded_runtime_status_counts": old_row.get("recorded_runtime_status_counts", {}),
            "historical_role_records": old_row.get("historical_role_records", 0),
            "frontier_status": frontier_row.get("status"),
            "next_action": frontier_row.get("next_action"),
            "root_cause_status": status,
            "interpretation": interpretation,
        })
    return {
        "family_count": len(families),
        "catalogue_position_count": sum(row["classified_positions"] for row in families),
        "families": families,
        "interpretation": (
            "The current family frontier is the authoritative deduplicated inventory. Family coverage is not a count of independent roots. A family is promoted only when a "
            "case-specific intervention identifies a numerical source; otherwise it remains measurement-only "
            "or a negative/partial control. Blocked and external-only families remain explicit rather than being treated as negative results."
        ),
    }


def root_cause_frontier(rows: list[dict[str, Any]], families: dict[str, Any]) -> dict[str, Any]:
    """Return the compact, human-facing frontier for unfinished attribution."""
    open_groups = []
    for row in rows:
        blocker = OPEN_ROOT_CAUSE_BLOCKERS.get(row["problem_group"])
        # A blocker describes the fallback frontier for an unresolved row.  A
        # later source intervention may close that row while leaving its
        # historical blocker text useful in the source file; do not expose a
        # closed row as open in the generated ledger.
        if blocker is None or "CLOSED" in row["closure"]:
            continue
        open_groups.append({
            "problem_group": row["problem_group"],
            "closure": row["closure"],
            **blocker,
        })
    measurement_only = [
        {
            "family_id": row["family_id"],
            "classified_positions": row["classified_positions"],
            "root_cause_status": row["root_cause_status"],
            "why_offline_stops": (
                "当前只有目录/端点测量，没有该家族的同输入 reference 与单变量来源干预；"
                "从位置数量或 update RMS 不能推导数学根因"
            ),
        }
        for row in families["families"]
        if row["root_cause_status"].startswith("MEASUREMENT_ONLY")
    ]
    blocked = [
        {
            "family_id": row["family_id"],
            "classified_positions": row["classified_positions"],
            "root_cause_status": row["root_cause_status"],
            "why_offline_stops": (
                "当前没有可执行的完整参考/运行绑定，不能把失败、超时或缺少 reference 当作 bias 或阴性结果"
            ),
        }
        for row in families["families"]
        if row["root_cause_status"].startswith("MEASUREMENT_BLOCKED")
    ]
    external_only = [
        {
            "family_id": row["family_id"],
            "classified_positions": row["classified_positions"],
            "root_cause_status": row["root_cause_status"],
            "why_offline_stops": "只有外部或历史证据，没有当前目录位置和统一测量入口",
        }
        for row in families["families"]
        if row["root_cause_status"].startswith("EXTERNAL_EVIDENCE")
    ]
    return {
        "open_problem_groups": open_groups,
        "measurement_only_operator_families": measurement_only,
        "blocked_operator_families": blocked,
        "external_only_operator_families": external_only,
        "interpretation": (
            "The open entries are not missing bookkeeping: the retained observables are "
            "insufficient to distinguish the listed competing sources. A new intervention "
            "must hold the other paths fixed; otherwise the stronger root-cause claim is "
            "not identifiable from the current artifacts."
        ),
    }


def main() -> None:
    prior = read("results/property/case_causal_audit_v1/scientific_case_closure.json")
    rows = []
    coverage_collections = []
    for row in prior["rows"]:
        if row["problem_group"] in {"common_input_silu_and_rms_backward_families", "reference_graph_regions"}:
            coverage_collections.append({
                "collection": row["problem_group"],
                "status": row["closure"],
                "scope": row["implementation_boundary"],
                "interpretation": row["remaining_limit"],
                "evidence": row["evidence"],
            })
            continue
        current = {
            "problem_group": row["problem_group"],
            "closure": row["closure"],
            "numerical_source": row["numerical_source"],
            "bias_formation": row["bias_formation"],
            "training_outcome": row["training_outcome"],
            "what_is_proven": row["remaining_limit"],
            "next_needed_observation": row["next_root_cause_test"],
            "evidence": row["evidence"],
        }
        if row["problem_group"] == "silu_backward_evaluation":
            natural_path = ROOT / "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json"
            natural_probe = json.loads(natural_path.read_text()) if natural_path.is_file() else None
            mediation_path = ROOT / "results/property/root_cause_closure_v1/silu_single_source_mediation_v7_32_20260921.json"
            mediation = json.loads(mediation_path.read_text()) if mediation_path.is_file() else None
            combined = (mediation or {}).get("mediation_summary", {}).get("reference_product_factor", {})
            joint_source_closed = (
                (mediation or {}).get("status") == "COMPLETE_PATH_PRESERVING_SILU_MEDIATION_INTERVENTIONS_V7"
                and combined.get("exact_write_count") == (mediation or {}).get("state_count")
                and float(combined.get("residual_norm_ratio_aggregate", 1.0)) <= 1e-9
            )
            current["closure"] = (
                "SOURCE_CLOSED_JOINT_PRODUCT_FACTOR_MEAN_BIAS_OPEN"
                if joint_source_closed
                else "SOURCE_BOUNDARY_NARROWED_RECIPROCAL_DIVISION_MEAN_BIAS_OPEN"
            )
            current["numerical_source"] = (
                "finite-precision candidate/reference response at a generated Triton gate-gradient endpoint; "
                "the generated derivative path materializes a product and a derivative-factor separately; "
                "jointly replacing those two FP32 intermediates with same-input reference values reproduces "
                "the native/reference write effect on the retained 32-state bank"
            )
            current["bias_formation"] = (
                "the same natural state bank gives the same candidate-minus-reference effect "
                "energy under native and explicit sigmoid source references; natural-operand probing finds "
                "the first local difference at reciprocal/division, while path-preserving joint product/factor "
                "replacement exactly mediates the downstream write profile; the declared empirical-bank "
                "aligned mean is now supported by a 128-draw probe, while unrestricted natural mean "
                "generalization remains open"
            )
            current["what_is_proven"] = (
                "one AST-checked gate-gradient endpoint has a reproducible candidate/reference profile on "
                "a 32-state natural bank. The first output-side division, output multiplication, FMA choices "
                "and saved-forward FP32 recomputation are not sufficient explanations; derivative-side "
                "sigmoid replacement alone is also incomplete. Replacing the generated derivative product "
                "and derivative-factor intermediates jointly with same-input FP32 reference values gives "
                "exact parameter-write agreement in all 32 states (aggregate effect ratio 1.0, cosine "
                "approximately 1.0, residual 0). This closes the reviewed source at the joint product/factor "
                "materialization boundary. The individual components are not interchangeable: factor-only "
                "replacement covers about 0.974 and product-only about 0.490 on the 32-state bank. "
                "A later 128-draw with-replacement probe supports a negative aligned write mean "
                "for the declared DeepSeek trajectory bank; an unrestricted natural-population "
                "claim and any finer instruction-level decomposition remain separate questions"
            )
            current["next_needed_observation"] = (
                "the unrestricted natural-population mean-bias claim still needs a predeclared bank "
                "representing that population (the declared empirical-bank probe is already complete); "
                "instruction-level decomposition below the jointly sufficient product/factor boundary is "
                "optional refinement, not a prerequisite for this source-closure statement"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/numerical_coverage_v1/silu_factorial_source_manifest_v2.json",
                "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/mapped_backward_667_in_out_ptr0-silu-common-input.json",
                "results/property/numerical_coverage_v1/deepseek128_silu_batch000/raw/mapped_backward_667_in_out_ptr0-silu-common-input.json",
                "results/property/root_cause_closure_v1/silu_source_factorial_v1.json",
                "results/property/root_cause_closure_v1/silu_intermediate_probe_v1.json",
                "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json",
                "scripts/analyze_silu_source_factorial.py",
                "scripts/probe_silu_intermediates.py",
                "scripts/capture_silu_natural_intermediates.py",
                "results/property/root_cause_closure_v1/silu_single_source_intervention_v2_4_20260920.json",
                "results/property/root_cause_closure_v1/silu_all_divisions_intervention_v3_4_20260920.json",
                "results/property/root_cause_closure_v1/silu_single_source_intervention_v4_16_20260920.json",
                "results/property/root_cause_closure_v1/silu_single_source_mediation_v6_16_20260921.json",
                "results/property/root_cause_closure_v1/silu_single_source_mediation_v7_16_20260921.json",
                "results/property/root_cause_closure_v1/silu_single_source_mediation_v7_32_20260921.json",
                "results/property/root_cause_closure_v1/silu_mean_probe_empirical_bank_128_20260921.json",
                "scripts/run_silu_single_source_intervention.py",
            ]
            current["derived"] = silu_factorial_evidence()
        if row["problem_group"] == "mm_gemm_output_and_accumulation":
            mm_derived = mm_source_evidence()
            current["numerical_source"] = (
                "case-specific finite-precision sources: output rounding only in Qwen128, "
                "kernel plus output rounding in Qwen64/Mamba, and kernel arithmetic in Phi"
            )
            current["bias_formation"] = (
                "same-operands decompositions isolate the listed source components in each "
                "case; conditional source-debiased ensembles then show a reproducible "
                "candidate-minus-repair downstream effect for the named Qwen cases and "
                "the resolved Mamba local/zero-moment branches"
            )
            current["closure"] = "CASE_SPECIFIC_SOURCE_AND_CONDITIONAL_F_B_EFFECT_CLOSED_NATURAL_GENERALIZATION_OPEN"
            current["what_is_proven"] = (
                "four concrete MM cases have independent same-operands source decompositions; "
                "retained conditional-debias ensembles show centered repair residuals and "
                "candidate local/zero-moment downstream bias in the named Qwen cases, with "
                "gradient/SGD branches resolved for Qwen and partially unresolved for Mamba. "
                "A 16-state DeepSeek fused-boundary source probe additionally isolates the "
                "three upstream BF16 MM output materialisations: each one, and all three jointly, "
                "produce a stable negative downstream aligned write effect; this is merged into "
                "the same MM/GEMM root family rather than counted again. "
                "The evidence remains conditional on each case's operands, source-debiased "
                "ensemble and implementation boundary"
            )
            current["next_needed_observation"] = (
                "an absolute high-precision downstream reference or independently sampled "
                "natural-state confirmation is needed for a stronger cross-case MM or "
                "population-bias claim; the current conditional result is already closed "
                "for the named same-input branches"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/conditional_debias/qwen128_vproj.json",
                "results/property/conditional_debias/qwen64_vproj.json",
                "results/property/conditional_debias/mamba_seq64_input_proj.json",
                "results/property/new_problem_group_search_v1/deepseek_fused_mm_source_probe_16_20260920.json",
                "scripts/run_deepseek_fused_mm_source_probe.py",
            ]
            current["derived"] = mm_derived
        if row["problem_group"] == "liger_fused_linear_ce_dw_accumulation":
            write_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_parameter_write_confirmation_v1.json"
            if write_path.exists():
                current["closure"] = "SOURCE_CLOSED_DECLARED_BANK_PROJECTED_MEAN_SUPPORTED_NATURAL_GENERALIZATION_OPEN"
                current["bias_formation"] = (
                    "the FP32 chunk-addition source is isolated, and a new 32-state replay measures "
                    "the candidate-minus-reference after a real torch AdamW parameter write in original "
                    "coordinates; the fixed-bank write-direction diagnostic is positive. A later "
                    "with-replacement bank confirmation tests the signed write projection before "
                    "confirmation is interpreted as a scoped population result"
                )
            population_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_population_mean_v1.json"
            if population_path.exists():
                current["bias_formation"] = (
                    "the FP32 chunk-addition source is isolated and the 32-calibration/64-confirmation "
                    "with-replacement empirical-bank run has a positive signed parameter-write "
                    "projection mean under its declared iid assumptions"
                )
            current["what_is_proven"] = (
                "the local addition-order source is reproduced in disjoint length-64 and length-256 "
                "gradient/update banks, and length-64 now has both 32 fixed-bank and 32-calibration/64-"
                "confirmation original-coordinate actual parameter-write measurements after torch AdamW. "
                "The latter supports a positive signed projection mean only for the declared empirical "
                "bank, carrier and zero-moment one-step protocol; the retained training run does not "
                "establish a material loss consequence"
            )
            current["next_needed_observation"] = (
                "a genuinely held-out or different-model state distribution and a warm-moment or "
                "multi-step comparison are still needed for natural generalization or a training-quality "
                "claim. The controlled label-condition probe did not identify a unique cancellation "
                "trigger"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/liger_fp32_chunk_order_v1/length64_confirmation.json",
                "results/property/liger_fp32_chunk_order_v1/length256_confirmation.json",
                "results/property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json",
                "results/property/liger_fp32_chunk_order_v1/length64_parameter_write_confirmation_v1.json",
                "results/property/liger_fp32_chunk_order_v1/length64_population_mean_v1.json",
                "results/property/liger_fp32_chunk_order_v1/input_conditions_parameter_write_v1.json",
            ]
            current["derived"] = liger_order_confirmation_evidence()
        if row["problem_group"] == "fused_rope_position_scaling":
            natural_path = ROOT / "results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json"
            natural_probe = json.loads(natural_path.read_text()) if natural_path.is_file() else None
            factorial_path = ROOT / "results/property/root_cause_closure_v1/ministral_vision_rope_source_factorial_natural_8_20260920.json"
            factorial_probe = json.loads(factorial_path.read_text()) if factorial_path.is_file() else None
            store_tolerance = (natural_probe or {}).get(
                "final_store_tolerance", {"max_abs": 0.0078125, "relative_l2": 2.0e-5}
            )
            natural_complete = bool(
                natural_probe
                and natural_probe.get("status") == "COMPLETE_NATURAL_INTERMEDIATE_PROBE"
                and all(
                    row0["final_output0_alignment"]["max_abs"] <= store_tolerance["max_abs"]
                    and row0["final_output1_alignment"]["max_abs"] <= store_tolerance["max_abs"]
                    and row0["final_output0_alignment"]["relative_l2"] <= store_tolerance["relative_l2"]
                    and row0["final_output1_alignment"]["relative_l2"] <= store_tolerance["relative_l2"]
                    for row0 in natural_probe["rows"]
                )
            )
            factorial_complete = bool(
                factorial_probe
                and factorial_probe.get("status") == "COMPLETE_SOURCE_FACTORIAL_NATURAL_OPERANDS"
                and factorial_probe.get("summary", {}).get("muladd_fp32", {}).get("aligned_write_sign_counts") == {"positive": 0, "negative": 8}
                and factorial_probe.get("summary", {}).get("trig_fp64", {}).get("write_effect_rms_mean") == 0.0
            )
            current["closure"] = (
                "NATURAL_SOURCE_FACTORIAL_ROOT_CLOSED_MEAN_BIAS_SCOPE_OPEN"
                if factorial_complete else (
                    "SOURCE_BOUNDARY_NARROWED_FINAL_EVAL_MEAN_BIAS_OPEN"
                    if natural_complete else "SOURCE_CLASS_NARROWED_NATURAL_ATTRIBUTION_OPEN"
                )
            )
            current["numerical_source"] = (
                "same-input source-factorial evidence on a real Ministral vision RoPE boundary "
                "shows that native BF16 trigonometric values are not sufficient to produce the "
                "effect (FP64-trig/BF16-rotation is exact identity), whereas FP32 multiply/add "
                "with native trig and one BF16 write-back reproduces the native-to-reference "
                "write direction; the remaining text fused endpoint is treated as the same "
                "multiply/add materialization family"
                if factorial_complete else
                "the saved fused kernel evaluates the declared FP32 rotary/position-scaling "
                "formula and stores the result as BF16; natural-operand replay reproduces the "
                "phase, trigonometric, rotation and position-scale intermediates and both final "
                "candidate stores, narrowing the reviewed source to the final fused evaluation/"
                "materialization boundary"
            )
            current["bias_formation"] = (
                "the source-factorial intervention changes only rotation multiply/add precision "
                "on eight real multimodal states: all eight aligned writes are negative, while "
                "the trig-only intervention is exact identity; this identifies the BF16-versus-"
                "FP32 rotation materialization as a sufficient source for the tested vision "
                "endpoint. It is family evidence rather than a second problem group, and does "
                "not establish a natural population mean for the text path"
                if factorial_complete else
                "same-input implementation difference and optimizer-state dependence are "
                "confirmed; repeated same-operand probes rule out tl_math versus libdevice "
                "as the tested-device explanation and show that order/materialization choices "
                "can differ at larger position scales. A separate Qwen3 standard-RoPE probe "
                "also confirms a fixed natural-bank aligned write effect. A new real Ministral "
                "multimodal-loss replay shows the same direction for the vision 2-D RoPE "
                "intermediate-precision intervention; this is family confirmation rather than "
                "a distinct problem group, while the natural-model contribution and state "
                "components of the original fused text path remain to separate"
            )
            current["what_is_proven"] = (
                "on eight real Ministral image/text states, holding all operands and downstream "
                "loss computation fixed, FP32 multiply/add followed by the original BF16 write "
                "reproduces the native-to-reference aligned write effect, while only increasing "
                "trigonometric precision produces exact zero effect. This closes the declared "
                "rotation multiply/add materialization source for the natural vision endpoint. "
                "It does not establish a population mean, text-path universality, or loss outcome"
                if factorial_complete else
                "the saved implementation contract is audited at source level; the tested "
                "RoPE arithmetic variants use identical tl_math and libdevice outputs on the "
                "same operands, and four natural Ministral states reproduce both candidate BF16 "
                "stores within the declared BF16 write tolerance with the reviewed phase/trig/rotation/scale sequence. This "
                "narrows the reviewed endpoint to the final fused evaluation/materialization boundary, but does not prove "
                "which operation inside that boundary is causal, establish a population mean bias, or separate moments "
                "from the step counter. The Qwen3 standard-RoPE confirmation is retained as "
                "same-family evidence rather than a duplicate problem group; the Ministral vision "
                "2-D RoPE replay likewise confirms a stable aligned write effect under a real "
                "multimodal language-loss endpoint without adding a new root-cause group"
            )
            current["next_needed_observation"] = (
                "none for the declared Ministral vision endpoint; a separate text-path source "
                "factorial and an independently sampled natural-state bank would be needed for "
                "a stronger family-wide or population mean-bias claim"
                if factorial_complete else
                "a path-preserving intervention changing only the final fused expression/materialization, plus an "
                "independently sampled natural-state bank for a population mean-bias claim; a fixed-step-counter "
                "optimizer comparison is still needed to separate the remaining state components"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v2.json",
                "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v3.json",
                "results/property/root_cause_closure_v1/rotary_arithmetic_source_probe_v4.json",
                "results/property/root_cause_closure_v1/rotary_source_contract_v1.json",
                "scripts/analyze_rotary_source_contract.py",
                "scripts/probe_rotary_arithmetic_sources.py",
                "results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json",
                "scripts/capture_rope_natural_intermediates.py",
                "results/property/new_problem_group_search_v1/qwen3_rope_layer13_kproj_16_20260918.json",
                "scripts/run_qwen3_rope_natural_probe.py",
                "results/property/new_problem_group_search_v1/ministral_vision_rope_materialization_natural_cpu8_20260919.json",
                "scripts/run_ministral_vision_rope_natural_probe.py",
            ]
            if factorial_complete:
                current["evidence"].extend([
                    "results/property/root_cause_closure_v1/ministral_vision_rope_source_factorial_natural_8_20260920.json",
                    "scripts/run_ministral_vision_rope_source_factorial.py",
                ])
            current["derived"] = rotary_arithmetic_probe_evidence()
        if row["problem_group"] == "attention_state_to_q_projection_region":
            current["closure"] = "SEMANTIC_REGION_ROOT_CLOSED_MULTI_SOURCE_DECOMPOSITION"
            current["numerical_source"] = (
                "the complete q_proj carrier is rooted at the actual attention-backward "
                "state S_bwd and its upstream gradient input U; nested source cuts separate the downstream cotangent, direct "
                "residual path, layer attenuation paths, and one fusion-delayed BF16 "
                "materialization contributor"
            )
            current["bias_formation"] = (
                "the S_bwd repair closes the directional carrier with an exact matched sham; "
                "the retained layer-23-to-27 path summaries quantify the remaining contributors "
                "and their opposing attenuation terms"
            )
            current["what_is_proven"] = (
                "all retained source branches of the declared semantic region are represented "
                "by exact nested interventions and closure checks; this closes the region-level "
                "root without claiming that one Triton kernel is defective"
            )
            current["next_needed_observation"] = (
                "none for the declared semantic-region root. The direct eight-state probes "
                "now isolate the generated softmax-backward output feeding bmm_75, its "
                "upstream gradient input U, its pre-softmax score input, its max/normalizer "
                "statistics, and the bmm_75 right key carrier; each intervention changes "
                "q_proj gradient. A stronger whole-region single-source claim would still "
                "require interaction-aware interventions for the upstream query/key and "
                "residual-stream producers"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/final/l23_go_path_summary.json",
                "results/final/l23_residual_l24_summary.json",
                "results/final/l23_residual_l25_summary.json",
                "results/final/l23_residual_l26_summary.json",
                "results/final/l23_residual_l27_summary.json",
                "results/final/l23_terminal_summary.json",
                "results/final/l23_final_norm_summary.json",
                "results/final/l23_key_actual_scale_materialization_32states.json",
                "results/coverage/cases/l23_qproj_attention_state_region.json",
                "results/property/root_cause_closure_v1/l23_softmax_kernel_output_probe_v1.json",
                "scripts/probe_l23_softmax_kernel_output.py",
                "results/property/root_cause_closure_v1/l23_softmax_kernel_input_probe_v1.json",
                "results/property/root_cause_closure_v1/l23_softmax_scores_probe_v1.json",
                "results/property/root_cause_closure_v1/l23_softmax_statistics_probe_v1.json",
                "results/property/root_cause_closure_v1/l23_softmax_carrier_right_probe_v1.json",
                "results/property/root_cause_closure_v1/l23_binding_inspection.json",
                "scripts/inspect_l23_qproj_binding.py",
            ]
            current["derived"] = attention_region_source_evidence()
        rows.append(current)
    rows.extend([
        {
            "problem_group": "liger_fused_linear_jsd_distillation",
            "closure": "SOURCE_CLOSED_REAL_TEXT_DISTILLATION_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "Liger fused linear JSD/CE distillation loss evaluated with BF16 chunked operands "
                "versus the same student/teacher hidden states, weights and labels evaluated by an "
                "explicit FP32 JSD plus CE reference"
            ),
            "bias_formation": (
                "the one-variable compiled JSD loss boundary produces a positive aligned scaling "
                "of the actual one-step student decoder write on a real BERT-tiny text bank"
            ),
            "training_outcome": "one-step student decoder parameter-write aligned bias; no multi-step quality endpoint",
            "what_is_proven": (
                "on 32 unique real BERT-tiny text windows, with masked student states and unmasked "
                "teacher states, the compiled Liger JSD path has mean write-effect relative RMS "
                "0.254%; the 16-state confirmation half has 12 positive and 4 negative statewise "
                "aligned values with an approximate 95% interval [+0.0079%, +0.1940%]. This is a "
                "scoped real-text distillation boundary, distinct from fused CE dW accumulation; "
                "it is not a long-run loss or population claim"
            ),
            "next_needed_observation": (
                "an independent teacher/student checkpoint pair and a predeclared multi-step distillation "
                "endpoint are needed before making a broader training-quality claim"
            ),
            "evidence": [
                "results/property/case_causal_audit_v1/liger_jsd_natural_training_boundary.json",
                "scripts/run_liger_jsd_natural_training_probe.py",
            ],
        },
        {
            "problem_group": "bert_fused_addmm_bias_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native BERT BF16 fused addmm versus the same attention dense boundary split into "
                "matrix multiplication plus FP32 bias materialization and BF16 write-back"
            ),
            "bias_formation": (
                "the one-variable fused-addmm versus split-linear boundary produces stable negative "
                "aligned scaling of the actual first-step LayerNorm-weight write on a real BERT-tiny CPU text bank"
            ),
            "training_outcome": "single-step LayerNorm-weight aligned bias on CPU; no CUDA or multi-step quality endpoint",
            "what_is_proven": (
                "on 32 real BERT-tiny text windows, native BF16 fused addmm versus split matmul plus "
                "FP32 bias materialization has mean write RMS 10.158%; the 16-state confirmation half "
                "has aligned-write mean -0.8785% and normal 95% interval [-1.3600%,-0.3970%]. "
                "The split native-mm-versus-FP32-bias control and full-FP32 control are both exact zero, "
                "so the source is the fused-linear implementation path rather than an isolated bias-add "
                "operation. This is a fixed CPU boundary result, not a population, CUDA or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled text bank or second checkpoint and a CUDA-specific confirmation "
                "would be needed for broader implementation claims; a multi-step endpoint would be needed "
                "for a quality claim"
            ),
            "evidence": [
                "results/property/case_causal_audit_v1/bert_fused_addmm_bias_materialization_boundary.json",
                "results/property/new_problem_group_search_v1/bert_linear_bias_materialization_natural_cpu32_20260919.json",
                "results/property/new_problem_group_search_v1/bert_linear_bias_factorial_control_cpu32_20260919.json",
                "results/property/new_problem_group_search_v1/bert_linear_bias_materialization_dense_carrier_cpu32_20260919.json",
                "results/property/case_causal_audit_v1/bert_fused_linear_cuda_boundary_check.json",
                "scripts/run_bert_linear_bias_materialization_natural_probe.py",
            ],
        },
        {
            "problem_group": "bert_attention_softmax_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native BERT CUDA attention softmax evaluated in BF16 versus the same Q/K/V and mask "
                "with softmax evaluated in FP32 and written back once in BF16"
            ),
            "bias_formation": (
                "the one-variable softmax materialization change produces stable negative aligned "
                "scaling of the actual first-step AdamW write on a real BERT-tiny CUDA text bank"
            ),
            "training_outcome": "single-step query-projection aligned bias; loss is exactly unchanged; no multi-step quality endpoint",
            "what_is_proven": (
                "on 32 real BERT-tiny text windows, with Q/K/V, mask, model weights and downstream "
                "path held fixed, native BF16 softmax versus explicit FP32 softmax has mean write RMS "
                "6.30%; the 16-state confirmation half has aligned-write mean -0.1885% and normal "
                "95% interval [-0.2329%,-0.1441%]. The held-out additive direction interval crosses "
                "zero, so this is an aligned softmax-materialization result rather than an additive "
                "vector-mean or loss-quality claim; it is distinct from the saved-P inconsistency group. "
                "A fresh 32-state rerun under the same declared boundary gives write RMS 6.47% and "
                "aligned interval [-0.285%,-0.184%], confirming the sign without adding a new group."
            ),
            "next_needed_observation": (
                "an independent checkpoint or state bank and a multi-step training endpoint if a "
                "population or quality claim is needed; a lower-level CUDA softmax implementation "
                "claim would require kernel identity capture"
            ),
            "evidence": [
                "results/property/case_causal_audit_v1/bert_attention_softmax_materialization_boundary.json",
                "results/property/new_problem_group_search_v1/bert_softmax_materialization_cuda_natural_16_20260920.json",
                "results/property/new_problem_group_search_v1/bert_softmax_materialization_cuda_natural_32_20260920.json",
                "scripts/run_bert_softmax_materialization_cuda_probe.py",
            ],
        },
        {
            "problem_group": "bert_attention_score_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native CUDA FP16 attention-score QK multiplication versus the same Q/K operands "
                "multiplied in FP32 and written back once in FP16"
            ),
            "bias_formation": (
                "the one-variable score-matmul materialization change produces stable negative "
                "aligned scaling of the actual first-step query-projection write"
            ),
            "training_outcome": "single-step query-projection aligned bias; no independent loss endpoint",
            "what_is_proven": (
                "on 32 real BERT-tiny text windows with model weights, inputs, mask, value path and "
                "classifier fixed, native CUDA FP16 QK score multiplication versus explicit FP32 "
                "multiplication has mean write RMS 3.693%; the 16-state confirmation half has 11 "
                "negative and 4 positive aligned values with normal 95% interval "
                "[-0.08434%,-0.03158%]. The held-out additive direction interval crosses zero, "
                "so this is an aligned score-matmul boundary result, distinct from the softmax "
                "materialization group and not a population or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independent checkpoint or state bank and a declared multi-step endpoint for "
                "generalization or quality claims; kernel-level attribution requires runtime identity "
                "capture"
            ),
            "evidence": [
                "results/property/case_causal_audit_v1/bert_attention_score_materialization_boundary.json",
                "results/property/new_problem_group_search_v1/bert_attention_score_materialization_float16_cuda32_20260919.json",
                "scripts/run_bert_attention_score_materialization_natural_probe.py",
            ],
        },
        {
            "problem_group": "bert_attention_value_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native CUDA FP16 attention-probability/value multiplication versus the same "
                "attention probabilities and values multiplied in FP32 and written back once in FP16"
            ),
            "bias_formation": (
                "the one-variable value-contraction materialization change produces stable negative "
                "aligned scaling of the actual first-step query-projection write"
            ),
            "training_outcome": "single-step query-projection aligned bias; no independent loss endpoint",
            "what_is_proven": (
                "on 32 real BERT-tiny text windows with scores, softmax weights, Q/K path, mask, "
                "model weights, value path and classifier held fixed, native CUDA FP16 value "
                "contraction versus explicit FP32 contraction has mean write RMS 3.563%; the "
                "16-state confirmation half has 15 negative and 1 positive aligned values with "
                "normal 95% interval [-0.07301%,-0.03879%]. The held-out additive direction "
                "interval crosses zero, so this is an aligned value-contraction boundary result, "
                "distinct from the score and softmax materialization groups and not a population or "
                "loss-quality claim."
            ),
            "next_needed_observation": (
                "an independent checkpoint or state bank and a declared multi-step endpoint for "
                "generalization or quality claims; kernel-level attribution requires runtime identity "
                "capture"
            ),
            "evidence": [
                "results/property/case_causal_audit_v1/bert_attention_value_materialization_boundary.json",
                "results/property/new_problem_group_search_v1/bert_attention_value_materialization_float16_cuda32_20260919.json",
                "scripts/run_bert_attention_score_materialization_natural_probe.py",
            ],
        },
        {
            "problem_group": "rmsnorm_cast_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "RMSNorm materialization boundary: the native generated-model path normalizes in FP32, "
                "casts the normalized value to the input dtype, then multiplies the BF16 weight; the "
                "source-isolating reference multiplies in FP32 and casts once"
            ),
            "bias_formation": (
                "the same one-variable cast-order change produces a reproducible aligned shrinkage in "
                "the first-step AdamW write on real decoder training graphs, including Qwen3 attention "
                "q_norm and k_norm boundaries"
            ),
            "training_outcome": "three real-model single-step write effects; loss intervals cross zero",
            "what_is_proven": (
                "on 26 natural OLMoE states, 26 natural Qwen3 states, and 16 natural Gemma-4 states, the FP32-weight-multiply "
                "reference changes gradient and parameter-write energy; the write aligned ratio has "
                "95% normal intervals [-0.0887,-0.0667], [-0.0576,-0.0357], and approximately "
                "[-0.2016,-0.0931] respectively. This is a scoped aligned update-bias result, "
                "not a common additive direction or a quality claim. Separate 16-state Qwen3 q_norm "
                "and k_norm probes reproduce the same negative aligned effect (about -2.3% to -5.0%) "
                "and are merged here rather than counted as new root causes"
            ),
            "next_needed_observation": (
                "an independently sampled state population and a longer paired training endpoint if "
                "a population or loss claim is needed; no additional operator group is implied by the "
                "three model confirmations"
            ),
            "evidence": [
                "results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json",
                "results/property/olmoe_rmsnorm_natural_v1/layer0_input_fp32multiply_26_direction.json",
                "results/property/olmoe_rmsnorm_natural_v1/qwen3_layer0_input_fp32multiply_26_direction.json",
                "results/property/new_problem_group_search_v1/gemma4_rmsnorm_natural_16_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3_qnorm_natural_16_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3_knorm_natural_16_20260918.json",
                "scripts/run_gemma4_rmsnorm_natural_probe.py",
                "scripts/run_olmoe_rmsnorm_natural_probe.py",
                "scripts/run_qwen3_qnorm_natural_probe.py",
            ],
            "derived": rmsnorm_cast_materialization_evidence(),
        },
        {
            "problem_group": "deepseek_embedding_backward_accumulation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native repeated-token embedding backward accumulation versus an otherwise identical "
                "FP32 index-add gradient accumulation with the original BF16 parameter write"
            ),
            "bias_formation": (
                "the one-variable backward accumulation change produces a stable negative aligned "
                "scaling of the actual first-step AdamW write on the real DeepSeek/Qwen3 text bank"
            ),
            "training_outcome": "single-step parameter-write aligned bias; forward loss is exactly unchanged",
            "what_is_proven": (
                "on 24 real DeepSeek/Qwen3 text states, native embedding backward versus the same-input "
                "FP32 index-add reference has mean write RMS 1.049% and confirmation aligned-write mean "
                "-0.00685% with a normal 95% interval [-0.00821%,-0.00548%]. The held-out additive "
                "direction interval crosses zero, so this is an aligned scaling result rather than an "
                "additive vector-mean or loss-quality claim"
            ),
            "next_needed_observation": (
                "an independently sampled text state bank or second checkpoint for external confirmation, "
                "and a multi-step training comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_backward_16_20260918.json",
                "scripts/run_deepseek_embedding_backward_natural_probe.py",
            ],
            "derived": deepseek_embedding_backward_accumulation_evidence(),
        },
        {
            "problem_group": "deepseek_embedding_gradient_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the final FP32 embedding gradient retained from one captured real-model "
                "cotangent versus the same repeated-index accumulation materialized to BF16; "
                "the cotangent and index-add arithmetic are held fixed"
            ),
            "bias_formation": (
                "the one-variable gradient-dtype materialization change reaches the actual first-step "
                "embedding write with a stable negative aligned effect, but at a numerically negligible scale"
            ),
            "training_outcome": "single-step embedding-write micro-bias; no loss or quality claim",
            "what_is_proven": (
                "On 16 real DeepSeek/Qwen3 text states, the embedding cotangent is captured once and "
                "reused for native BF16 versus FP32 gradient retention. The mean write-effect RMS is "
                "2.99e-8 and the 8-state confirmation aligned interval is strictly negative "
                "[-2.11e-10,-9.99e-11]. The held-out vector direction interval crosses zero. "
                "This closes a distinct gradient-materialization source boundary, but its measured "
                "magnitude is too small to support a practical training-impact claim."
            ),
            "next_needed_observation": (
                "an independently sampled bank would test reproducibility; a quality endpoint is not "
                "justified unless a materially larger effect is found"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json",
                "scripts/run_deepseek_embedding_gradient_cast_materialization_probe.py",
            ],
            "derived": deepseek_embedding_gradient_materialization_evidence(),
        },
        {
            "problem_group": "deepseek_fused_embedding_nll_partial_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the real generated fused embedding/NLL/normalization backward boundary's partial "
                "term materialization before reduction: native FP32 partials versus a single declared "
                "BF16 partial-rounding variant, with the compiled call ABI and all inputs fixed"
            ),
            "bias_formation": (
                "the one-variable BF16 partial-materialization intervention changes the actual embedding "
                "gradient and first-step write with stable negative aligned scaling; the FP64 reduction "
                "control is nearly identity, separating partial rounding from generic reduction precision"
            ),
            "training_outcome": "single-step embedding-write aligned bias at a fused backward boundary; loss unchanged; no quality claim",
            "what_is_proven": (
                "On 16 real DeepSeek/Qwen3 text states, the generated fused boundary is kept fixed and "
                "only partial terms are rounded to BF16 before reduction. The BF16-partial variant has "
                "mean write RMS 1.704% and an 8-state confirmation aligned interval "
                "[-0.01171%,-0.00557%]. The FP64-reduction control has write RMS 1.52e-6 and an interval "
                "crossing zero. A follow-up same-boundary probe shows that BF16 materialization of the "
                "base term and final value gives 2.609% and 1.683% write RMS, with confirmation aligned "
                "intervals [-0.03821%,-0.03039%] and [-0.01144%,-0.00540%]. We keep these as source "
                "components of the same fused intermediate-materialization group rather than inflating "
                "the problem count."
            ),
            "next_needed_observation": (
                "an independently sampled text bank if external confirmation is needed; a multi-step "
                "quality endpoint would require a declared training comparison"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_8_v3_20260920.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_8_terms_v6_20260920.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_terms_v7_20260920.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_fp32_replay_20260920.json",
                "scripts/run_deepseek_embedding_fused_boundary_probe.py",
            ],
            "derived": deepseek_fused_embedding_nll_partial_materialization_evidence(),
        },
        {
            "problem_group": "deepseek_fused_embedding_nll_reduction_order",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the same generated fused embedding/NLL/normalization backward boundary with "
                "FP32 term arithmetic held fixed and only the reduction element order reversed"
            ),
            "bias_formation": (
                "the FP32 reverse-order intervention changes the actual embedding gradient and "
                "first-step write with a small stable positive aligned effect; the FP64 control "
                "remains near identity, separating order from generic precision"
            ),
            "training_outcome": "single-step embedding-write aligned bias at a fused backward boundary; effect is micro-scale and no quality claim is made",
            "what_is_proven": (
                "On 16 real DeepSeek text states, reversing only the FP32 reduction element order "
                "at the compiled fused boundary gives mean write RMS 1.61e-6 and an 8-state "
                "confirmation aligned interval [5.72e-11,1.96e-9]. The same run's FP64 reduction "
                "control has a near-zero interval crossing zero. This closes a distinct same-precision "
                "non-associative reduction source at the declared boundary, but is too small for a "
                "practical quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled text bank if external confirmation is needed; a multi-step "
                "quality endpoint is not justified by this micro-effect"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_reverse_v5_20260920.json",
                "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json",
                "scripts/run_deepseek_embedding_fused_boundary_probe.py",
            ],
            "derived": deepseek_fused_embedding_nll_reduction_order_evidence(),
        },
        {
            "problem_group": "gemma4_causal_nll_loss_evaluation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native causal cross-entropy loss evaluation versus explicit FP32 shifted "
                "log-softmax/gather on identical Gemma-4 logits; the upstream forward/logit "
                "graph and lm_head carrier are held fixed"
            ),
            "bias_formation": (
                "the one-variable NLL evaluation boundary produces a stable negative aligned "
                "scaling of the first-step AdamW write on the real Gemma-4 E2B text bank"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero",
            "what_is_proven": (
                "on 24 real Gemma-4 E2B text states, native cross-entropy versus explicit FP32 "
                "log-softmax/gather on identical logits has mean write RMS 0.576% and confirmation "
                "aligned-write mean -0.001611% with a normal 95% interval "
                "[-0.001762%,-0.001459%]. The held-out additive direction interval crosses zero, "
                "so this is an aligned scaling result rather than an additive vector-mean or "
                "loss-quality claim"
            ),
            "next_needed_observation": (
                "an independently sampled text bank or second checkpoint for external confirmation, "
                "and a multi-step training comparison if a quality consequence is needed; a lower-level "
                "instruction claim would require a path-preserving fused-cross-entropy intervention"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json",
                "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_16_20260918.json",
                "scripts/run_gemma4_softcapped_nll_natural_probe.py",
            ],
            "derived": gemma4_causal_nll_loss_evaluation_evidence(),
        },
        {
            "problem_group": "bert_nll_loss_evaluation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native BERT-tiny masked-LM cross-entropy evaluation versus explicit FP32 "
                "shifted log-softmax/gather on identical compiled logits; the upstream model "
                "forward and decoder carrier are held fixed"
            ),
            "bias_formation": (
                "the one-variable NLL evaluation boundary produces a stable positive aligned "
                "scaling of the actual first-step AdamW decoder-weight write on a real document bank"
            ),
            "training_outcome": "single-step decoder-weight aligned bias; loss interval crosses zero",
            "what_is_proven": (
                "On 128 real BERT-tiny document windows, native cross entropy versus explicit "
                "FP32 log-softmax/gather on identical compiled logits has mean write RMS "
                "1.469% and a 64-state confirmation aligned-write interval "
                "[+0.00246%,+0.02550%]. The held-out additive direction interval crosses zero, "
                "as does the loss-difference interval, so this is a fixed-suite aligned NLL "
                "evaluation result rather than an additive vector-mean or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled document bank or second checkpoint is needed for a "
                "broader NLL claim; a multi-step quality endpoint would be needed for a training "
                "consequence, and a lower-level kernel claim would require runtime identity capture"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json",
                "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_32_20260918.json",
                "scripts/run_bert_tiny_nll_natural_probe.py",
            ],
            "derived": bert_nll_loss_evaluation_evidence(),
        },
        {
            "problem_group": "gemma4_final_logit_softcap_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Gemma-4 BF16 divide/tanh/multiply at the final-logit soft-cap boundary "
                "versus the same hidden/logit input evaluated in FP32 and written back once in BF16"
            ),
            "bias_formation": (
                "the one-variable final-logit materialization change produces a stable negative "
                "aligned scaling of the actual first-step lm_head AdamW write on the real Gemma-4 text bank"
            ),
            "training_outcome": "single-step lm_head parameter-write aligned bias; multi-step loss consequence not measured",
            "what_is_proven": (
                "on 26 real Gemma-4 E2B text states with the hidden state held fixed, native BF16 "
                "soft-cap arithmetic versus the same-input FP32 reference has mean write RMS 2.22%; "
                "the corresponding logit-gradient endpoint is positively aligned in all 26 states. "
                "The 13-state confirmation half has 12 negative and one positive aligned write values, "
                "with a descriptive normal 95% interval [-0.133%,-0.079%]. This closes the declared "
                "final-logit materialization boundary as a fixed-suite aligned-bias result, distinct "
                "from the downstream causal-NLL evaluation group; it is not a population or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled text bank or second checkpoint, followed by a declared "
                "multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_candidate_20260919.json",
                "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_20260919.json",
                "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json",
                "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_logits_26_20260919.json",
                "scripts/run_gemma4_final_logit_softcap_logits_probe.py",
                "scripts/run_gemma4_final_logit_softcap_parameter_probe.py",
            ],
            "derived": gemma4_final_logit_softcap_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_audio_attention_softcap_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_TOTAL_EFFECT",
            "numerical_source": (
                "native Gemma-4 audio attention logit softcap arithmetic in FP32 versus the "
                "same single attention layer with only the softcap branch evaluated in BF16 "
                "or FP64 before returning to FP32"
            ),
            "bias_formation": (
                "the one-variable softcap precision intervention changes the actual layer-0 "
                "audio-attention q_proj gradient and first-step write on every real speech state; "
                "the fixed-suite total effect is large, while the confirmation aligned direction is heterogeneous"
            ),
            "training_outcome": "single-step audio-attention q_proj total write effect; no population mean or long-run quality claim",
            "what_is_proven": (
                "On sixteen real speech states from the Gemma-4 E2B audio-language graph, native "
                "FP32 softcap arithmetic and the same-input BF16/FP64 softcap variants share all "
                "audio, text, labels and downstream computation. The BF16 variant has mean q_proj "
                "write-effect RMS 193.12% and the FP64 variant 215.30%; the full-suite descriptive "
                "aligned intervals are [-77.36%,-3.70%] and [-92.79%,-3.22%], respectively. "
                "The eight-state confirmation aligned intervals cross zero, so this is a source-closed "
                "fixed-suite total-effect group, not a claimed population directional bias or loss-quality result."
            ),
            "next_needed_observation": (
                "an independently sampled real-speech bank and a predeclared directional endpoint "
                "would be needed for a population mean-bias claim; a multi-step audio-language run "
                "would be needed for a quality consequence"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_attention_softcap_layer0_natural_16_20260920.json",
                "scripts/run_gemma4_audio_attention_softcap_natural_probe.py",
            ],
            "derived": gemma4_audio_attention_softcap_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_audio_attention_softmax_probability_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native FP32 audio-attention softmax probabilities retained in FP32 versus the "
                "same probabilities materialized to BF16 immediately before value contraction; "
                "logits, softcap and downstream path are held fixed"
            ),
            "bias_formation": (
                "the one-variable probability-materialization change reaches the real layer-0 "
                "q_proj gradient and first-step write, with a stable negative aligned scaling; "
                "the same intervention crosses zero in layer 1 and is therefore not claimed as "
                "a family-wide audio-attention property"
            ),
            "training_outcome": "single-step layer-0 q_proj aligned write bias; loss interval exactly zero; no multi-step quality claim",
            "what_is_proven": (
                "On sixteen real Gemma-4 E2B speech states, the native FP32 softmax arithmetic, "
                "logits, softcap, value path and language loss are shared. Only FP32 probability "
                "retention versus BF16 materialization changes. The layer-0 q_proj write-effect RMS "
                "averages 1.77% and the 8-state confirmation aligned interval is strictly negative; "
                "the layer-1 repeat crosses zero, so the closed claim is scoped to the declared "
                "layer-0 boundary and fixed waveform bank."
            ),
            "next_needed_observation": (
                "an independent speech bank or second checkpoint for external confirmation, and a "
                "declared multi-step audio-language endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_natural_16_20260920.json",
                "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_layer1_natural_16_20260920.json",
                "scripts/run_gemma4_audio_attention_softmax_natural_probe.py",
            ],
            "derived": gemma4_audio_attention_softmax_probability_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_audio_output_projection_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Gemma-4 audio-tower output projection in BF16 versus the same audio "
                "features projected in FP32 and written back once in BF16"
            ),
            "bias_formation": (
                "the one-variable audio output-projection materialization change produces large "
                "nonzero gradient and first-step write differences under a full audio-language loss"
            ),
            "training_outcome": "real-audio language-loss difference and single-step projection-write effect; no long-run quality claim",
            "what_is_proven": (
                "on sixteen real LibriSpeech waveforms processed by a local Gemma-4 E2B checkpoint, "
                "the native BF16 and FP32-accumulate projection paths share the audio encoder, text "
                "decoder, labels and checkpoint. The audio projection write-effect RMS averages 88.83%, "
                "and every state has a nonzero loss and projection-gradient difference. The eight-state "
                "confirmation aligned-write interval is entirely negative, while the held-out additive "
                "projection interval crosses zero; this closes a natural fixed-suite audio-language "
                "aligned-bias boundary, not a population mean-bias or long-run loss claim."
            ),
            "next_needed_observation": (
                "an independently sampled real-speech/text bank and a declared multi-step audio-language "
                "training endpoint; a GPU kernel-level claim would additionally require runtime identity capture"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_16_20260919.json",
                "scripts/run_gemma4_audio_language_output_projection_natural.py",
            ],
            "derived": gemma4_audio_output_projection_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_audio_lightconv_glu_product_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype LightConv1d GLU gate-times-value multiplication versus the same "
                "audio features multiplied in FP32 and written back once in the original dtype"
            ),
            "bias_formation": (
                "the one-variable GLU product intervention produces a stable negative aligned "
                "scaling of the actual first-step AdamW write under the full audio-language loss"
            ),
            "training_outcome": "single-step audio LightConv linear_start.weight aligned bias; loss interval crosses zero; no long-run quality claim",
            "what_is_proven": (
                "on sixteen real speech states from the Gemma-4 E2B audio-language graph, the "
                "candidate and FP32-product reference share waveforms, text, labels and the full "
                "downstream path. The mean write-effect RMS is 328.15%, and the eight-state "
                "confirmation aligned-write interval is entirely negative at [-167.57%,-83.65%]. "
                "The held-out additive projection interval crosses zero, so this is a fixed-suite "
                "aligned-scaling result at the GLU product boundary, not a population or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled speech bank or checkpoint and a declared multi-step "
                "audio-language endpoint; a kernel-level claim additionally needs runtime identity capture"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_glu_layer0_natural_8_20260919.json",
                "results/property/new_problem_group_search_v1/gemma4_audio_glu_layer0_natural_16_20260919.json",
                "scripts/run_gemma4_audio_glu_materialization_natural.py",
            ],
            "derived": gemma4_audio_lightconv_glu_product_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_audio_lightconv_depthwise_conv_backward_accumulation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Gemma-4 audio LightConv1d depthwise causal Conv1d backward weight-gradient "
                "accumulation versus the same padded window evaluated with explicit FP32 "
                "multiply-accumulate and original-dtype writeback"
            ),
            "bias_formation": (
                "the one-variable depthwise-convolution boundary produces a positive aligned "
                "scaling of the actual first-step AdamW write under the protocol's ratio-of-sums "
                "estimand on the confirmation speech states"
            ),
            "training_outcome": "single-step depthwise Conv1d weight aligned bias; statewise signs are heterogeneous; no long-run loss claim",
            "what_is_proven": (
                "On sixteen real speech states from the Gemma-4 E2B audio-language graph, the "
                "candidate and explicit FP32-window reference share the model, waveforms, text, "
                "labels and downstream path. The mean write-effect RMS is 31.45%. The eight-state "
                "confirmation ratio-of-sums aligned scaling is +1.975%, and deleting any single "
                "confirmation state leaves the ratio positive. A separate boundary check shows "
                "the explicit window has the same padding semantics and nearly identical forward "
                "outputs, so the observed large effects are primarily in backward weight-gradient "
                "accumulation. The statewise aligned interval crosses zero; this is therefore a "
                "fixed-suite ratio-of-sums aligned-bias result, not a population mean or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled real-speech bank to test whether the positive ratio-of-sums "
                "persists, followed by a declared multi-step audio-language endpoint if a quality "
                "consequence is needed; runtime identity capture is needed for a released-kernel claim"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_depthwise_conv_layer0_natural_16_20260920.json",
                "results/property/new_problem_group_search_v1/gemma4_audio_depthwise_conv_boundary_check_20260920.json",
                "scripts/run_gemma4_audio_depthwise_conv_natural.py",
            ],
            "derived": gemma4_audio_lightconv_depthwise_conv_backward_accumulation_evidence(),
        },
        {
            "problem_group": "gemma4_audio_subsampling_convolution_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Gemma-4 audio subsampling Conv2d layer-1 BF16 accumulation versus the "
                "same captured convolution window evaluated in FP32 and written back in BF16"
            ),
            "bias_formation": (
                "the one-variable Conv2d accumulation/materialization change produces a robust "
                "negative aligned scaling of the actual first-step AdamW write under the "
                "ratio-of-sums estimand on confirmation speech states"
            ),
            "training_outcome": "single-step audio subsampling Conv2d weight aligned bias; no long-run loss claim",
            "what_is_proven": (
                "On sixteen real speech states from the Gemma-4 E2B audio-language graph, native "
                "layer-1 Conv2d and the explicit FP32-accumulate/BF16-write reference share the "
                "model, waveforms, text, labels and downstream path. The mean write-effect RMS is "
                "125.05%. The eight-state confirmation ratio-of-sums aligned scaling is -35.98%, "
                "and deleting any single confirmation state leaves it between -50.53% and -12.72%. "
                "The captured-input boundary check gives native-versus-explicit forward relative RMS "
                "below 2.4e-5 across three states, so the effect is not a padding or input mismatch. "
                "Statewise ratios are heterogeneous; this is a fixed-suite ratio-of-sums aligned-bias "
                "result, not a population or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independently sampled real-speech bank to test persistence, followed by a declared "
                "multi-step audio-language endpoint if a quality consequence is needed; runtime identity "
                "capture is needed for a released-kernel attribution"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_raw_16_20260920.json",
                "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_boundary_check_20260920.json",
                "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer0_natural_16_20260919.json",
                "scripts/run_gemma4_audio_subsample_conv_materialization_natural.py",
                "scripts/run_gemma4_audio_subsample_conv_boundary_check.py",
            ],
            "derived": gemma4_audio_subsampling_convolution_materialization_evidence(),
        },
        {
            "problem_group": "gemma4_rms_row_reduction_order",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the Gemma-4 RMSNorm row square-sum reduction boundary: native FP32 row-wise "
                "square-sum order versus the same-input reversed FP32 order, with the original "
                "normalization, BF16 weight and write-back path otherwise unchanged"
            ),
            "bias_formation": (
                "the one-variable same-input reduction-order change produces a stable negative "
                "aligned scaling of the actual first-step AdamW write on the real Gemma-4 text bank"
            ),
            "training_outcome": "single-step input_layernorm.weight aligned bias; loss interval crosses zero; multi-step loss not measured",
            "what_is_proven": (
                "on 26 real Gemma-4 E2B text states, the native row square-sum reduction and the "
                "reversed FP32 row square-sum reference share the model graph, target and original "
                "write-back. The mean write-effect RMS is 17.40%; all 13 confirmation aligned-write "
                "values are negative and their descriptive normal 95% interval is "
                "[-4.628%,-0.252%]. The recorded loss differences have an interval crossing zero. "
                "This is distinct from the existing RMSNorm cast-before-weight-multiplication group, "
                "but it is a declared model-level reduction-order variant rather than proof of a "
                "released kernel's internal reduction order or a population/loss-quality claim"
            ),
            "next_needed_observation": (
                "runtime source binding or generated-kernel comparison to establish whether a released "
                "implementation uses this order; an independent bank or second checkpoint and a "
                "multi-step quality endpoint are needed for broader or loss claims"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma4_rms_row_reduction_natural_26_20260919.json",
                "scripts/run_gemma4_rms_row_reduction_natural_probe.py",
            ],
            "derived": gemma4_rms_row_reduction_order_evidence(),
        },
        {
            "problem_group": "deberta_disentangled_relative_attention_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native DeBERTa c2p/p2c disentangled relative-position score products in BF16 "
                "versus the same query/key/relative-position path evaluated in FP32 and returned "
                "in the native attention dtype"
            ),
            "bias_formation": (
                "the one-variable relative-attention score materialization change produces a "
                "stable negative aligned scaling of the first-step AdamW write"
            ),
            "training_outcome": "two-layer single-step parameter-write aligned bias; loss intervals cross zero; no quality claim",
            "what_is_proven": (
                "on 32 real text windows from a local DeBERTa-v3-small backbone, both an early "
                "layer and the final encoder layer show entirely negative descriptive confirmation "
                "aligned-write intervals (about -85.7% and -80.2%). The c2p/p2c boundary is distinct "
                "from RoPE, ALiBi and generic BERT attention-score controls. The local checkpoint does "
                "not include a pretrained MLM head, so the loss is a declared auxiliary-head probe and "
                "not a pretrained MLM quality result."
            ),
            "next_needed_observation": (
                "a checkpoint with a trained task head or a backbone-only declared endpoint for a "
                "quality claim; an independently sampled state bank or runtime kernel identity for "
                "population or implementation-level claims"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer0_seed0_32_20260919.json",
                "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer5_32_20260919.json",
                "scripts/run_deberta_disentangled_attention_materialization_natural_probe.py",
            ],
            "derived": deberta_disentangled_relative_attention_materialization_evidence(),
        },
        {
            "problem_group": "bloom_alibi_attention_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Bloom baddbmm QK plus ALiBi score materialization versus the same causal "
                "attention path with explicit FP32 QK multiplication and ALiBi addition before "
                "native-dtype probability evaluation"
            ),
            "bias_formation": (
                "the one-variable ALiBi score materialization change produces a stable negative "
                "aligned scaling of the first-step AdamW write"
            ),
            "training_outcome": "two-layer single-step query-key-value write aligned bias; loss intervals cross zero; no quality claim",
            "what_is_proven": (
                "on 16 real text windows from a local Bloom-560m checkpoint, both layer 0 and layer "
                "11 show entirely negative descriptive confirmation aligned-write intervals (about "
                "-12.1% and -10.8%), with mean write-effect RMS about 45.7% and 42.8%. The same ALiBi "
                "tensor, causal mask and model weights are used on both sides; this is distinct from "
                "the DeBERTa c2p/p2c relative-position family and does not claim a population or loss "
                "quality consequence."
            ),
            "next_needed_observation": (
                "an independently sampled text bank or second checkpoint for a broader claim and a "
                "declared multi-step training endpoint for quality impact"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_16_20260919.json",
                "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_layer11_16_20260919.json",
                "scripts/run_bloom_alibi_attention_materialization_natural_probe.py",
            ],
            "derived": bloom_alibi_attention_materialization_evidence(),
        },
        {
            "problem_group": "qwen3_attention_sdpa_eager_backend",
            "closure": "SOURCE_CLOSED_SEMANTIC_BACKEND_BOUNDARY",
            "numerical_source": (
                "the Qwen3 attention implementation boundary: native eager attention versus the "
                "transformers SDPA interface, including the independently forced Flash-versus-math "
                "SDPA contrast, with the same model, inputs, and attention inputs at the selected layer"
            ),
            "bias_formation": (
                "replacing only the attention backend changes the downstream q_proj gradient and first-step "
                "AdamW write; both the eager-versus-SDPA and forced Flash-versus-math contrasts show a "
                "negative aligned write effect on declared Qwen3 text banks"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero",
            "what_is_proven": (
                "on 16 real Qwen3-1.7B text states at layer 13 q_proj, a path-preserving eager-versus-SDPA "
                "boundary gives mean write RMS about 36.14% and aligned-write 95% interval "
                "[-7.94%,-5.42%]. Separate eight- and sixteen-state forced-math-SDPA replays give the same "
                "qualitative effect (about 37.59% and 35.68% RMS; aligned intervals [-8.52%,-5.77%] and "
                "[-7.25%,-5.61%]), so the result is not explained "
                "solely by selecting a flash backend. The existing 32-state layer-0 forced Flash-versus-math "
                "suite independently gives about 36.91% write RMS and aligned ratio -6.81%. The corrected "
                "same-layer source probe holds Q/K/V inputs exactly fixed and shows that FP32 score contraction, "
                "FP32 value/softmax variants and the full-FP32 eager variant do not reproduce the full SDPA effect "
                "vector. Therefore the causal root is the declared attention-backend boundary as a whole, not any "
                "one tested score/value operation; the lower-level instruction decomposition is an optional refinement. "
                "These are one attention-backend problem family, not separate model-specific groups, and no "
                "loss-quality consequence is claimed"
            ),
            "next_needed_observation": (
                "an independently sampled state bank if a broader population claim is needed; a lower-level "
                "accumulation/fusion intervention can refine the already closed semantic backend root but is "
                "not required for the declared boundary result"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/qwen3_sdpa_vs_eager_layer13_qproj_8_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3_attention_backend_isolated_layer13_qproj_16_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3_attention_backend_math_isolated_layer13_qproj_8_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3_attention_backend_math_isolated_layer13_qproj_16_20260920.json",
                "results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_v1/raw.json",
                "results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_v1/analysis.json",
                "docs/fused_attention_family_audit.md",
                "scripts/run_qwen3_sdpa_vs_eager_natural_probe.py",
                "scripts/run_qwen3_attention_backend_isolated_probe.py",
                "scripts/run_qwen3_attention_backend_math_isolated_probe.py",
                "results/property/root_cause_closure_v1/qwen3_attention_component_source_probe_v3_16_20260920.json",
                "results/property/new_problem_group_search_v1/qwen3_attention_component_source_probe_16_v3_20260920.json",
                "scripts/run_qwen3_attention_component_source_probe.py",
                "results/property/root_cause_closure_v1/qwen3_attention_backend_decomposition_16_20260920.json",
                "scripts/run_qwen3_attention_backend_decomposition.py",
            ],
        },
        {
            "problem_group": "mamba_softplus_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the Mamba sequential recurrence's delta-softplus materialisation: native input-dtype "
                "softplus versus FP32 softplus followed by the original input-dtype cast, with the "
                "same model, tokens, recurrence and parameter write on both sides"
            ),
            "bias_formation": (
                "the one-variable softplus precision change produces a stable negative aligned scaling "
                "of the first-step AdamW write on two real text banks"
            ),
            "training_outcome": "single-step parameter-write aligned bias; independent multi-step loss not measured",
            "what_is_proven": (
                "on a real Mamba-130M checkpoint and layer-0 dt_proj.weight, the 32-state text bank has "
                "mean write RMS 9.50% and confirmation aligned-write mean -0.439% with a normal 95% interval "
                "[-0.470%,-0.407%]; the shorter 16-state bank gives the same sign. "
                "The held-out vector-direction interval crosses zero, so this closes a scoped aligned scaling "
                "source rather than a full additive vector-mean or quality claim"
            ),
            "next_needed_observation": (
                "an independently sampled state bank or second checkpoint for external confirmation, and a "
                "declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_16_20260918.json",
                "scripts/run_mamba_softplus_materialization_natural_probe.py",
            ],
            "derived": mamba_softplus_materialization_evidence(),
        },
        {
            "problem_group": "mamba_discrete_transition_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "Mamba discrete state-transition materialization: native FP32 exp(A*delta) "
                "versus a model-dtype exponent argument followed by exp and FP32 recurrence "
                "value, with the same sequential recurrence and downstream path"
            ),
            "bias_formation": (
                "the one-variable exponent-argument materialization change produces a stable "
                "negative aligned scaling of the first-step AdamW write on real Mamba text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; independent multi-step loss not measured",
            "what_is_proven": (
                "on a real Mamba-130M checkpoint and layer-0 A_log, 16 document-derived text "
                "windows give mean write RMS 26.05% and a confirmation aligned-write mean "
                "-3.817% with a normal 95% interval [-5.050%,-2.584%]. The held-out additive "
                "direction interval crosses zero, so this is a scoped aligned transition-materialization "
                "result rather than an additive population or loss-quality claim. It is distinct "
                "from the already recorded Softplus, causal-convolution, D skip, state-output and z-gate boundaries."
            ),
            "next_needed_observation": (
                "an independently sampled state bank or second checkpoint for external confirmation, "
                "and a declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_discrete_transition_materialization_natural_cpu16_20260919.json",
                "scripts/run_mamba_discrete_transition_materialization_natural_probe.py",
            ],
            "derived": mamba_discrete_transition_materialization_evidence(),
        },
        {
            "problem_group": "rwkv_time_decay_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "RWKV sequential WKV/time-mix recurrence: native model-dtype time_decay "
                "exponential versus the same recurrent boundary with the time_decay exponential "
                "evaluated in FP32 and returned to the model dtype"
            ),
            "bias_formation": (
                "the one-variable time_decay exponential precision change reproduces the all-FP32 "
                "write effect on every declared state, whereas changing only recurrent value "
                "materialization does not reproduce the effect"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero; no multi-step quality claim",
            "what_is_proven": (
                "on 16 real README-text states from RWKV-4-world-169m block 0, the native candidate "
                "versus the same-input decay_fp32 reference has mean write RMS 29.25% and a normal "
                "95% aligned-write interval [-11.34%,-0.43%], with all 16 statewise aligned values "
                "negative. The decay_fp32 and all_fp32 arms are identical in the recorded endpoint, "
                "while value_fp32 alone has a smaller 6.09% mean write RMS and an interval crossing "
                "zero. This closes the time_decay exponential/materialization source for the declared "
                "RWKV checkpoint and text bank; it is distinct from the Mamba recurrence boundaries "
                "and does not establish a population or multi-step loss claim."
            ),
            "next_needed_observation": (
                "an independent RWKV text bank or second checkpoint for external confirmation, and a "
                "declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json",
                "scripts/run_rwkv_time_mix_natural_probe.py",
            ],
            "derived": rwkv_time_decay_materialization_evidence(),
        },
        {
            "problem_group": "rwkv_receptance_sigmoid_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "RWKV attention receptance gate: native model-dtype sigmoid versus the same "
                "receptance computation evaluated in FP32 and returned to the model dtype"
            ),
            "bias_formation": (
                "the one-variable sigmoid materialization change produces a stable negative "
                "aligned scaling of the actual first-step AdamW write while the recurrent WKV "
                "path and all other attention computations stay fixed"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss intervals cross zero; no multi-step quality claim",
            "what_is_proven": (
                "on a real RWKV-4-world-169m checkpoint and block-0 receptance.weight, two "
                "independent 32-state real-text banks each have 16/16 negative held-out "
                "aligned-write values. The confirmation intervals are [-0.1366%,-0.1272%] "
                "and [-0.1298%,-0.1226%], with write-effect RMS means about 7.66% and 7.72%. "
                "This is a scoped sigmoid-materialization aligned update result, not a population "
                "mean, additive vector-bias or loss-quality claim, and it is distinct from the "
                "RWKV time_decay exponential group and the sparse channel-mix negative screen."
            ),
            "next_needed_observation": (
                "an independent RWKV checkpoint or held-out natural state bank for external "
                "confirmation, and a declared multi-step training endpoint if a quality consequence "
                "is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_natural_32_20260920.json",
                "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_method_32_20260920.json",
                "scripts/run_rwkv_receptance_sigmoid_natural_probe.py",
            ],
            "derived": rwkv_receptance_sigmoid_materialization_evidence(),
        },
        {
            "problem_group": "bert_layernorm_compiled_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "compiled native BERT LayerNorm agrees with an explicit FP32 reduction/materialisation "
                "control, while the eager native LayerNorm path differs on the same real checkpoint and inputs"
            ),
            "bias_formation": (
                "the compiled-versus-eager boundary produces a repeatable aligned parameter-write difference "
                "across 32 document-derived token windows under the declared one-step AdamW protocol"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero",
            "what_is_proven": (
                "on one real pretrained BERT-tiny checkpoint and one LayerNorm, the compiled native path "
                "matches the explicit FP32 formula while the eager native path differs; the 32-state eager "
                "comparison has aligned write mean -1.1717% with a normal 95% interval [-1.7647%, -0.5787%]. "
                "This is a scoped aligned update-bias result, not an additive direction, population or loss-quality claim"
            ),
            "next_needed_observation": (
                "a second checkpoint or independently sampled state bank for external confirmation, and a "
                "path-preserving compiler/materialisation intervention if a lower-level instruction cause is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/bert_tiny_layernorm_eager_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/bert_tiny_layernorm_eager_natural_20260918.json",
                "results/property/new_problem_group_search_v1/bert_tiny_layernorm_fp32_reference_20260918.json",
                "scripts/run_bert_tiny_layernorm_natural_probe.py",
            ],
            "derived": bert_layernorm_compiled_materialization_evidence(),
        },
        {
            "problem_group": "bert_embedding_sum_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype BERT word, position and token-type embedding addition versus an "
                "otherwise identical FP32 three-term sum followed by the original native-dtype write"
            ),
            "bias_formation": (
                "the one-variable embedding-sum materialization change produces stable negative "
                "aligned scaling of the actual first-step AdamW write on real BERT-tiny text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real BERT-tiny document-derived states at embedding.weight, native embedding "
                "addition versus the same-input FP32 three-term sum has mean write RMS 6.21% and a "
                "confirmation aligned-write interval [-0.305%,-0.186%]. The held-out additive "
                "direction interval crosses zero, so this is a scoped aligned embedding-sum result, "
                "not an additive population or loss-quality claim. It is distinct from the BERT "
                "LayerNorm compiled/eager boundary and from embedding backward accumulation."
            ),
            "next_needed_observation": (
                "an independent BERT text bank or second checkpoint, and a multi-step training "
                "comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/bert_embedding_sum_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/bert_embedding_sum_materialization_natural_16_20260918.json",
                "scripts/run_bert_embedding_sum_materialization_natural_probe.py",
            ],
            "derived": bert_embedding_sum_materialization_evidence(),
        },
        {
            "problem_group": "bert_pooler_tanh_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ADDITIVE_BIAS",
            "numerical_source": (
                "native model-dtype BERT pooler tanh evaluation versus the same pooler tanh evaluated "
                "in FP32 followed by one original-dtype write"
            ),
            "bias_formation": (
                "the one-variable tanh evaluation/materialization intervention produces a reproducible "
                "held-out additive gradient direction at the embedding carrier on real BERT-tiny text states"
            ),
            "training_outcome": "single-step embedding-gradient directional bias; pooler-dense direction and loss consequence not confirmed",
            "what_is_proven": (
                "on 32 real BERT-tiny document-derived text windows, the FP32-tanh repair has mean gradient "
                "effect RMS 0.811% at bert.embeddings.word_embeddings.weight and a held-out additive-direction "
                "interval [0.00325%, 0.09100%]. The same boundary at bert.pooler.dense.weight has a direction "
                "interval crossing zero. This closes the tanh source boundary for the tested carrier and does "
                "not claim a population mean, all-parameter effect, or loss-quality consequence."
            ),
            "next_needed_observation": (
                "an independently sampled BERT text bank or second checkpoint, plus a path-preserving local-output "
                "readback if the exact tanh instruction/materialization is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/bert_pooler_tanh_natural_cpu32_20260919.json",
                "results/property/new_problem_group_search_v1/bert_pooler_tanh_dense_natural_cpu32_20260919.json",
                "scripts/run_bert_pooler_tanh_natural_probe.py",
            ],
            "derived": bert_pooler_tanh_materialization_evidence(),
        },
        {
            "problem_group": "granite_residual_addition_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Granite attention residual addition, residual + scaled attention output, "
                "versus the same addition evaluated in FP32 and cast once before the unchanged MoE path"
            ),
            "bias_formation": (
                "the one-variable residual-addition materialization change produces stable negative "
                "aligned scaling of the actual first-step AdamW write on real Granite text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real Granite MoE text states at layer-0 self_attn.o_proj.weight, native residual "
                "addition versus the same-input FP32 residual-plus-scaled-output addition has mean write "
                "RMS 39.79% and a confirmation aligned-write interval [-8.884%,-7.220%]. The held-out "
                "additive direction interval crosses zero, so this is a scoped aligned residual-addition "
                "result, not an additive population or loss-quality claim. It is distinct from the Granite "
                "expert-contribution accumulation-order group because the MoE path is unchanged."
            ),
            "next_needed_observation": (
                "an independent Granite text bank or second checkpoint, and a multi-step training "
                "comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/granite_residual_addition_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/granite_residual_addition_materialization_natural_16_20260918.json",
                "scripts/run_granite_residual_addition_materialization_natural_probe.py",
            ],
            "derived": granite_residual_addition_materialization_evidence(),
        },
        {
            "problem_group": "granite_moe_gate_product_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Granite MoE expert activation(gate) times up product versus the same "
                "product evaluated in FP32 and cast once, with routing, expert projections, output "
                "gating and repeated-destination combine unchanged"
            ),
            "bias_formation": (
                "the one-variable gate-times-up product materialization change produces stable negative "
                "aligned scaling and a positive held-out directional projection of the actual first-step "
                "AdamW write on real Granite text states"
            ),
            "training_outcome": "single-step parameter-write aligned and held-out directional bias; loss consequence not measured",
            "what_is_proven": (
                "on 32 real Granite MoE text states at layer-0 block_sparse_moe.input_linear.weight, "
                "native gate-times-up multiplication versus the same-input FP32 product has mean write "
                "RMS 40.41%, confirmation aligned-write interval [-8.429%,-6.822%], and a held-out "
                "calibration-direction projection interval [0.090%,1.700%]. The source is therefore "
                "closed for this declared boundary; it is not a population or loss-quality claim and "
                "is distinct from expert accumulation order and residual addition."
            ),
            "next_needed_observation": (
                "an independently sampled Granite text bank or second checkpoint, and a multi-step "
                "training comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/granite_moe_gate_product_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/granite_moe_gate_product_materialization_natural_16_20260918.json",
                "scripts/run_granite_moe_gate_product_materialization_natural_probe.py",
            ],
            "derived": granite_moe_gate_product_materialization_evidence(),
        },
        {
            "problem_group": "granite_moe_output_gate_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Granite MoE expert-output times routing-gate product versus the same "
                "product evaluated in FP32 and cast once, with routing, expert projections, gate-times-up "
                "product and repeated-destination combine unchanged"
            ),
            "bias_formation": (
                "the one-variable output-gate product materialization change produces stable negative "
                "aligned scaling of the actual first-step AdamW write on real Granite text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real Granite MoE text states at layer-0 block_sparse_moe.input_linear.weight, "
                "native expert-output times routing-gate multiplication versus the same-input FP32 product "
                "has mean write RMS 42.28% and confirmation aligned-write interval [-10.481%,-7.614%]. "
                "The held-out additive direction interval crosses zero, so this is a scoped aligned "
                "output-gate result, not an additive population or loss-quality claim. It is distinct "
                "from the gate-times-up product and expert-contribution accumulation-order groups."
            ),
            "next_needed_observation": (
                "an independently sampled Granite text bank or second checkpoint, and a multi-step "
                "training comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/granite_moe_output_gate_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/granite_moe_output_gate_materialization_natural_16_20260918.json",
                "scripts/run_granite_moe_gate_product_materialization_natural_probe.py",
            ],
            "derived": granite_moe_output_gate_materialization_evidence(),
        },
        {
            "problem_group": "granite_moe_router_score_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native model-dtype Granite router score projection and softmax-weight materialization "
                "versus the same score projection evaluated in FP32, with the native top-k indices frozen"
            ),
            "bias_formation": (
                "the one-variable router-score precision change produces stable negative aligned scaling "
                "of the actual first-step router-weight write while selected experts and expert computation "
                "remain fixed"
            ),
            "training_outcome": "single-step router-write aligned bias; multi-step loss consequence not measured",
            "what_is_proven": (
                "on 32 real Granite text windows, freezing the native top-k indices removes routing-set "
                "changes while the FP32 score/weight reference retains a mean router-write RMS of about "
                "192.68%.  The 16-state confirmation aligned-write mean is -51.72% with a descriptive "
                "95% interval [-67.68%,-35.75%], and 15/16 values are negative.  This is a scoped "
                "router score/weight materialization bias, distinct from the natural top-k selection "
                "sensitivity and the separate expert-contribution accumulation-order group; it is not "
                "an additive population or loss-quality claim. A 128-state OLMoE replay with the "
                "same top-k-freezing intervention has aligned-write interval [-1.503%,-0.279%], "
                "providing cross-model confirmation of this source rather than a second group."
            ),
            "next_needed_observation": (
                "an independently sampled Granite text bank or second checkpoint, and a multi-step "
                "training endpoint if a quality consequence is needed; the frozen-selection intervention "
                "should remain explicit in any broader claim"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_16_20260919.json",
                "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json",
                "results/property/new_problem_group_search_v1/granite_router_projection_materialization_natural_16_20260919.json",
                "results/property/new_problem_group_search_v1/olmoe_router_score_natural_128_20260920.json",
                "results/property/new_problem_group_search_v1/olmoe_router_score_natural_16_20260920.json",
                "results/property/new_problem_group_search_v1/olmoe_router_score_natural_32_20260920.json",
                "results/property/new_problem_group_search_v1/olmoe_router_score_natural_64_20260920.json",
                "scripts/run_granite_router_projection_materialization_natural_probe.py",
                "scripts/run_olmoe_router_score_natural_probe.py",
            ],
            "derived": granite_router_score_materialization_evidence(),
        },
        {
            "problem_group": "gemma3_vision_patch_convolution",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native BF16 patch-convolution accumulation versus an otherwise identical FP32 "
                "accumulation with native-dtype write-back"
            ),
            "bias_formation": (
                "the one-variable accumulation precision change produces a large, consistently "
                "negative aligned scaling of the actual first-step AdamW write on the real Gemma-3 "
                "image/text input bank"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero",
            "what_is_proven": (
                "on 26 real Gemma-3 image/text states at the vision patch convolution, the native "
                "BF16 candidate versus the same-input FP32-accumulation reference has mean write "
                "RMS 106.63% and confirmation aligned-write mean -55.78% with a normal 95% interval "
                "[-63.90%,-47.67%]. The additive held-out direction interval crosses zero, so this "
                "is an aligned scaling result rather than an additive vector-mean claim or a loss-quality claim"
            ),
            "next_needed_observation": (
                "an independently sampled image/text state bank or a second checkpoint for external "
                "confirmation, followed by a training-length comparison if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json",
                "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_16_20260918.json",
                "scripts/run_gemma3_conv_clean_natural_probe.py",
            ],
            "derived": gemma3_vision_patch_convolution_evidence(),
        },
        {
            "problem_group": "qwen3vl_position_interpolation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype four-tap weighted interpolation of the Qwen3-VL learned 2-D "
                "position table versus the same taps summed in FP32 and written back once in "
                "the original dtype; the visual blocks, language model and loss are unchanged"
            ),
            "bias_formation": (
                "the one-variable position-interpolation materialization change produces a "
                "repeatable aligned scaling of the actual first-step AdamW write to the learned "
                "visual position table on real multimodal loss states"
            ),
            "training_outcome": "real image/text loss probe and single-step parameter-write aligned bias; no long-run quality claim",
            "what_is_proven": (
                "on 16 real images from the declared Qwen3-VL reranker bank, native-dtype and "
                "FP32 weighted position interpolation produce a fixed-suite mean write RMS of "
                "0.0423% for visual.pos_embed.weight. The per-state aligned-write ratios are all "
                "negative, with a descriptive normal 95% interval approximately [-0.347%,-0.236%]. "
                "The held-out additive direction is mixed (3 positive, 5 negative), so this is a "
                "scoped aligned position-interpolation result rather than an additive vector-mean, "
                "population or long-run quality claim. The Ministral patch-merger layout control was "
                "screened to exact identity and is not counted here."
            ),
            "next_needed_observation": (
                "an independently sampled multimodal image bank or second checkpoint for external "
                "confirmation, plus a longer paired training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_probe_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_8_20260918.json",
                "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_16_20260918.json",
                "scripts/run_qwen3vl_position_interpolation_probe.py",
                "scripts/run_qwen3vl_position_interpolation_training_probe.py",
            ],
            "derived": qwen3vl_position_interpolation_evidence(),
        },
        {
            "problem_group": "mamba_causal_conv_accumulation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native Mamba causal depthwise convolution evaluation/materialization versus an otherwise "
                "identical FP32 convolution, cast back before the original native activation"
            ),
            "bias_formation": (
                "the one-variable convolution evaluation/materialization change produces a negative held-out direction "
                "and stable negative aligned scaling of the actual first-step AdamW write on real Mamba text states"
            ),
            "training_outcome": "single-step parameter-write aligned and held-out directional bias; loss consequence not measured",
            "what_is_proven": (
                "on 32 real Mamba text states, native causal_conv1d_fn versus the same-input FP32-convolution "
                "reference has mean write RMS 28.56%; the confirmation aligned-write interval is "
                "[-5.28%,-3.30%], and the held-out calibration direction projection interval is "
                "[-0.194%,-0.011%]. The 16-state README bank independently has aligned interval "
                "[-4.07%,-2.85%]. This is a scoped causal-convolution evaluation/materialization result, not a "
                "population or long-run loss claim. It shares the broad accumulation-precision mechanism "
                "with the Gemma patch-convolution case but is a distinct natural operator family."
            ),
            "next_needed_observation": (
                "an independent Mamba text bank or second checkpoint and a declared multi-step training "
                "endpoint if quality impact is needed; backend identity should be recorded before a Triton-specific claim"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_causal_conv_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_causal_conv_materialization_natural_16_20260918.json",
                "scripts/run_mamba_causal_conv_materialization_natural_probe.py",
            ],
            "derived": mamba_causal_conv_accumulation_evidence(),
        },
        {
            "problem_group": "mamba_state_output_contraction_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Mamba recurrent state-to-C contraction versus an otherwise identical "
                "FP32 state/C contraction with native-dtype write-back before the D/z path"
            ),
            "bias_formation": (
                "the one-variable state-output contraction precision change produces stable negative "
                "aligned scaling of the actual first-step AdamW write on real Mamba text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real Mamba text states, native sequential selective scan versus the same-input "
                "FP32 state-to-C contraction reference has mean write RMS 30.64%; the confirmation "
                "aligned-write interval is [-13.40%,-1.40%]. The held-out calibration-direction "
                "interval crosses zero, so this is a scoped aligned state-output contraction result, "
                "not an additive population or long-run loss claim. It is a distinct Mamba operator "
                "boundary from delta-softplus and causal-convolution probes, although all share a broad "
                "precision/materialization mechanism."
            ),
            "next_needed_observation": (
                "an independent Mamba text bank or second checkpoint, backend identity capture, and a "
                "declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_state_output_contraction_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_state_output_contraction_natural_16_20260918.json",
                "scripts/run_mamba_state_output_contraction_natural_probe.py",
            ],
            "derived": mamba_state_output_contraction_materialization_evidence(),
        },
        {
            "problem_group": "mamba_d_skip_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Mamba residual D*hidden_states product versus an otherwise identical "
                "FP32 product with native-dtype write-back before the z path"
            ),
            "bias_formation": (
                "the one-variable residual skip-product precision change produces stable negative "
                "aligned scaling of the actual first-step AdamW write on real Mamba text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real Mamba text states, native sequential selective scan versus the same-input "
                "FP32 residual D*hidden_states product reference has mean write RMS 29.23%; the "
                "confirmation aligned-write interval is [-5.75%,-3.77%]. The held-out calibration-"
                "direction interval crosses zero, so this is a scoped aligned skip-product result, "
                "not an additive population or long-run loss claim. It is a distinct output boundary "
                "from state-to-C contraction, delta-softplus and causal-convolution probes, although "
                "all share a broad precision/materialization mechanism."
            ),
            "next_needed_observation": (
                "an independent Mamba text bank or second checkpoint, backend identity capture, and a "
                "declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_d_skip_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_d_skip_materialization_natural_16_20260918.json",
                "scripts/run_mamba_state_output_contraction_natural_probe.py",
            ],
            "derived": mamba_d_skip_materialization_evidence(),
        },
        {
            "problem_group": "mamba_z_gate_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native-dtype Mamba selective-scan output times SiLU(z) gate versus an otherwise "
                "identical FP32 product with native-dtype write-back; recurrence, state/C contraction "
                "and D skip remain unchanged"
            ),
            "bias_formation": (
                "the one-variable z-gate product precision change produces stable negative aligned "
                "scaling of the actual first-step AdamW write on real Mamba text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and loss consequence not measured",
            "what_is_proven": (
                "on 32 real Mamba text states at layer-0 out_proj.weight, native scan-output times "
                "SiLU(z) multiplication versus the same-input FP32 product has mean write RMS 28.36% "
                "and confirmation aligned-write interval [-4.479%,-3.253%]. The held-out additive "
                "direction interval crosses zero, so this is a scoped aligned z-gate result, not an "
                "additive population or loss-quality claim. It is distinct from the Mamba state-output, "
                "D-skip, softplus and causal-convolution boundaries."
            ),
            "next_needed_observation": (
                "an independent Mamba text bank or second checkpoint, backend identity capture, and a "
                "declared multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_z_gate_materialization_wikitext_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_z_gate_materialization_natural_32_20260918.json",
                "results/property/new_problem_group_search_v1/mamba_z_gate_materialization_natural_16_20260918.json",
                "scripts/run_mamba_z_gate_materialization_natural_probe.py",
            ],
            "derived": mamba_z_gate_materialization_evidence(),
        },
        {
            "problem_group": "mamba_fused_selective_scan_reassociation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "the actual mamba_ssm fused selective-scan/reassociation boundary versus an "
                "explicit same-layer sequential recurrence; the selected-layer causal convolution, "
                "x_proj, dt_proj, state/output contraction, gate and downstream model path are held fixed"
            ),
            "bias_formation": (
                "a local actual fused layer produces a stable negative aligned scaling of the first-step "
                "AdamW write, and replacing only its selective scan with an explicit differentiable "
                "recurrence reproduces that signed aligned component on the confirmation states; the "
                "full effect vector is not claimed identical"
            ),
            "training_outcome": "single-step parameter-write aligned bias; no multi-step loss endpoint",
            "what_is_proven": (
                "On 16 real Mamba text states at layer 3, the actual installed mamba_ssm fused scan "
                "versus an all-layer sequential reference has mean write-effect RMS 31.46%; the 12-state "
                "confirmation aligned-write interval is [-8.74%,-3.32%], with all 12 values negative. "
                "A same-layer source cut keeps the actual causal convolution and projection inputs and "
                "replaces only the selective scan by an explicit recurrence; its confirmation interval is "
                "[-9.41%,-2.95%], with all 12 values negative and a mean close to the fused result. "
                "The source-cut effect vector has mean cosine about 0.587 with the fused effect and mean "
                "relative residual about 0.897, so the closure is only for the signed aligned component, "
                "not for the complete residual vector. "
                "This closes a fixed-bank source boundary at the fused selective-scan/reassociation path, "
                "rather than assigning the whole multi-layer fused region to an unspecified Mamba cause. "
                "The held-out frozen additive direction still crosses zero, so the claim is a scoped aligned "
                "write bias and not a full vector-mean, population or loss-quality claim."
            ),
            "next_needed_observation": (
                "an independent Mamba text bank or second checkpoint, plus a declared multi-step training "
                "endpoint if an external-generalization or quality consequence is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/mamba_official_local_scan_layer3_16_20260920.json",
                "results/property/new_problem_group_search_v1/mamba_scan_source_cut_layer3_16_20260920.json",
                "scripts/run_mamba_official_local_scan_probe.py",
                "scripts/run_mamba_scan_source_cut.py",
            ],
        },
        {
            "problem_group": "gemma_gelu_backward_evaluation",
            "closure": "SOURCE_BOUNDARY_NARROWED_MEAN_BIAS_OPEN",
            "numerical_source": "tanh evaluation and expression spelling in a generated Triton tanh-GELU backward product",
            "bias_formation": "same-bank source variants change gradient/update response on the selected confirmation case; native tanh and fused multiply-add agree; the natural candidate expression is reproduced componentwise",
            "training_outcome": "not measured as an independent full training result",
            "what_is_proven": "on the same 32-state natural bank, native tanh and fused multiply-add agree through local/gradient/update/write while explicit exponential tanh changes the profile; a componentwise natural probe reproduces the candidate tanh, polynomial, derivative-factor and derivative expressions exactly on four reviewed states. The corrected path-preserving output-level intervention was rerun on all 32 states with the generated kernel's temporary association reproduced explicitly. The tested tanh, polynomial, derivative-factor and final-derivative variants account for about 0.597 of the candidate/reference write-effect norm in aggregate, but their direction cosine is about 0.461 and residual norm ratio about 0.898; therefore they are an important partial mediator, not a sufficient root. The superseded eight-state run is excluded from evidence. The exact remaining instruction/materialization path and the natural-input mean bias remain unresolved",
            "next_needed_observation": "an exact generated-instruction or intermediate-materialization intervention not covered by the tested arithmetic boundaries; a natural GELU population claim would additionally require a predeclared independent source distribution",
            "evidence": ["results/property/numerical_coverage_v1/gemma_gelu_capture_v2", "results/property/numerical_coverage_v1/gemma_gelu_tanh_exp_confirmation_v3", "results/property/numerical_coverage_v1/gemma_gelu_tanh_native_confirmation_v3", "results/property/numerical_coverage_v1/gemma_gelu_fma_confirmation_v1", "results/property/numerical_coverage_v1/gemma_gelu_tanh_fma_confirmation_v7", "results/property/root_cause_closure_v1/gelu_source_factorial_v1.json", "results/property/root_cause_closure_v1/gelu_natural_intermediates_v1.json", "results/property/root_cause_closure_v1/gelu_componentwise_intermediates_v1.json", "results/property/root_cause_closure_v1/gelu_arithmetic_interventions_4_20260920.json", "results/property/root_cause_closure_v1/gelu_single_source_mediation_v4_32_20260921.json", "scripts/analyze_gelu_source_factorial.py", "scripts/capture_gelu_natural_intermediates.py", "scripts/capture_gelu_componentwise_intermediates.py", "scripts/run_gelu_single_source_intervention.py"],
            "derived": gelu_evidence(),
        },
        {
            "problem_group": "gptneo_gelu_native_fp32_materialization",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native GPT-Neo NewGELUActivation evaluated in the model dtype versus the same "
                "input evaluated with the explicit FP32 tanh-GELU formula followed by one original-"
                "dtype write-back"
            ),
            "bias_formation": (
                "the one-variable activation-evaluation/materialization change produces a stable "
                "negative aligned scaling of the actual first-step AdamW write across 32 real text states"
            ),
            "training_outcome": "single-step parameter-write aligned bias; loss interval crosses zero and no multi-step quality claim is made",
            "what_is_proven": (
                "On 32 real text states from the declared GPT-Neo-125M bank, the native NewGELU "
                "candidate and the same-input FP32 tanh-GELU reference have mean write RMS "
                "21.77%, with a normal 95% aligned-write interval [-2.4275%,-2.2874%]. The "
                "native expression is reproduced exactly in BF16 (mean relative L2 0), while the "
                "native-versus-FP32 formula difference is about 0.2515% relative L2. The loss "
                "interval crosses zero, so this closes the arithmetic source and fixed-suite aligned "
                "write effect, not a population mean or long-run quality consequence. It is a "
                "separate native implementation boundary from the Gemma generated-Triton joint "
                "derivative/product-factor boundary; Gemma's natural mean-bias question remains "
                "separate, even though both belong to the broad GELU semantic family."
            ),
            "next_needed_observation": (
                "an independent GPT-Neo text bank or second checkpoint for external confirmation, "
                "and a multi-step training endpoint if a quality consequence is needed"
            ),
            "evidence": [
                "results/property/root_cause_closure_v1/gptneo_gelu_natural_32_20260920.json",
                "results/property/root_cause_closure_v1/gptneo_gelu_natural_16_20260920.json",
                "scripts/run_gptneo_gelu_natural_probe.py",
            ],
            "derived": gptneo_gelu_evidence(),
        },
        {
            "problem_group": "gemma_rms_feature_reduction_order",
            "closure": "NEGATIVE_SOURCE_CONTROL_NO_ROOT_CAUSE",
            "numerical_source": "FP32 feature-reduction order",
            "bias_formation": "isolated order change is tiny or exact identity and has no confirmed direction",
            "training_outcome": "not measured",
            "what_is_proven": "the tested order change is not sufficient to explain the earlier combined endpoint effect",
            "next_needed_observation": "none for this negative control; a different RMS source would require a new predeclared intervention",
            "evidence": ["results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json"],
            "derived": rms_order_evidence(),
        },
        {
            "problem_group": "granite_router_topk_selection",
            "closure": "NEGATIVE_CONTROL_FIXED_SUITE_IDENTITY",
            "numerical_source": "selection tie-order variation with unchanged selected set",
            "bias_formation": "no local, gradient, write or loss difference observed in 24 fixed states",
            "training_outcome": "no difference observed; no population or quality claim",
            "what_is_proven": "the tested legal tie-order variation did not create a measurable training difference under the declared deterministic protocol",
            "next_needed_observation": "none unless a different selection semantic (changed selected set, NaN, or non-tie scores) is explicitly studied",
            "evidence": ["results/property/numerical_coverage_v1/granite_selection_suite24_v1/summary.json"],
            "derived": selection_evidence(),
        },
        {
            "problem_group": "granite_moe_expert_contribution_order",
            "closure": "SOURCE_CLOSED_DECLARED_BANK_PROJECTED_MEAN_SUPPORTED_NATURAL_GENERALIZATION_OPEN",
            "numerical_source": "FP32 expert-contribution accumulation order",
            "bias_formation": "reversing the legal expert accumulation order changes the FP32 local/gradient values and produces a signed write projection with a positive one-sided mean bound on a new with-replacement empirical-bank confirmation; this is scoped to the declared bank and carrier",
            "training_outcome": "not measured",
            "what_is_proven": "the source is isolated for the stated MoE boundary; the new 32-calibration/64-confirmation run supports a positive signed parameter-write projection mean under iid-with-replacement draws from the declared empirical bank. This implies a nonzero vector mean only for that scoped population and test assumptions, not for all Granite or natural pretraining states",
            "next_needed_observation": "a different-model or genuinely held-out state distribution for external generalization, plus a warm-moment or multi-step comparison before claiming a training consequence",
            "evidence": [
                "results/property/training_numerical_analysis_v2/granite_expert_order_confirmation_v2/recomputed.json",
                "results/property/training_numerical_analysis_v2/granite_expert_order_confirmation_v2/raw.json",
                "results/property/granite_expert_order_population_v1/protocol.json",
                "results/property/granite_expert_order_population_v1/result.json",
                "results/property/training_numerical_analysis_v2/",
            ],
            "derived": granite_expert_evidence(),
        },
        {
            "problem_group": "olmoe_router_expert_accumulation",
            "closure": "SOURCE_CLOSED_FIXED_NATURAL_SCOPED_ALIGNED_BIAS",
            "numerical_source": (
                "native BF16 repeated-destination expert contribution accumulation versus an "
                "otherwise identical FP32 accumulation followed by one BF16 write-back"
            ),
            "bias_formation": (
                "the one-variable accumulation-dtype intervention produces stable negative aligned "
                "scaling of the actual cold-start AdamW write while routing and expert selection stay fixed"
            ),
            "training_outcome": "single-step parameter-write aligned bias; additive direction and multi-step loss consequence not measured",
            "what_is_proven": (
                "on 16 real OLMoE text states, the candidate and FP32-repair arms have identical selected "
                "experts and routing weights. The confirmation half has a parameter-write aligned interval "
                "of [-5.97%,-4.54%] and mean -5.25%, while the held-out calibration-direction projection "
                "interval crosses zero. This is a scoped natural router-combine aligned update bias, distinct "
                "from the Granite FP32 order case; it is not an additive population mean or a loss-quality claim"
            ),
            "next_needed_observation": (
                "an independent OLMoE input bank or checkpoint and a declared multi-step training endpoint "
                "if a quality consequence or external generalization claim is needed"
            ),
            "evidence": [
                "results/property/new_problem_group_search_v1/olmoe_router_accum_natural_16_20260919.json",
                "scripts/run_olmoe_router_accum_natural_probe.py",
            ],
            "derived": olmoe_router_accumulation_evidence(),
        },
    ])
    # Gemma GELU was previously the only unresolved source row.  Promote it
    # only when the corrected, path-preserving 32-state intervention exists
    # and proves a jointly sufficient boundary.  The natural-population mean
    # bias remains a separate open claim and is intentionally encoded in the
    # closure suffix rather than in the root-cause frontier.
    gemma_gelu = next(
        row for row in rows if row["problem_group"] == "gemma_gelu_backward_evaluation"
    )
    gelu_mediation_path = ROOT / "results/property/root_cause_closure_v1/gelu_single_source_mediation_v6_32_20260921.json"
    gelu_mediation = (
        json.loads(gelu_mediation_path.read_text())
        if gelu_mediation_path.is_file()
        else None
    )
    gelu_joint = (gelu_mediation or {}).get("mediation_summary", {}).get(
        "reference_derivative_product", {}
    )
    gelu_joint_closed = bool(
        (gelu_mediation or {}).get("status")
        == "COMPLETE_PATH_PRESERVING_GELU_MEDIATION_INTERVENTIONS_V4"
        and gelu_joint.get("exact_write_count") == (gelu_mediation or {}).get("state_count")
        and float(gelu_joint.get("residual_norm_ratio_aggregate", 1.0)) <= 1e-9
    )
    if gelu_joint_closed:
        gemma_gelu["closure"] = "SOURCE_CLOSED_JOINT_DERIVATIVE_PRODUCT_MEAN_BIAS_OPEN"
        gemma_gelu["numerical_source"] = (
            "generated Triton tanh-GELU backward derivative/product path; replacing the derivative "
            "and product-factor FP32 intermediates jointly with same-input reference values reproduces "
            "the native/reference write effect"
        )
        gemma_gelu["bias_formation"] = (
            "the jointly sufficient derivative/product-factor materialization boundary mediates the "
            "candidate/reference write difference on the retained natural state bank; this is a source "
            "closure, not a natural-population mean-bias claim"
        )
        gemma_gelu["what_is_proven"] = (
            "On the 32-state natural bank, the path-preserving joint derivative/product-factor override "
            "has exact parameter-write agreement in all 32 states, aggregate effect ratio 1.0, cosine "
            "1.0 and residual 0. Tanh-only leaves the effect unchanged; factor-only and product-only "
            "are not sufficient. This closes the reviewed source at the joint intermediate-materialization "
            "boundary. A 128-draw with-replacement probe from the declared Gemma-4 trajectory bank "
            "supports a negative aligned native-minus-reference write mean, but a finer instruction-level "
            "decomposition and the unrestricted natural-input population mean bias remain outside this result."
        )
        gemma_gelu["next_needed_observation"] = (
            "a bank representing the unrestricted natural training population and a declared quality "
            "endpoint are needed for stronger claims; any finer instruction decomposition is optional "
            "refinement below the jointly sufficient boundary"
        )
        gemma_gelu["evidence"] = list(gemma_gelu.get("evidence", [])) + [
            "results/property/root_cause_closure_v1/gelu_single_source_mediation_v6_32_20260921.json",
            "results/property/root_cause_closure_v1/gelu_mean_probe_empirical_bank_128_20260922.json",
        ]
        gemma_gelu["derived"] = gelu_evidence()
    family_frontier = operator_family_frontier()
    payload = {
        "schema": "kernel-analyzer-current-root-cause-closure-v1",
        "status": "CURRENT_EVIDENCE_LEDGER_WITH_EXPLICIT_OPEN_BRANCHES",
        "counting_rule": "One active row is one deduplicated scientific problem group; negative-control rows are retained separately for calibration and are not part of the active case count. Model, layer, state and implementation conditions are evidence within a group.",
        "closure_scale": {
            "END_TO_END": "source, propagation, targeted intervention and a declared training outcome are connected",
            "SOURCE_CLOSED": "the local arithmetic/source choice is isolated, while natural bias or quality consequence remains open",
            "SOURCE_BOUNDARY_NARROWED": "natural replay or source-choice probes narrow the reviewed boundary, but a candidate-path intervention has not established a unique sufficient cause",
            "PARTIAL": "a transport or semantic-region contributor is isolated, but the whole region has additional sources",
            "NEGATIVE_CONTROL": "the tested variant did not produce a difference; this does not prove every variant is safe",
            "OPEN": "existing measurements show a difference or response, but competing source explanations remain",
        },
        "summary": {
            "problem_group_count": len(rows),
            "active_problem_group_count": sum(
                not r["closure"].startswith(("NEGATIVE_CONTROL", "NEGATIVE_SOURCE"))
                for r in rows
            ),
            "active_case_count": sum(
                not r["closure"].startswith(("NEGATIVE_CONTROL", "NEGATIVE_SOURCE"))
                for r in rows
            ),
            "end_to_end_count": sum(r["closure"].startswith("END_TO_END") for r in rows),
            "source_or_local_closed_count": sum(
                "CLOSED" in r["closure"] or r["closure"].startswith("END_TO_END")
                for r in rows
            ),
            "negative_control_count": sum(r["closure"].startswith("NEGATIVE_CONTROL") or r["closure"].startswith("NEGATIVE_SOURCE") for r in rows),
            "groups_with_open_branch": sum("OPEN" in r["closure"] for r in rows),
            "coverage_collection_count": len(coverage_collections),
        },
        "coverage_collections": coverage_collections,
        "not_counted_candidates": NOT_COUNTED_CANDIDATES,
        "source_record_inventory": source_record_inventory(),
        "generalization_benchmark_frontier": generalization_benchmark_frontier(),
        "operator_family_frontier": family_frontier,
        "root_cause_frontier": root_cause_frontier(rows, family_frontier),
        "rows": rows,
    }
    for row in rows:
        row["case_role"] = (
            "NEGATIVE_CONTROL"
            if row["closure"].startswith(("NEGATIVE_CONTROL", "NEGATIVE_SOURCE"))
            else "ACTIVE_BIAS_CASE"
        )
    # Keep the full row list for backwards-compatible audits, but expose the
    # scientific case population and calibration controls as separate
    # collections.  A negative control is retained evidence; it is not an
    # unresolved active problem and must not inflate the deduplicated case
    # count.
    payload["active_problem_groups"] = [
        row for row in rows if row["case_role"] == "ACTIVE_BIAS_CASE"
    ]
    payload["negative_controls"] = [
        row for row in rows if row["case_role"] == "NEGATIVE_CONTROL"
    ]
    payload["summary"]["root_cause_unresolved_count"] = len(
        payload["root_cause_frontier"]["open_problem_groups"]
    )
    # The active ledger is intentionally broader than the final scientific
    # result set: it retains cases whose source boundary is still open so the
    # missing intervention is auditable.  A bias is counted as a final result
    # only when it is not one of the explicit root-cause blockers above.
    open_root_ids = {
        item["problem_group"]
        for item in payload["root_cause_frontier"]["open_problem_groups"]
    }
    payload["final_root_cause_groups"] = [
        row
        for row in payload["active_problem_groups"]
        if row["problem_group"] not in open_root_ids
    ]
    payload["open_root_cause_groups"] = [
        row
        for row in payload["active_problem_groups"]
        if row["problem_group"] in open_root_ids
    ]
    payload["summary"]["root_cause_closed_final_count"] = len(
        payload["final_root_cause_groups"]
    )
    payload["summary"]["final_source_or_local_closed_count"] = sum(
        "CLOSED" in row["closure"] or row["closure"].startswith("END_TO_END")
        for row in payload["final_root_cause_groups"]
    )
    payload["summary"]["active_open_root_cause_count"] = len(
        payload["open_root_cause_groups"]
    )
    payload["summary"]["active_source_or_local_closed_count"] = sum(
        (
            row["case_role"] == "ACTIVE_BIAS_CASE"
            and ("CLOSED" in row["closure"] or row["closure"].startswith("END_TO_END"))
        )
        for row in rows
    )
    payload["summary"]["generalization_or_mean_open_count"] = sum(
        "OPEN" in row["closure"] for row in rows
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    # The mean audit reads the current ledger. Recompute it here so the human
    # summary cannot silently reuse an older set of problem groups.
    from analyze_all_case_mean_bias import build as build_mean_report

    mean_report = build_mean_report()
    mean_rows = {row["problem_group"]: row for row in mean_report["rows"]}
    if set(mean_rows) != {row["problem_group"] for row in rows}:
        raise ValueError("mean-bias audit and root-cause ledger have different problem groups")
    for row in rows:
        row["mean_bias_evidence"] = mean_rows[row["problem_group"]]["mean_bias_status"]
        if row["problem_group"] == "silu_backward_evaluation" and isinstance(row.get("derived"), dict):
            probe = row["derived"].get("declared_empirical_bank_mean_probe")
            population_test = mean_rows[row["problem_group"]].get("population_test")
            if isinstance(probe, dict) and isinstance(population_test, dict):
                probe["population_test"] = population_test
        if row["problem_group"] == "gemma_gelu_backward_evaluation" and isinstance(row.get("derived"), dict):
            probe = row["derived"].get("evidence", {}).get("declared_empirical_bank_mean_probe")
            population_test = mean_rows[row["problem_group"]].get("population_test")
            if isinstance(probe, dict) and isinstance(population_test, dict):
                probe["population_test"] = population_test
    # Keep the count of scoped mean-bias results separate from the count of
    # active scientific problem groups.  A source/trajectory case may be an
    # active problem without supporting a population mean-bias claim.
    scoped_mean_bias_count = sum(
        row["mean_bias_evidence"].startswith(("SUPPORTED_ALIGNED_MEAN", "SUPPORTED_PROJECTED_MEAN"))
        for row in rows
        if row["case_role"] == "ACTIVE_BIAS_CASE"
    )
    payload["summary"]["scoped_mean_bias_supported_count"] = scoped_mean_bias_count
    payload["summary"]["active_without_scoped_mean_count"] = (
        payload["summary"]["active_case_count"] - scoped_mean_bias_count
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    mean_out = (Path(_OUTPUT_DIR) / "all_case_mean_bias_audit_v1.json" if _OUTPUT_DIR
                else ROOT / "results/property/root_cause_closure_v1/all_case_mean_bias_audit_v1.json")
    mean_out.parent.mkdir(parents=True, exist_ok=True)
    mean_out.write_text(json.dumps(mean_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    legacy = read("results/property/declared_persistent_4096/all_bias_case_audit.json")

    lines = [
        "# 全部案例：根因、bias 与训练后果",
        "",
        "本页是当前案例结论的唯一维护入口。按科学问题组去重，模型、层号和训练状态只作为组内证据。来源已定位、平均 bias、实际参数写入和训练 loss 分别判断；一个结论不能借另一个实验的协议补齐。机器记录及完整原始证据路径在 `results/property/case_causal_audit_v1/root_cause_closure_current.json`。",
        "",
        "本轮按统一的 native/reference 参数写入协议重新采集了 SiLU（128 个 DeepSeek 轨迹库有放回抽样状态）和 Gemma GELU（128 个 Gemma-4 轨迹库有放回抽样状态）。两者均支持声明经验状态库内的负 aligned 写入均值；这不推广为自然训练总体，也不替代独立 loss 干预。机器记录分别为 `results/property/root_cause_closure_v1/silu_mean_probe_empirical_bank_128_20260921.json` 与 `results/property/root_cause_closure_v1/gelu_mean_probe_empirical_bank_128_20260922.json`。",
        "",
        f"按‘最终结果必须有已闭合根因’的严格口径，当前最终问题组是 **{payload['summary']['root_cause_closed_final_count']} 个**；账本仍保留 **{payload['summary']['active_case_count']} 个活动证据组**，其中 **{payload['summary']['active_open_root_cause_count']} 组**因竞争性根因尚未排除而不计入最终结果。最终组中 **{payload['summary']['final_source_or_local_closed_count']} 组**已有声明边界内的来源或局部闭合，**{payload['summary']['scoped_mean_bias_supported_count']} 组**在声明的 history/经验输入分布下有 projected 或 aligned mean bias 支持；另 **{payload['summary']['root_cause_closed_final_count'] - payload['summary']['scoped_mean_bias_supported_count']} 组**虽有根因闭合，但目前只有来源、轨迹或局部响应证据，不能称为总体 mean bias。另有 **{payload['summary']['negative_control_count']} 个阴性控制**（保留用于校准，不计入活动案例），以及 **{payload['summary']['coverage_collection_count']} 个覆盖集合**（不计入问题组）。活动组中 {payload['summary']['end_to_end_count']} 个连到声明幅度的训练 loss 改善；平均偏向的逐组统计由同一脚本从保存数据复算。",
        "",
        "阴性控制已从活动问题调查中分离：`active_problem_groups` 保留全部活动证据，`final_root_cause_groups` 才是可以写入最终结果的根因闭合集合，`open_root_cause_groups` 只用于记录尚未完成的归因工作，`negative_controls` 只用于校准和显示测试的非阳性边界；完整兼容审计仍保留在 `rows`。因此本页的最终问题数量不把根因未闭合组或阴性控制计入。",
        "",
        "### 本轮去重增量与未计数候选",
        "",
        "上一轮已闭合的三个 Gemma-4 音频问题组继续保留：LightConv GLU 门控乘法物化 `gemma4_audio_lightconv_glu_product_materialization`、depthwise causal Conv1d backward 权重梯度累加 `gemma4_audio_lightconv_depthwise_conv_backward_accumulation`，以及音频下采样 Conv2d layer-1 的累加/物化 `gemma4_audio_subsampling_convolution_materialization`。本轮又闭合了同一真实音频语言图中 layer-0 attention-logit softcap 的独立边界 `gemma4_audio_attention_softcap_materialization`：16 个真实语音状态上只改变 softcap 的 BF16/FP64 求值，q_proj 写入 RMS 平均约 193.1%/215.3%，但确认半区方向区间跨零，因此只作根因闭合的固定集合总效应，不声称总体 signed bias。随后又闭合了 layer-0 attention probability 的 BF16 物化边界 `gemma4_audio_attention_softmax_probability_materialization`：只改变 softmax 概率在 value contraction 前是否写回 BF16，16 个真实语音状态的 q_proj 写入 RMS 约 1.77%，确认 aligned 区间严格为负；layer-1 重复区间跨零，因此该结果明确限定在 layer-0。与此同时，本轮新增并闭合了 GPT-Neo native GELU 的独立实现边界 `gptneo_gelu_native_fp32_materialization`，以及 Gemma 生成 Triton GELU 的联合 derivative/product-factor 中间物化边界：后者在 32 个自然状态上逐位复现 native/reference 写入，但自然总体 mean bias 仍另行判断。此前已加入的 DeBERTa c2p/p2c relative-attention 与 Bloom ALiBi attention 两组仍保留；它们分别代表不同的位置注意力数学边界，不与已有 RoPE、BERT QK、Qwen SDPA 或一般 MM 位置重复计数。新加入的 RWKV time-mix 组在同一 block、同一真实文本 bank 上只改变 `time_decay` 指数物化，且 decay-only 与 all-FP32 endpoint 完全一致而 value-only 不复现；本轮又在两个独立 32 状态真实文本库上闭合了 RWKV attention receptance sigmoid 的单变量 FP32 物化边界，确认半区均为负，形成区别于 time_decay 和 channel-mix 阴性筛查的新递归问题组。整体 Mamba fused recurrence 仍没有形成方向确认；`indexed_accumulation_reduction_order` 仅作为条件性算子级结果保留，不计入自然训练活动数。",
        "",
        "此外，DeepSeek embedding 的同一真实 cotangent 还完成了一个独立的最终梯度物化隔离：保持重复 token 的 index-add 完全不变，只比较 FP32 保留与 BF16 写回，16 状态确认 aligned 区间严格为负，但写入 RMS 约 3e-8，因此计为已闭合的微小来源组，不把它包装成实际训练风险。",
        "下列候选保留在机器账本的 `not_counted_candidates` 中，避免把探索性 endpoint、跨模型复现或运行阻塞误报为新的问题组。",
        "Mamba 的整体 fused recurrence 与 Softplus backward 候选仍保留在未计数候选。现在官方 fused backend 已在兼容环境中实际运行：32 个真实文本窗口的整体 fused-vs-sequential 对照有约 27.5% 的首步写入 RMS，但 24 个 held-out 投影为 11 正/13 负，近似 95% 区间 [-4.32e-4, 2.71e-4] 跨零；随后扩到 64 个真实 bank 状态仍为 15 正/17 负，均值约 -1.84e-3 的近似 95% 区间继续跨零；最新 126 个确认状态为 67 正/59 负，投影均值 -1.32e-4，aligned-gradient 区间 [-1.536e-2,1.256e-3] 仍跨零；同一语义在 mambapy 路径的 32 个状态也为 18 正/14 负，aligned-gradient 区间 [-1.681e-3,7.151e-4] 跨零。因此它关闭了执行阻塞，却没有满足稳定 signed-bias 的晋级条件。新增加的路径保持 softplus 物化探针是不同问题：它固定顺序递归，只改变 delta-softplus 的输入 dtype/FP32 计算选择，在 16/32 个真实文本状态上均得到稳定负 aligned write scaling，因而已进入活动问题组；这不等于整体 recurrence 的所有差异都来自 softplus。最近对 layer-0 `x_proj.weight` 的 24 状态重放也得到约 30.5% 写入 RMS，但 11/5 的方向分裂与 [-1.543e-2,1.177e-2] 的 aligned-gradient 区间跨零，因此仍是大幅候选而不是新的闭合问题组。更深层 1024-token 筛查中，layer-23 的 `A_log` 与 `dt_proj.bias` 逐状态为零，`x_proj` 方向也接近零均值。",
        "旧 Gemma RMS 记录仍有两个 forward-normalized endpoint 的固定集合写入 RMS 约 6.7% 和 8.2%，相关 row-square-sum/simple-backward endpoint 约 2.2%--7.7%，但这些旧记录都缺少精确运行 endpoint binding，因此继续保留为历史候选；新的 Gemma-4 同输入 RMSNorm probe 已把可复现的 cast-order 边界并入既有 `rmsnorm_cast_materialization` 组，而不是再计一个 Gemma 专属组。Gemma-4 causal NLL 已通过新的同 logits 24-state 边界 probe 进入活动组；本轮又用 128 个真实文档窗口闭合了 BERT-tiny masked-LM NLL 求值边界，旧 BERT 32-window screen 仅作为开发筛查保留。旧 Gemma softcapped-NLL capture 仅作为历史记录保留。另有 v4/v5/v9 Triton signature campaign 的 31 条有效真实模型 endpoint 记录，其中若干写入差异很大，但它们仍是包含上游差异的区域替换，暂不计为独立根因。",
        "对这 31 条区域记录，`scripts/build_natural_candidate_frontier.py` 已自动按语义线索拆成 8 个候选族：embedding dense backward、NLL、softmax/attention、normalization/reduction、RoPE/attention rotation、layout/transpose、SiLU 和 recurrence/Softplus。该机器账本只用于排序下一轮单变量隔离，不增加问题组数量；结果见 `results/property/case_causal_audit_v1/natural_candidate_frontier.json`。",
        "其中最高优先级的 DeepSeek embedding-dense 区域已做源级筛查：该融合 kernel 将 BF16 输入立即转为 FP32，并混合 NLL/归一化项，现有区域写入差异不能归因到唯一 kernel 数值根因。对同模型族 Qwen 的 embedding backward 做了同一输入、同一参数边界的独立 FP32 index-add probe，16 状态写入 effect RMS 均值仅约 0.0497%，远低于该融合区域约 51% 的 screen；在 FUSED_MIXED 原始 DeepSeek 128-token 输入库上复查 8 个状态，embedding-only 写入 effect RMS 约 0.755%，确认 aligned 区间约 [-0.00991%,-0.00286%]；同一输入库上固定 logits 的 native NLL 与显式 FP32 log-softmax/gather 对 lm_head/embedding carrier 给出逐状态零 gradient/write effect，loss 仅有数值舍入量级差异；再用同一真实 embedding cotangent 直接比较 BF16 gradient materialisation 与 FP32 gradient retention，16 个真实文本状态的 write effect RMS 约 3.0e-8，aligned 区间严格为负但效应极小。与此同时，在同一真实生成 fused embedding/NLL/normalization backward 调用边界上，只把 partial term 在归约前从 FP32 改为 BF16，16 状态 write RMS 约 1.70%，aligned 区间严格为负；FP64 reduction control 约 1.5e-6%，区间跨零。后续同边界的 base-term 和 final-value 物化干预在 16 状态上分别产生约 2.61% 和 1.68% 写入 RMS，确认 aligned 区间也严格为负；它们是同一 fused intermediate-materialization 问题的不同源分量，不另计新组。这一组干预闭合了该边界内的物化机制，但不解释更大 fused 区域的全部来源。因此大区域差异不能归因于 embedding 累加、最终 gradient cast、NLL loss 求值或 reviewed reduction precision 本身，剩余区域候选仍保留为需要运行时边界隔离的区域。记录见 `results/property/new_problem_group_search_v1/deepseek_embedding_dense_source_screen_20260918.json`、`results/property/new_problem_group_search_v1/qwen_embedding_backward_natural_16_20260920.json`、`results/property/root_cause_closure_v1/deepseek_embedding_backward_seq128_8_20260920.json`、`results/property/root_cause_closure_v1/deepseek_nll_same_logits_seq128_8_20260920.json`、`results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json`、`results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json` 与 `results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_terms_v7_20260920.json`。",
        "本轮额外复查了两个高信号候选：Qwen embedding dense 区域没有被独立 embedding backward probe 解释；BERT CUDA softmax 的 16/32 状态重跑复现了既有 `bert_attention_softmax_materialization` 的负 aligned scaling（32 状态约 -0.234%），因此是既有组的确认，不增加新组。当前候选池仍有 8 个族、31 条区域记录，但没有新的已隔离根因可晋级。对本地 26 个有配置记录的模型做语义缺口审计后，8 个可识别语义族全部已有对应问题组或明确的未闭合组，未发现一个可以直接进入新根因实验的未代表语义族；该审计仍只是调度依据，不把配置标签当作问题组证明。记录见 `results/property/case_causal_audit_v1/natural_semantic_gap_frontier.json`。",
        "对同一批 31 条记录的 aligned screen 显示 26 条记录在 32 个状态上全部为负 aligned scaling，5 条为混合或精确零；但这 31 条记录都允许上游差异，因而该现象只能作为候选排序信号，不能直接计作 26 个 bias 案例。屏幕结果保存在 `results/property/case_causal_audit_v1/signature_candidate_aligned_screen.json`。",
        "本轮还对两个高优先级候选做了自然边界筛查：BERT NLL 的 native cross-entropy 与同 logits 的显式 FP32 log-softmax/gather 在 32 个窗口上有非零差异，但 projected/aligned 区间均跨零；BERT embedding backward 与同一上游梯度的显式 FP32 index-add 的写入差异约为 5.46e-8，方向也跨零。二者保留为未计数筛查记录，不升级为新的自然问题组。对 DeepSeek/Qwen3 的 GQA repeat_kv 边界，native expand+reshape 与 repeat_interleave 在 k_proj、v_proj 两个真实参数上各 16 个状态逐位给出零 gradient/write/loss effect；因此该具体候选被记录为阴性筛查，不能解释较大的非隔离 layout 区域记录。随后对 DeepSeek layout/transpose 候选做了同一输入、同一 attention arithmetic 的路径保持复放：layer-25 v_proj 与 layer-13 q_proj 均在 8 个真实状态上逐位得到零 gradient/write/loss effect。因此这个具体 layout 拼写被排除为新问题组；更大的融合区域仍因包含其他操作和上游差异而保持未隔离。",
        "Gemma-4 audio attention 的 softmax 也做了同边界来源隔离：在 layer-0 和 layer-1 的 16 个真实语音状态上，只将 FP32 softmax 改为 FP64 softmax，写入 RMS 分别约 99.8% 和 98.9%，但 aligned-write 区间分别为 [-52.7%, +18.1%] 和 [-53.8%, +8.7%]，held-out 方向区间同样跨零。因此这是一个高方差候选，不能按严格 bias 口径晋升；记录保留在 `not_counted_candidates`，不计入问题组。",
        "随后三轮共 24 个模型的统一生成输入筛查只重新确认已有 RMSNorm 与 RoPE 族；其中生成小算子层面的 activation、linear、Softplus、softmax 和 SDPA 没有通过方向筛查，但独立的 Qwen3 真实 attention backend 路径保持探针随后确认了 SDPA/eager 问题组。Gemma-4 与 Ministral 的个别路径仍是无法测量。其余候选更可能是已有族的复现或未隔离区域，而不是可以直接计数的新问题组。",
        "BERT-tiny pooler tanh 候选在 8 状态初筛时方向不足，因此没有直接计数；扩大到 32 个真实文档窗口后，embedding carrier 的 held-out additive direction 区间变为正，而 pooler-dense 对照仍跨零，遂升级为 carrier-specific 的 `bert_pooler_tanh_materialization` 活动组。BERT attention residual-addition 在 32 个真实文本窗口的 CPU 和 CUDA 路径上均逐位得到零 gradient/write/loss effect，记录为 `bert_residual_addition_materialization` 的 exact-identity 阴性筛查；这只关闭了声明的两条路径，不外推到其他融合实现。",
        "外部 Liger/DFuzz 候选也完成了同一口径复核：ORPO 的表观差异来自参考公式漏乘 beta，TVD 来自 mean/batchmean reduction 不一致，SimPO 与正确显式公式一致，GeGLU 比较了 exact 与 tanh-GELU；SwiGLU、torch.compile 和 fused Adam 仍是区域或执行模式候选，尚无新的闭合自然问题组。机器记录见 `results/property/new_problem_group_search_v1/external_liger_candidate_triage_20260919.json`。",
        "Gemma-4 音频编码器曾作为上一轮语义空白候选并已计入活动集合；本轮在同一真实音频语言图中又闭合了 LightConv GLU gate-times-value 乘法边界，以及 depthwise causal Conv1d backward 权重梯度累加边界。GLU 的 16 状态确认 aligned-write 区间为 [-167.57%,-83.65%]；depthwise Conv1d 的 8 状态确认 ratio-of-sums aligned scaling 为 +1.975%，删去任一确认状态后仍为正，但逐状态方向混合。两者的 additive projection 与 loss 区间均跨零，不能外推为总体或质量结论。DeBERTa 的 32 个真实文本窗口在第 0、5 层的确认 aligned-write 区间分别为 [-87.57%,-83.80%] 与 [-81.08%,-79.35%]；Bloom 的 16 个窗口在第 0、11 层分别为 [-18.13%,-6.00%] 与 [-12.54%,-9.13%]。这些结果都只支持声明的 fixed-suite aligned scaling；DeBERTa 的辅助 MLM head 是本地新初始化，不能写成预训练 MLM 质量结果。完整记录见对应 `results/property/new_problem_group_search_v1/` JSON 与 GLU/depthwise/DeBERTa/Bloom 探针脚本。",
        "Gemma-4 音频下采样 Conv2d 的 layer-0 16 状态筛查仍跨零，因而不单独计数；但 layer-1 16 状态的原坐标 ratio-of-sums aligned scaling 约为 -38.0%，删去任一确认状态后仍为负，并有三状态 captured-input 前向边界检查，因此已晋级为 `gemma4_audio_subsampling_convolution_materialization`。它与 Gemma-3 视觉 patch convolution、Gemma-4 音频 output projection 和 LightConv depthwise Conv1d 是不同的算子边界；layer-0 的未确认结果作为同族阴性/异质性证据保留。",
        "Gemma-4 final-logit softcap 候选随后扩展到完整 26 个真实文本状态：logit-gradient endpoint 在 26/26 状态正向对齐，lm_head 参数写入确认半区则有 12/13 个负 aligned 值，描述性正态区间约为 [-0.001327,-0.000787]。因此来源边界、梯度方向和固定集合写入 bias 均已闭合，新增 `gemma4_final_logit_softcap_materialization` 问题组；它仍不外推为总体或 loss 结论。机器记录见 `results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_logits_26_20260919.json` 与 `results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json`。",
        "Granite MoE router score projection 的自然 full-route 筛查显示 15/16 状态会改变 selected expert set，但方向混合；随后冻结 native top-k 集合并扩大到 32 个真实文本窗口，确认半区 15/16 个 aligned router-write 值为负，均值约 -51.72%，描述性 95% 区间约为 [-67.68%,-35.75%]。因此它现在作为独立的 `granite_moe_router_score_materialization` 问题组计入：根因是 router score/weight 物化边界，而不是 expert selection 或 expert contribution order；仍不外推为总体或 loss 结论。随后在 OLMoE 同一 source-isolated 边界上扩展到 128 个真实文本状态，aligned-write 区间为 [-1.503%,-0.279%]，确认该根因可跨模型复现；按去重规则合并到 Granite 组，不增加新组。机器记录见 `results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json` 与 `results/property/new_problem_group_search_v1/olmoe_router_score_natural_128_20260920.json`。",
        "",
        "| 类型 | 问题组 | 数值来源定位 | 平均 bias 证据 | 训练后果 | 证据 |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["final_root_cause_groups"]:
        group = row["problem_group"]
        lines.append(
            f"| {row['case_role']} | `{group}`<br>{PROBLEM_NAMES[group]} | {ROOT_CAUSE_LEVEL[group]} | "
            f"{mean_bias_description(row['mean_bias_evidence'])} | "
            f"{TRAINING_RESULT[group]} | {short_evidence_link(row)} |"
        )
    lines += [
        "",
        f"活动证据组中另有 **{payload['summary']['active_open_root_cause_count']} 组**未进入上表：它们确实观察到差异或路径效应，但现有单变量证据仍允许多个根因解释。它们只列在下一节的开放根因表中，不能作为最终 bias 结果。",
    ]
    lines += [
        "",
        "### 根因已定位但仍有范围限制的对象",
        "",
        "以下对象的声明范围根因已经定位，但仍有自然总体、跨路径或更细指令层面的未决范围；这不等同于根因未闭合：",
        "",
        "- **融合 RoPE 与位置缩放：**真实 vision-RoPE source factorial 显示 FP64 trig + native BF16 rotation 是 exact identity，而 FP32 multiply/add + BF16 write-back 单独复现 native→FP32 的负 aligned write effect，full-FP32 与其一致；因此声明的最终旋转乘加/物化来源已闭合。Qwen3 标准 RoPE 的 16 状态自然 probe 也显示同族的负 aligned 写入区间，但按同一 RoPE 物化问题合并，不新增模型专属组。自然总体 mean bias、text-path universality 和 loss 仍需另外的声明协议。",
        "",
        "- **Gemma GELU backward 求值：**同一 32-state 自然 bank 上，路径保持的 derivative 与 product-factor 两个 FP32 中间量联合替换后，32/32 状态参数写入逐位复现 reference（effect ratio=1、余弦=1、残差=0）；tanh-only、factor-only 和 product-only 均不是充分替换。因此来源在声明范围内闭合到联合中间物化边界；更细的指令分解和自然总体 mean bias 仍开放。",
        "",
        "### 冻结 benchmark 的根因边界",
        "",
        "冻结的 16 个 benchmark 位置均有测量结果，但其 AOT 端点替换本身不隔离单计算来源；它们作为方法验证集合逐项保留，不计作新的根因问题组。",
        "",
        "### 当前为何停止继续离线归因",
        "",
        "下面的阻塞是由现有观测的可识别性决定的，不是尚未整理文件。没有新的同输入单变量干预或新的自然状态观测，继续重算聚合量不会增加根因信息。",
        "",
        "对 attention 区域，S_bwd 语义来源和五个路径保持单变量替换已经闭合了该声明区域的根因：它确实是 attention-backward 状态到 q-projection 的实现边界，而不是一个未定位的随机差异。现有 baseline 与单变量替换仍不足以把总响应唯一分配给 query、key、residual-stream producer 或交互项；这属于已闭合语义区域内的可选细分，不再把整个 bias 排除在最终根因结果之外。",
        "",
        "| 问题组 | 当前阻塞 | 下一条可区分观测 |",
        "|---|---|---|",
    ]
    for item in payload["root_cause_frontier"]["open_problem_groups"]:
        lines.append(
            f"| {item['problem_group']} | {item['why_offline_stops']} | {item['next_observation']} |"
        )
    lines += [
        "",
        "目录中仅有覆盖/端点测量而没有来源隔离的族，同样不升级为根因："
        + "、".join(
            item["family_id"]
            for item in payload["root_cause_frontier"]["measurement_only_operator_families"]
        )
        + "。这些族若要进入根因表，必须先建立声明 reference、同输入边界和单变量干预；"
        "仅增加位置数不会消除当前不可识别性。",
        "",
        "当前仍无法形成有效统一测量的族为："
        + "、".join(
            f"`{item['family_id']}`"
            for item in payload["root_cause_frontier"]["blocked_operator_families"]
        )
        + "。它们的执行失败、超时或 reference 缺失不被计作阴性。仅有外部证据而没有目录位置的族为："
        + "、".join(
            f"`{item['family_id']}`"
            for item in payload["root_cause_frontier"]["external_only_operator_families"]
        )
        + "，同样不计入当前问题组。",
        "",
        "Liger、Granite expert 顺序、AdamW8bit、saved-P 与 Qwen3 eager/SDPA backend 已有各自声明范围内的来源定位。MM/GEMM 的四个具体位置也各有来源分解；它们没有被合并成一个跨所有矩阵乘法的共同物理根因。Gemma RMS、Granite Top-k，以及本轮 Qwen3 mask、attention contraction expression 和 q_proj/v_proj output-layout 对照，是有效阴性控制，不属于‘差异复现失败’或待定位阳性。",
        "",
        "### 已定位来源但尚未建立更强结论",
        "",
        "AdamW8bit 的证据支持声明 history 下 aligned 标量平均效应及训练修改收益，未证明全向量均值是 loss 改善的唯一中介。Liger 和 Granite 的正投影均值只针对各自有限经验输入库、固定参数载体和冷启动一步 AdamW。Softmax saved-P 的局部不一致与参数传播已测，但自然训练状态总体的均值和稳定质量损害未建立。Attention 到 q-projection 的语义区域根因已闭合，但 query/key/residual-stream 贡献的细分交互不是总体 bias 或根因闭合的前置条件。SiLU 与 Gemma GELU 的联合 product/factor 或 derivative/product 物化边界均已通过 32-state 路径保持干预闭合，但两者自然训练总体 mean bias 仍未测试。RoPE 的新 source factorial 已在真实 vision operands 上闭合 FP32 multiply/add 后 BF16 写回这一声明边界内的来源，但不外推到总体 mean bias。",
        "",
        "若校准方向只用校准单位确定、确认单位按声明机制独立抽样且其单侧均值下界有效，那么确认投影均值为正可推出**该声明分布**的向量均值非零。对已知有限经验输入库，有放回抽样只给这个库定义的分布以条件性结论；它不增加新的自然训练状态。固定集合的正投影计数、非零 RMS 或单条轨迹分叉都不能代替这个检验。逐项估计量与区间保存在 `results/property/root_cause_closure_v1/all_case_mean_bias_audit_v1.json`。",
        "",
        "### 已撤下或过期的解释",
        "",
        "Gemma square-sum 端点的实际执行源与声明对象不符，已撤出活动问题组；原诊断记录保留。旧 Liger 8192 维摘要与公式 update 的约 `3.08e-9` 不是真实写入测量；后续长度 64 的原坐标 torch AdamW 写入复采才用于写入结论。旧 Granite 固定集合仅 3/16 个确认状态改变单个写入坐标、方向投影为零；后来的经验库抽样是不同协议，不能回填为旧集合的前瞻结论。",
        "",
        "## 本轮新增的可核对结论",
        "",
        "Liger 的同精度加法顺序来源已有长度 64 和 256 的固定集合测量：参数梯度沿校准集合所得 schedule-mean 方向的正投影数分别为 14/16 和 11/16。另一次长度 64 重放直接测量了真实 torch AdamW 参数写入，32 状态写入 RMS 约 1.391e-6；写入自己的校准方向在确认半区有 12/16 正投影。旧采集将状态编号传作独立单位而产生的总体标签已不被当前汇总采用；write-v1 内梯度参考误取 hidden gradient 的字段也被排除，正确的同银行梯度数据来自 length64_confirmation。此前自然标签、反转标签和常量标签探针尚未确定不抵消的充分条件；新增的 32 校准/64 确认有放回抽样在原坐标真实写入上得到正的冻结方向投影均值，单侧 95% 下界为 1.762e-7；长度 256 的另一份经验 bank 同样得到正的单侧 95% 下界 1.551e-7。这只支持各自声明经验 bank、载体和冷启动一步 AdamW 协议下的 projected mean bias，不能推广到自然总体。1024 步顺序对照的 loss 差异回到零，长期质量结论仍开放。",
        "",
        "SiLU 的 source-factorial run3、自然中间量 probe 与严格路径保持干预现在形成闭环：exp/分母的自然差异首先出现在 reciprocal/division；但单独替换输出端、FMA 或 derivative-side sigmoid 仍不充分。将生成 kernel 的 derivative product 与 derivative-factor 两个 FP32 中间物化值同时替换为同输入参考值后，32 个真实状态全部逐位复现 native/reference 参数写入（effect ratio 1.0、余弦约 1.0、残差 0）；factor-only 约覆盖 0.974，product-only 约覆盖 0.490。由此来源闭合在联合 product/factor 物化边界；自然训练总体 mean bias 仍需独立状态库。机器复算见 `results/property/root_cause_closure_v1/silu_source_factorial_v1.json`、`results/property/root_cause_closure_v1/silu_intermediate_probe_v1.json`、`results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json`、`results/property/root_cause_closure_v1/silu_single_source_intervention_v2_4_20260920.json`、`results/property/root_cause_closure_v1/silu_all_divisions_intervention_v3_4_20260920.json`、`results/property/root_cause_closure_v1/silu_arithmetic_interventions_v4_4_20260920.json`、`results/property/root_cause_closure_v1/silu_single_source_intervention_v4_16_20260920.json`、`results/property/root_cause_closure_v1/silu_single_source_mediation_v6_16_20260921.json`、`results/property/root_cause_closure_v1/silu_single_source_mediation_v7_16_20260921.json` 与 `results/property/root_cause_closure_v1/silu_single_source_mediation_v7_32_20260921.json`。",
        "",
        "RMSNorm 的自然训练 probe 将原生 OLMoE、Qwen3 与 Gemma-4 的真实训练图分别和一个只改变物化位置的参考比较：归一化结果先 cast 到输入 dtype 再乘 BF16 权重，或保留 FP32 权重乘法后一次写回。OLMoE 26 状态、Qwen3 26 状态和 Gemma-4 16 状态均出现不跨零的 aligned 写入效应；Gemma-4 的区间约为 [-20.16%, -9.31%]。新增 Qwen3 layer-13 q_norm 与 k_norm 各 16 状态也复现同一负 aligned 区间（约 -2.3% 至 -5.0%），因此合并为同一物化根因而不增加问题组。三个 loss 区间均跨零，因此当前闭合的是三种真实模型、单步 AdamW 条件下的 **aligned update bias**，不是 additive 方向 bias、总体保证或训练质量结论。机器复算见 `results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json`、`results/property/new_problem_group_search_v1/gemma4_rmsnorm_natural_16_20260918.json`、`results/property/new_problem_group_search_v1/qwen3_qnorm_natural_16_20260918.json`、`results/property/new_problem_group_search_v1/qwen3_knorm_natural_16_20260918.json` 及对应 probe。",
        "",
        "DeepSeek/Qwen3 embedding backward 是本轮新增的另一组自然问题：同一模型、同一 token 输入和同一 BF16 权重下，只把 native 重复 token 梯度累加替换为 FP32 index-add，再恢复原参数写回。24 个真实文本状态的写入 RMS 平均约 1.049%，确认半区 aligned-write 均值为 -0.006848%，正态 95% 区间 [-0.008211%, -0.005485%]，而 additive 方向区间跨零；前向 loss 完全相同。因此这里已经闭合的是 embedding backward 的累加精度边界与固定集合 aligned scaling，不是总体均值、长期 loss 或新的 fused-CE 问题。机器记录见 `results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json` 与 `scripts/run_deepseek_embedding_backward_natural_probe.py`。",
        "Gemma-4 causal NLL 是本轮新增的独立自然组：同一 Gemma-4 E2B checkpoint、同一输入和同一 logits 图上，只将 native `cross_entropy` 替换为显式 FP32 shifted `log_softmax`/`gather`。24 个真实文本状态的 lm_head 写入 RMS 平均约 0.576%，确认半区 aligned-write 均值为 -0.001611%，正态 95% 区间 [-0.001762%,-0.001459%]，而 additive 方向和 loss 区间均跨零。这闭合的是 NLL 求值边界，不是上游 soft-capped logits，也不与 Liger 的 dW 分块累加重复。机器记录见 `results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json` 与 `scripts/run_gemma4_softcapped_nll_natural_probe.py`。BERT-tiny masked-LM NLL 也已在 128 个真实文档窗口上完成同 logits 复核：native cross-entropy 与显式 FP32 `log_softmax`/`gather` 的 lm_head 写入 aligned 区间为 [+0.00246%,+0.02550%]，但 additive 方向和 loss 区间跨零，因此只闭合固定集合的 NLL aligned-bias 边界，不作总体或质量结论。机器记录见 `results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json`。",
        "",
        "Gemma-3 视觉 patch convolution 也已形成独立的自然组：native BF16 累加与同输入 FP32 累加、原 dtype 写回的单变量对照，在 26 个图文状态上给出约 106.63% 的写入 RMS，aligned-write 均值 -55.78%，正态 95% 区间 [-63.90%,-47.67%]；additive 方向和 loss 区间均未确认。这是一个幅度很大的固定集合 aligned scaling 根因，不应被改写为视觉模型整体质量崩溃或跨 checkpoint 保证。机器记录见 `results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json`。",
        "",
        "Gemma GELU 的同一位置、相同输入和同一 32-state 自然 bank 上，显式指数重构会改变 local/gradient/update/write profile，而 native tanh 与融合乘加表达式在原坐标统计中逐阶段一致。逐分量自然 probe 复现了候选表达式；修正参考常数并按生成 kernel 临时量顺序重做后，联合替换 derivative 与 product-factor 两个 FP32 中间物化值，在 32/32 状态逐位复现 native/reference 参数写入（effect ratio=1、余弦=1、残差=0）。tanh-only 不改变写入，factor-only/product-only 不是充分替换。因此来源在声明范围内闭合到联合中间物化边界；更细指令分解与自然输入总体 mean bias 仍未证明。机器复算见 `results/property/root_cause_closure_v1/gelu_source_factorial_v1.json`、`results/property/root_cause_closure_v1/gelu_natural_intermediates_v1.json`、`results/property/root_cause_closure_v1/gelu_componentwise_intermediates_v1.json`、`results/property/root_cause_closure_v1/gelu_single_source_intervention_v6_32_20260921.json`。",
        "",
        "RoPE 的同输入算术探针在多个随机 seed 和三个位置尺度上均显示 tl_math 与 libdevice 输出逐位一致；四个自然 Ministral 状态上，FP64 trig + native BF16 rotation 为 exact identity，而 FP32 multiply/add + BF16 write-back 单独复现 native→FP32 的负 aligned write effect，full-FP32 与其一致。因此声明的 vision-RoPE source factorial 已闭合最终旋转乘加/物化边界；自然总体 mean bias、text-path universality 和 moments 与 step counter 的分离仍开放。机器复算见 `results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json` 与 `results/property/root_cause_closure_v1/ministral_vision_rope_source_factorial_natural_8_20260920.json`。",
        "",
        "Attention 复合区域的五组八状态 probe 命中 layer-23 q_proj 的真实生成路径：生成的 softmax-backward 输出与 bmm_75 左输入在值和 buffer 上均一致，替换它后八个状态的 q_proj 梯度 L2 变化为 0.010805--0.014142；只替换同一调用的上游梯度输入 U、pre-softmax score、max/normalizer statistics，或 bmm_75 右侧 key carrier，八个状态也都产生非零 q_proj 梯度变化。它们是分别成立的单变量路径证据，变化量不能直接相加或解释成独立贡献比例；因此不把多来源 attention 区域错误归因为单一 kernel。输入交互、上游 query/key producer 与 residual-stream 来源仍未唯一分解。机器复算见 `results/property/root_cause_closure_v1/l23_softmax_kernel_output_probe_v1.json`、`l23_softmax_kernel_input_probe_v1.json`、`l23_softmax_scores_probe_v1.json`、`l23_softmax_statistics_probe_v1.json` 与 `l23_softmax_carrier_right_probe_v1.json`。",
        "",
        "Gemma RMS 的 endpoint-by-endpoint 归约顺序干预说明，早期同时替换造成的较大差异来自上游传播，不能归因给第二个 RMS endpoint。Granite Top-k 的 24 状态对照没有改变选中集合、值、梯度、写入或 loss。Granite expert 累加顺序的原始 32 状态确认半区只有 3/16 状态各改变 1 个写入坐标，原坐标 RMS 为 0.002281%，且其固定集合确认投影为零。随后在结果揭示前冻结方向并从声明经验 bank 有放回独立抽取 32 个校准与 64 个确认单位；参数写入投影均值为 3.995e-6，单侧 95% t 下界为 2.079e-7，53/64 为正。这支持声明经验 bank、载体和冷启动写入协议下的 projected mean bias，并不推广到所有 Granite 或自然预训练状态。逐阶段复算见 results/property/root_cause_closure_v1/reduction_mean_bias_audit_v1.json；独立确认见 results/property/granite_expert_order_population_v1/result.json。",
        "",
        "OLMoE router 重复目标 expert 累加是本轮新增的自然问题组：在同一真实 OLMoE checkpoint 和 16 个文本状态上，只将 native BF16 repeated-destination accumulation 替换为 FP32 accumulation 后一次 BF16 写回，expert 选择与 routing weights 在所有状态保持一致。确认半区的参数写入 aligned 区间为 [-5.97%,-4.54%]，而 held-out calibration-direction projection 区间跨零；因此这是一个独立的 OLMoE router-combine aligned update bias，不是 additive 方向总体均值或多步 loss 结论。它与 Granite 的 FP32 expert-order 组分别记录：前者改变累加 dtype，后者改变同精度累加顺序。机器记录见 `results/property/new_problem_group_search_v1/olmoe_router_accum_natural_16_20260919.json`。",
        "OLMoE router score 物化先在 16/32/64 状态上表现为高幅度但高方差候选；继续扩展到 128 个真实状态后，冻结 native top-k、只把 router score 物化改为 FP32 的 aligned-write 区间为 [-1.503%,-0.279%]。这确认了与 Granite 相同的 router score/weight 物化根因，因此按去重规则并入既有 `granite_moe_router_score_materialization` 组，而不是新建第二个问题组。无效的 embedding 近似 top-k 初跑已废弃，机器记录使用 `olmoe_router_score_natural_16_20260920.json`、`olmoe_router_score_natural_32_20260920.json`、`olmoe_router_score_natural_64_20260920.json` 与 `olmoe_router_score_natural_128_20260920.json`。",
        "Granite MoE top-k routing-weight 的新单变量探针固定了 router logits、选中 expert、expert 计算和重复目标累加，只改变 softmax gate 权重在乘法前是否从 FP32 写回 BF16。审计文档语料上的 16/64/128 个自然状态，expert-output 写入 RMS 分别约为 15.9%、15.5% 和 13.9%，但确认 aligned 区间分别为 [-0.75%,+0.70%]、[-2.57%,+0.86%] 和 [-1.23%,+0.23%]，均跨零；额外的 router.weight 双载体复查区间也跨零（[-3.05%,+4.87%]）。换用 Tiny Shakespeare 自然文本的 128 状态复查，expert-output 区间为 [-1.26%,+0.58%]，router.weight 区间为 [-1.32%,+0.55%]，仍跨零。因此这是跨语料可复现但方向不稳定的高方差候选，不晋升为新的自然 bias 组。机器记录见 `results/property/new_problem_group_search_v1/granite_router_gate_materialization_natural_16_20260920.json`、`granite_router_gate_materialization_natural_64_20260920.json`、`granite_router_gate_materialization_natural_128_20260920.json`、`granite_router_gate_materialization_natural_64_v2_20260920.json` 与 `granite_router_gate_materialization_shakespeare_128_20260920.json`。",
        "",
        "### 新增的索引累加候选（2026-09-18）",
        "",
        "本轮另外对 `index_add` 的重复目标累加做了一个独立的算子级探针。相同 BF16 操作数与显式输入顺序参考相比，在升序、降序和交替动态范围输入上都得到 `SYSTEMATIC_BIAS_CONFIRMED`；16 个样本的输出 RMS 约为 2.18%--2.46%，留出方向端点同向。把输入顺序反转后，aligned 符号也反转。相同探针的 backward 梯度端点为零，而输出端点仍确认，说明该探针中的差异首先出现在前向累加。",
        "",
        "用 FP32 显式顺序累加和分组求和作为来源对照时二者相等；BF16 顺序累加相对该高精度对照产生额外误差，且升序/降序的大小不同。因此这里闭合的是‘有限精度下重复目标的非结合累加与实现归约顺序’这一算子级来源，不是一个新的数值机制，也不是自然 LLM 训练总体的 bias 或 loss 结果。FP32 candidate/reference 探针在当前样本量下仅为 `INCONCLUSIVE`，所以不能把该结论改写成‘FP32 普遍有 bias’。",
        "",
        f"按本文的自然训练问题组计数，活动组现为 {payload['summary']['active_case_count']}；按‘可独立接入的算子级问题’计数，另有 1 个条件性 `indexed_accumulation_reduction_order` 组。它仍暂不加入自然训练活动表，直到有真实模型边界、参数写入和预先声明的训练状态分布确认。可复现脚本与结果见 `scripts/run_indexed_accumulation_bias_probe.py` 和 `results/property/new_problem_group_search_v1/indexed_accumulation_root_cause_summary_20260918.json`。",
        "",
        "同一轮对卷积、通用 Softplus 和通用 LayerNorm 做了受控筛查：卷积只出现约 0.4%--0.7% 的高方差输出差异且方向区间跨零，Softplus 在所测输入上前向/反向逐位一致，LayerNorm 的 BF16 顺序差异也未形成确认方向。因此这些筛查不新增问题组，不能把‘有非零 RMS’直接升级成 bias。",
        "",
        "## 使用边界",
        "",
        "所有固定集合结果只覆盖各自声明的状态、参数和实现边界。‘源已闭合’不自动表示自然总体 bias 已闭合；‘训练轨迹不相同’不自动表示质量持续恶化。后续若要把开放分支升级，必须新增能区分表中竞争解释的观测，而不是从已有聚合量反推。",
        "",
        "逐问题组的离线耗尽审计见 `results/property/root_cause_closure_v1/exhaustion_audit.json`；其中每个问题组都明确列出当前可推出的结论和仍缺少的区分性观测。",
        "",
        "## 全量来源记录审计",
        "",
        "当前账本对仓库保留的 866 条来源记录做自动归属：551 条有效测量位置、301 条历史矩阵记录、8 条旧案例复审和 6 条角色记录。它们互有重叠，不能相加成独立问题数。历史长程审计另有 "
        f"{legacy['unique_matrix_case_count']} 个主矩阵 ID、{legacy['case_count']} 行和 {legacy['final_case_count']} 行旧 bias+loss 标签；这些也是历史协议标签。只有具备独立同输入来源干预的记录才进入上面的根因问题组。完整归属在机器字段 `source_record_inventory`。",
        "",
        "机器结果：`results/property/case_causal_audit_v1/root_cause_closure_current.json`。",
        "",
        "## 冻结 benchmark 与算子族的逐项根因边界",
        "",
        f"冻结 benchmark 共 {payload['generalization_benchmark_frontier']['case_count']} 个案例；这些案例均有测量结果，但当前没有任何一个仅凭 benchmark 的 AOT endpoint 替换就升级为新的根因闭环。逐项机器记录见 `generalization_benchmark_frontier`。",
        "",
        "| benchmark 案例 | 模型 | 算子族 | update 结果 | 根因状态 |",
        "|---|---|---|---|---|",
    ]
    for case in payload["generalization_benchmark_frontier"]["cases"]:
        lines.append(
            f"| `{case['case_id']}` | {case['model']} | {case['family']} | "
            f"{case['primary_update_result']} | {case['root_cause_status']} |"
        )
    lines += [
        "",
        f"算子目录共 {payload['operator_family_frontier']['family_count']} 个族、目录记录约 {payload['operator_family_frontier']['catalogue_position_count']} 个位置。下表只给出当前根因证据等级，不把目录族数量当成独立 bias 数：",
        "",
        "| 算子族 | 目录位置 | 当前根因证据 |",
        "|---|---:|---|",
    ]
    for family in payload["operator_family_frontier"]["families"]:
        lines.append(f"| `{family['family_id']}` | {family['classified_positions']} | {family['root_cause_status']}；{family['interpretation']} |")
    measurement_only = [
        family["family_id"] for family in payload["operator_family_frontier"]["families"]
        if family["root_cause_status"].startswith("MEASUREMENT_ONLY")
    ]
    lines += [
        "",
        "仅有覆盖或端点测量、尚无该家族独立来源隔离的目录类别为："
        + "、".join(f"`{name}`" for name in measurement_only)
        + "。类别名来自目录，不等于已经出现一个待解释的自然 bias 阳性；需要先绑定具体实现、同输入参考和单变量干预。",
        "",
        "## 覆盖集合（不计入科学问题组）",
        "",
        "原审计中的 99 个同输入 SiLU/RMS 位置和 31 个 reference-graph 区域保留为覆盖集合。它们证明统一流程可以运行，但没有逐项完成根因隔离，因此不计入上表的独立问题组数量。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT_JSON), "groups": len(rows), "end_to_end": payload["summary"]["end_to_end_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
