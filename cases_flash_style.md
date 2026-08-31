# Flash-style case 机制报告

> Historical evidence report. Its 7/48 invocation counts are not the current
> long-horizon headline. Current labels come from `docs/current_mainline.md`
> and the machine audit under `results/property/declared_persistent_4096/`:
> 43 rows currently have long-run bias evidence plus paired loss separation,
> while only four have explicit late-window confirmation. This file must not
> override those labels or treat every outcome-relevant candidate as final.

更新时间：2026-08-17。

本文只整理已有证据，不提出共同 property。参照方法是 Qiu 和 Yao 的
[Flash Attention 低精度训练分析](https://arxiv.org/abs/2510.04212)。论文先复现训练失败，
再定位到具体 forward 数值误差，推导其进入 backward 和权重梯度的路径，识别有偏舍入与
相似低秩更新方向，最后用数学等价的定向修改完成长训练稳定化验证。

## 1. 统一证据链

本文用同一组门检查所有项目案例：

| 门 | 必须回答的问题 |
|---|---|
| M0 现象 | 自然输入和未修改模型上是否出现稳定、可重复的有符号误差，并明确它属于 precision contrast 还是 same-dtype optimization contrast？ |
| M1 F+B | 是否把一个具体 forward 与其实际 backward、保存张量、cotangent 和输出边完整绑定，并给出解析公式？ |
| M2 边界定位 | 是否通过逐层替换把误差缩小到一个 exact endpoint 或闭合 semantic region？ |
| M3 算术根因 | 是否解释了该边界内部是哪种舍入、累加、重构、融合或 schedule 产生误差？ |
| M4 因果干预 | 只修复该原因时效应是否消失，matched sham 是否为零，其他量是否保持不变？ |
| M5 carrier | 误差是否到达真实参数梯度，并在固定参数方向上具有 coherent carrier？ |
| M6 累积 | candidate/repair 的成对权重轨迹是否沿该方向持续分叉？ |
| M7 训练后果 | 定向修改是否在足够长的完整训练中修复 loss、谱范数或训练稳定性？ |

Flash Attention 论文完成了 M0--M7。本项目目前最强的案例完成到 M6；没有项目案例完成
论文的 M7 长训练失稳与稳定化。因此下文的“strict Flash-style”只表示项目内 M0--M6
标准，不等同于论文完整结论。

## 2. 计数边界

当前机器结果有三种不同计数，不能混用：

- 7 个此前保留的项目内 strict Flash-style 案例；
- 41 个后来通过 exact endpoint T1--T4 的实例；
- 二者合计 48 个 invocation/endpoint 实例，尚未按物理机制去重；
- 另有 16 个 endpoint 通过 T3、但在 T4 被拒绝，不是 strict case；
- property population 中的 57 个 T3 positive 正好是 41 个 T4 pass 加 16 个 T4 reject，
  不能称为 57 个完整案例。

权威快照是
`results/coverage/endpoint_case_rescreen.json.gz`。该快照仍有未完成 endpoint，因此 48 是
当前下界，不是最终全覆盖后的案例总数。

7 个详细案例的根因状态可先概括为：

| Case | 当前最深原因 | M3 算术根因 | 跨状态具体机制 | M7 |
|---|---|---|---|---|
| Qwen seq128 `lm_head dX` | BF16 GEMM reduction/arithmetic | 部分闭合 | 失败 | 未做 |
| Liger fused CE `dW` | chunk geometry × BF16 accumulator | 闭合 | 通过 | 未做 |
| Phi-4 seq64 `lm_head dX` | MM kernel arithmetic；rounding 非 coherent | 闭合到 kernel arithmetic | 通过 | 未做 |
| Mamba seq64 `in_proj` | MM kernel difference + output rounding | 部分闭合 | 失败 | 未做 |
| Qwen seq64 `v_proj` | local MM accumulation 非零，但完整 source 未分解 | 部分闭合 | 失败 | 未做 |
| Qwen layer-27 softmax | reconstructed-P → `dS` | 闭合到 semantic region | 失败 | 未做 |
| Qwen layer-23 `q_proj` | attention-backward state \(S_{bwd}\) | 闭合到 semantic region | 不作为单-kernel检验 | 未做 |

## 3. 文献锚点：Flash Attention

### 数学单元

对于

\[
S=\alpha QK^T,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
\]

backward 使用

\[
\delta=\operatorname{rowsum}(dO\circ O),
\qquad dS=\alpha P\circ(dP-\delta),
\qquad dQ=dS K.
\]

因此 query 权重梯度误差可写成

\[
dW^Q_{hp}-dW^Q_{lp}
=\alpha\sum_T(\delta_{lp}-\delta_{hp})[T]
  (PK)[T]^T X[T].
\]

### 根因和特征

- source：BF16 计算 \(\bar O=\bar P V\) 时的有偏加法舍入；
- 触发结构：多个 softmax 最大值使若干 \(\bar P\) 精确等于 1，问题特征上的
  \(V\) 多为同号值；
- carrier：不同 token 和训练步的 \((PK)[T]^TX[T]\) 具有相似低秩方向；
- accumulation：有偏系数持续乘到相似低秩矩阵上，权重谱范数和激活异常增长；
- repair：利用 softmax shift invariance 改变 row maximum，使 \(\bar P<1\)，不改变
  实数数学语义；
- validation：修改后完成长训练稳定化。

这是唯一完成 M0--M7 的锚点，不计入本项目发现数。

## 4. 已保留的 7 个项目 strict cases

### 4.1 Qwen3-1.7B seq128 `lm_head` input VJP MM

**F+B。** 对 \(Y=XW^T\)，实际 backward 为

\[
dX=QW,\qquad dW=Q^TX.
\]

误差位于实际 `dX` GEMM，不是根据名称或 shape 猜测的边。forward、保存的 \(W\)、真实
cotangent \(Q\) 和实际 backward 都已绑定。

**原因。** BF16 GEMM reduction/arithmetic path 产生有方向的 `dX` 误差。关闭 BF16
reduced-precision reduction 可去除约 91.05% 的局部 RMS residual；剩余部分与 FMA、
reduction tree 和 accumulation order 一致，但尚未进一步归到单条硬件算术事件。

**carrier 和累积。** analytic-VJP repair 与 matched control 已完成；误差到达真实参数梯度，
成对 32-step FP32-master 轨迹发生权重分叉。它通过项目内 M0--M6，但跨无关自然状态的
完整 carrier bootstrap 区间跨零。因此它是 trajectory-local case，不是跨状态 property
positive，也没有 M7 长训练结论。

证据：`results/final/precision.json.gz`、`results/final/trajectory.json.gz`、
`results/coverage/lmhead_t3_confirmation.json`。

### 4.2 Liger fused linear cross entropy `dW`

**F+B。** 闭合区域为

\[
Z=HW^T,\quad
G=N^{-1}(\operatorname{softmax}(Z)-\operatorname{onehot}(a)),
\quad dH=GW,\quad dW=G^TH.
\]

**原因。** 在 \(T=128\) 时，Liger 将 token 分成 64 个 two-token chunks，并把 64 个
chunk 的 `dW` contribution 顺序存储、相加到 BF16 accumulator。只把 accumulator
提升到 FP32，可去除约 95.7% 的 candidate-added `dW` error，同时 loss、`dH` 和 309 个
非目标参数梯度保持 bitwise exact。

**carrier 和累积。** 误差直接进入 tied `model.embed_tokens.weight`；24 个 held-out state
的 carrier 区间为正。32-step paired trajectory 中只有 tied weight gradient 改变，FP32
master distance 从 \(8.59\times10^{-6}\) 增长到 \(2.24\times10^{-3}\)。添加数学上被
忽略的零行会改变 chunk geometry，并且只有 BF16 accumulator 保持方向性，说明根因是
`chunk geometry × BF16 accumulation`，不是额外独立案例。

该案例完成 M0--M6，且是目前根因最具体的项目案例之一；没有 M7。

证据：`archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json`、
`liger.fused_ce.certificate.json`、`liger.fused_ce.chunk.certificate.json` 和
`results/final/trajectory.json.gz`。

### 4.3 Phi-4-mini seq64 `lm_head` input-gradient MM

**F+B。** 对

\[
Y=XW^T,\qquad dX=QW,\qquad dW=Q^TX,
\]

forward root、保存的 \(X,W\)、真实 \(Q\) 和两个实际 backward edge 均已 compiler-bound。

**原因。** 相同 BF16 operands 下，coherent source 是 MM kernel arithmetic difference；
最终 BF16 output-rounding 项不 coherent。same-dtype candidate-minus-reference arm 为零。
FP32 MM 后恢复 BF16 ABI 的 repair 在所有 32 个 state 降低局部误差，loss 保持完全一致，
matched sham 的所有参数梯度完全一致。

**carrier 和累积。** immediate carrier 是 `model.norm.weight`，32-state 完整坐标置信区间
为正。只更新该参数、冻结其他参数的 32-step FP32-master SGD 轨迹单调分叉。它完成
M0--M6，也通过跨状态具体机制检验；但 T4 是 bounded one-parameter trajectory，不是
完整模型训练，因此没有 M7。

证据：`results/coverage/cases/phi4_seq64_lmhead_dx.json`。

### 4.4 Mamba-130M seq64 layer-0 `in_proj` forward MM

**F+B。** forward 为 \(Y=XW^T\)，实际 backward 为 \(dX=QW,dW=Q^TX\)，具体 invocation
已闭合。

**原因。** 完整 local decomposition 显示两个 coherent source：generated MM kernel
difference 和 deterministic BF16 output rounding。轨迹 repair 只把固定 BF16 operands
下的 MM accumulation 提升到 FP32，再恢复原 BF16 ABI；它没有修复 output rounding 或
inherited operand error。因此已证明 local MM arithmetic 是一个因果 source，但尚未把
整个 observed error 归到一个单一 schedule event。

**carrier 和累积。** repair/sham 和 layer-0 `in_proj.weight` 的 32-step AdamW paired
trajectory 通过，固定方向 projection 在 steps 1/8/16/32 持续增长。四状态跨自然输入
pilot 的 carrier 不 coherent，所以它是 trajectory-local M0--M6 case；M3 对“全部误差”
仍是 partial，也没有 M7。

证据：`results/coverage/cases/mamba_seq64_input_proj.json`、
`mamba_seq64_input_proj_repair_pilot.json`、`mamba_seq64_input_proj_trajectory.json`。

### 4.5 Qwen3-1.7B seq64 layer-0 `v_proj` forward MM

**F+B。** 同样是

\[
Y=XW^T,\qquad dX=QW,qquad dW=Q^TX.
\]

**原因。** same-dtype optimization contrast 为零，precision contrast 有稳定方向。局部
FP32-MM-plus-BF16-ABI repair 只改变少量输出坐标，说明 local accumulation 确实有非零
因果效应，但四状态 carrier 区间跨零；当前证据没有把完整 T1 error 分解为 kernel、
output rounding 和 inherited operands。因此 M3 只能写 partial。

**carrier 和累积。** 同一 repair 在 32-step paired trajectory 上使
`model.layers.0.self_attn.v_proj.weight` 的固定方向 projection 从 0.003999 增长到
0.004595。它通过项目内 M0--M6，但不通过跨状态机制检验，没有 M7。

证据：`results/coverage/cases/qwen64_vproj.json`、
`qwen64_vproj_repair_pilot.json`、`qwen64_vproj_trajectory.json`。

### 4.6 Qwen3-1.7B seq128 layer-27 softmax saved-state region

**F+B。** 对每个 softmax row，

\[
p=\operatorname{softmax}(a),\qquad
d a=p\circ(q-\langle p,q\rangle).
\]

实际 generated backward 使用由 BF16 logits/max/sum 重构的 probability，而不是 typed
true-forward FP32 probability。

**原因。** 因果 source 已定位到 `saved/reconstructed P -> dS` semantic boundary。只在
exact generated `dS` boundary 用 true-forward-P analytic VJP 替换重构概率，并恢复原
BF16 `dS` ABI，forward loss 不变，matched sham 为零，真实 q/k VJP 随之改变。当前不能
再归因到一条唯一 Triton instruction。

**carrier 和累积。** 只更新 layer-27 q/k projection weights 的 32-step trajectory 中，
固定方向 projection 在 steps 1/8/16/32 为 0.004468、0.005082、0.005168、0.005250。
它是 closed semantic-region M0--M6 case；跨无关 state 的 semantic error 不 coherent，
没有 M7。

证据：`results/coverage/cases/qwen128_softmax_fb.json`、
`qwen128_softmax_fb_formal.json`、`qwen128_softmax_saved_p_trajectory.json`。

### 4.7 Qwen3-1.7B layer-23 `q_proj` attention-state region

**F+B。** 闭合关系为

\[
S_{bwd}=\alpha J_{softmax}(P)^T(DV^T),\qquad
G_q=S_{bwd}K,
\]

\[
Y=HW^T,qquad dW=G_q^TH.
\]

对应误差分解为

\[
\Delta dW=(\Delta G)^TH_{ref}+G_{ref}^T\Delta H+(\Delta G)^T\Delta H.
\]

**原因。** final same-input q-projection GEMM 不是 source。恢复 `bmm_76` 左输入
\(S_{bwd}\) 单独即可关闭方向；只恢复 \(K\) 不行。保守的 joint `S_bwd/K` repair 也通过，
而额外的 \(K\) contribution 不具有方向性。因而 causal root 是 attention-backward state
\(S_{bwd}\) semantic region；更深的 forward/backward contributors 有重叠，目前不能拆成
一个独立 kernel bug。

**carrier 和累积。** 真实 carrier 是 layer-23 `q_proj.weight` 的指定 tile，repair/sham、
32-state carrier 和 32-step AdamW trajectory 均通过。它完成 semantic-region M0--M6，
但 M3 没有下钻到唯一算术指令，也没有 M7。

证据：`results/coverage/cases/l23_qproj_attention_state_region.json`。

## 5. 41 个新增 strict endpoint 实例

这些实例都满足：完整 concrete F+B、full-coordinate T1、exact endpoint reference
replacement/sham T2、真实完整 carrier T3、paired 32-step T4。它们证明“该 exact endpoint
的 candidate/reference 差异会因果性地进入并累积到指定参数”。

但 41/41 的 `single_kernel_root_attribution` 都是 `false`。T2 把 endpoint 输出替换成
reference value，只定位了边界，没有回答边界内部是舍入、reassociation、reduction tree、
融合物化还是近似函数造成误差。因此它们是 M0--M2、M4--M6 闭合的 endpoint cases，M3
全部未闭合，不能按 Flash Attention 论文规范宣称已经找到算术根因。

### 5.1 DeepSeek forward `add`，25 个实例

数学根为 \(Y=A+B\)，VJP 为 broadcasting-aware `sum_to_shape(q)`。25 个 endpoint 都在
attention generated region，carrier 都是对应层的 `q_proj.weight`。共同现象是 same-dtype
compiled endpoint 相对 AOT reference 的方向差异；具体 add 内部根因未知。

```text
seq64:  forward:47,299,317,371,389,407,443,461,515,533,551,587,605,623
seq128: forward:317,443,515,551,569
seq256: forward:83,317,461,551,569,605
```

精确 candidate IDs 以前缀 `deepseek8b:seq*:forward:*:in_out_ptr0` 组成；对应 exact roots
依次是 `add_26/add_166/.../add_346`，完整逐项绑定保存在 endpoint re-screen artifact。

### 5.2 DeepSeek forward `bmm`，7 个实例

数学根为

\[
Y[b]=A[b]B[b],\quad dA[b]=q[b]B[b]^T,\quad dB[b]=A[b]^Tq[b].
\]

carrier 均为对应层 `o_proj.weight`。内部 BMM arithmetic/reduction 原因尚未分解。

```text
deepseek8b:seq64:forward:460:output_0
deepseek8b:seq64:forward:550:output_0
deepseek8b:seq64:forward:568:output_0
deepseek8b:seq64:forward:586:output_0
deepseek8b:seq64:forward:622:output_0
deepseek8b:seq256:forward:460:output_0
deepseek8b:seq256:forward:550:output_0
```

### 5.3 Qwen forward `rsqrt`，4 个实例

数学根为

\[
Y=X^{-1/2},\qquad dX=-\tfrac12 qY^3.
\]

endpoint replacement 已证明因果性，但 fused reduction、epsilon/add、`rsqrt`
approximation 和 output materialization 尚未拆开。

| Candidate | Carrier |
|---|---|
| `qwen:seq64:forward:232:in_out_ptr1` | layer-12 `o_proj.weight` |
| `qwen:seq64:forward:273:in_out_ptr1` | layer-14 `down_proj.weight` |
| `qwen:seq256:forward:133:in_out_ptr0` | layer-7 `k_norm.weight` |
| `qwen:seq256:forward:322:in_out_ptr1` | layer-17 `o_proj.weight` |

### 5.4 Qwen `mm`，3 个实例

数学根为 \(Y=AB,dA=qB^T,dB=A^Tq\)。具体 MM reduction/accumulation/output-cast
分解尚未做。

| Candidate | Phase | Carrier |
|---|---|---|
| `qwen:seq64:backward:553:output_0` | backward | layer-26 `down_proj.weight` |
| `qwen:seq256:backward:552:output_0` | backward | layer-26 `down_proj.weight` |
| `qwen:seq256:forward:483:output_0` | forward | layer-26 `o_proj.weight` |

### 5.5 Qwen backward `sum`，1 个实例

`qwen:seq256:backward:581:out_ptr0` 对应 `sum_51`，carrier 是 layer-26
`k_norm.weight`。解析 VJP 是把 cotangent 沿被归约维 expand 回输入 shape。具体 reduction
tree 和累加精度尚未分解。

### 5.6 Phi-4 forward `rsqrt`，1 个实例

`phi4:seq128:forward:108:in_out_ptr1` 对应 `forward_g4__rsqrt_13`，carrier 是 layer-6
`o_proj.weight`。endpoint 因果链通过，`rsqrt` region 内部算术根因未知。

以上 41 个 endpoint 被压缩为 9 个 model/phase/root/carrier recurrence patterns，仅用于阅读；
9 也不是已经证明的独立物理机制数。

### 5.7 完整 endpoint 明细

下表逐项保留全部 41 个实例。这里的 T1 区间是基于完整坐标误差向量得到的 cross-state
U-statistic cluster-bootstrap 95% 区间；数值尺度随 endpoint tensor 不同，不能跨行直接
比较大小。

| Candidate | Exact AOT endpoint | Carrier | T1 95% interval |
|---|---|---|---:|
| `qwen:seq64:forward:232:in_out_ptr1` | `forward:graph0:rsqrt_51` | layer-12 `o_proj.weight` | [2.74e-08, 1.18e-07] |
| `qwen:seq64:forward:273:in_out_ptr1` | `forward:graph0:rsqrt_60` | layer-14 `down_proj.weight` | [1.04e-08, 4.70e-08] |
| `qwen:seq64:backward:553:output_0` | `backward:graph0:mm_213` | layer-26 `down_proj.weight` | [3.28e-06, 7.16e-05] |
| `qwen:seq256:forward:133:in_out_ptr0` | `forward:graph0:rsqrt_30` | layer-7 `k_norm.weight` | [1.33e-07, 3.50e-07] |
| `qwen:seq256:forward:322:in_out_ptr1` | `forward:graph0:rsqrt_71` | layer-17 `o_proj.weight` | [4.29e-10, 8.57e-09] |
| `qwen:seq256:forward:483:output_0` | `forward:graph0:mm_185` | layer-26 `o_proj.weight` | [729, 1.50e+03] |
| `qwen:seq256:backward:552:output_0` | `backward:graph0:mm_213` | layer-26 `down_proj.weight` | [1.64e-05, 9.95e-05] |
| `qwen:seq256:backward:581:out_ptr0` | `backward:graph0:sum_51` | layer-26 `k_norm.weight` | [4.60e-09, 1.86e-08] |
| `phi4:seq128:forward:108:in_out_ptr1` | `forward:graph0:forward_g4__rsqrt_13` | layer-6 `o_proj.weight` | [3.77e-07, 1.51e-06] |
| `deepseek8b:seq64:forward:47:in_out_ptr0` | `forward:graph0:add_26` | layer-2 `q_proj.weight` | [28.8, 35.2] |
| `deepseek8b:seq64:forward:299:in_out_ptr0` | `forward:graph0:add_166` | layer-16 `q_proj.weight` | [11.5, 20.0] |
| `deepseek8b:seq64:forward:317:in_out_ptr0` | `forward:graph0:add_176` | layer-17 `q_proj.weight` | [32.4, 49.6] |
| `deepseek8b:seq64:forward:371:in_out_ptr0` | `forward:graph0:add_206` | layer-20 `q_proj.weight` | [19.9, 40.2] |
| `deepseek8b:seq64:forward:389:in_out_ptr0` | `forward:graph0:add_216` | layer-21 `q_proj.weight` | [10.9, 26.8] |
| `deepseek8b:seq64:forward:407:in_out_ptr0` | `forward:graph0:add_226` | layer-22 `q_proj.weight` | [23.0, 38.1] |
| `deepseek8b:seq64:forward:443:in_out_ptr0` | `forward:graph0:add_246` | layer-24 `q_proj.weight` | [21.3, 49.4] |
| `deepseek8b:seq64:forward:460:output_0` | `forward:graph0:bmm_50` | layer-25 `o_proj.weight` | [7.31e+03, 1.24e+04] |
| `deepseek8b:seq64:forward:461:in_out_ptr0` | `forward:graph0:add_256` | layer-25 `q_proj.weight` | [39.7, 64.7] |
| `deepseek8b:seq64:forward:515:in_out_ptr0` | `forward:graph0:add_286` | layer-28 `q_proj.weight` | [23.6, 43.6] |
| `deepseek8b:seq64:forward:533:in_out_ptr0` | `forward:graph0:add_296` | layer-29 `q_proj.weight` | [30.0, 57.6] |
| `deepseek8b:seq64:forward:550:output_0` | `forward:graph0:bmm_60` | layer-30 `o_proj.weight` | [9.60e+03, 1.43e+04] |
| `deepseek8b:seq64:forward:551:in_out_ptr0` | `forward:graph0:add_306` | layer-30 `q_proj.weight` | [43.1, 65.8] |
| `deepseek8b:seq64:forward:568:output_0` | `forward:graph0:bmm_62` | layer-31 `o_proj.weight` | [7.50e+03, 1.39e+04] |
| `deepseek8b:seq64:forward:586:output_0` | `forward:graph0:bmm_64` | layer-32 `o_proj.weight` | [7.03e+03, 1.19e+04] |
| `deepseek8b:seq64:forward:587:in_out_ptr0` | `forward:graph0:add_326` | layer-32 `q_proj.weight` | [36.7, 60.0] |
| `deepseek8b:seq64:forward:605:in_out_ptr0` | `forward:graph0:add_336` | layer-33 `q_proj.weight` | [22.3, 57.4] |
| `deepseek8b:seq64:forward:622:output_0` | `forward:graph0:bmm_68` | layer-34 `o_proj.weight` | [4.90e+03, 8.42e+03] |
| `deepseek8b:seq64:forward:623:in_out_ptr0` | `forward:graph0:add_346` | layer-34 `q_proj.weight` | [23.9, 40.6] |
| `deepseek8b:seq128:forward:317:in_out_ptr0` | `forward:graph0:add_176` | layer-17 `q_proj.weight` | [121, 202] |
| `deepseek8b:seq128:forward:443:in_out_ptr0` | `forward:graph0:add_246` | layer-24 `q_proj.weight` | [79.7, 173] |
| `deepseek8b:seq128:forward:515:in_out_ptr0` | `forward:graph0:add_286` | layer-28 `q_proj.weight` | [59.6, 110] |
| `deepseek8b:seq128:forward:551:in_out_ptr0` | `forward:graph0:add_306` | layer-30 `q_proj.weight` | [116, 185] |
| `deepseek8b:seq128:forward:569:in_out_ptr0` | `forward:graph0:add_316` | layer-31 `q_proj.weight` | [111, 181] |
| `deepseek8b:seq256:forward:83:in_out_ptr0` | `forward:graph0:add_46` | layer-4 `q_proj.weight` | [160, 211] |
| `deepseek8b:seq256:forward:317:in_out_ptr0` | `forward:graph0:add_176` | layer-17 `q_proj.weight` | [531, 847] |
| `deepseek8b:seq256:forward:460:output_0` | `forward:graph0:bmm_50` | layer-25 `o_proj.weight` | [8.21e+04, 1.46e+05] |
| `deepseek8b:seq256:forward:461:in_out_ptr0` | `forward:graph0:add_256` | layer-25 `q_proj.weight` | [428, 734] |
| `deepseek8b:seq256:forward:550:output_0` | `forward:graph0:bmm_60` | layer-30 `o_proj.weight` | [1.28e+05, 1.82e+05] |
| `deepseek8b:seq256:forward:551:in_out_ptr0` | `forward:graph0:add_306` | layer-30 `q_proj.weight` | [582, 815] |
| `deepseek8b:seq256:forward:569:in_out_ptr0` | `forward:graph0:add_316` | layer-31 `q_proj.weight` | [460, 702] |
| `deepseek8b:seq256:forward:605:in_out_ptr0` | `forward:graph0:add_336` | layer-33 `q_proj.weight` | [299, 580] |

## 6. 有误差但不能计为 strict case 的结果

### 6.1 DeepSeek seq128 attention softmax

`vl-fb-0881` 已完成 softmax forward 与实际 backward 的数学和 compiler binding。forward
保存的 scaled/masked logits 与 backward `dS` 都有正的 32-state directional interval。
但尚未证明 forward saved-logit error 导致 backward error；第一次 repair 因 FP32 storage
被错误传入冻结的 `*bf16` ABI 而 fail closed。有效 T2、真实 carrier 和 T4 都缺失。

结论：完整 local F+B directional candidate，不是 strict Flash-style case。

证据：`results/coverage/cases/deepseek128_attention_softmax_fb.md`。

### 6.2 Qwen seq128 layer-0 `v_proj`

完整 decomposition 表明，主要 coherent T1 source 是 deterministic FP32-to-BF16 output
rounding；固定 operands 下的 generated-MM kernel difference 不 coherent。局部 MM repair
因而不是主要 source，32-step 固定方向 projection 在 step 16 后于 step 32 回落。

结论：已解释的 precision source，但 M6 失败，不是 strict case。

证据：`results/coverage/cases/qwen128_vproj.json`、
`qwen128_vproj_precision_decomposition.json`、`qwen128_vproj_trajectory.json`。

### 6.3 Qwen3-VL SiLU backward decomposition

一个相同 SiLU forward 下的 backward-only intervention 可复现 AOT decomposition 与 eager
`aten.silu_backward` 的梯度差异，F+B 和因果数值差异成立。但全参数误差在六个自然 state
上的平均 pairwise inner product 为负，32-step frozen-direction projection 也不增长。

结论：真实 numerical-difference case，但 M5/M6 不满足，不是 directional-bias case。

证据：`round2.md`、`results/coverage/cases/qwen3vl_layer0_silu_trajectory.json`。

### 6.4 16 个 T3-pass/T4-reject endpoint

这些 endpoint 已有 coherent parameter carrier，但 paired trajectory 不满足预先定义的
directional accumulation，所以不能作为 strict cases：

```text
qwen:seq64:forward:295:in_out_ptr0
qwen:seq128:forward:5:out_ptr2
qwen:seq256:forward:115:in_out_ptr0
qwen:seq256:forward:295:in_out_ptr0
qwen:seq256:backward:589:out_ptr0
phi4:seq256:forward:18:in_out_ptr0
phi4:seq256:forward:22:in_out_ptr1
phi4:seq256:forward:93:in_out_ptr1
deepseek8b:seq64:forward:619:in_out_ptr0
deepseek8b:seq64:backward:1854:out_ptr0
deepseek8b:seq128:forward:115:in_out_ptr1
deepseek8b:seq128:forward:117:out_ptr0
deepseek8b:seq128:forward:160:in_out_ptr1
deepseek8b:seq128:forward:619:in_out_ptr0
deepseek8b:seq256:forward:619:in_out_ptr0
deepseek8b:seq256:backward:871:output_0
```

## 7. 当前可以安全得出的结论

1. 项目已经有 7 个详细复核的 trajectory-local/semantic-region strict cases，以及 41 个
   新的 strict endpoint instances。
2. 只有 Liger、Phi MM 等少数案例把误差进一步拆到了相对具体的 physical arithmetic
   source；部分 MM 案例只完成了 source class，尚未完成 instruction/schedule-level 根因。
3. 两个 attention semantic-region case 已证明 causal boundary，但没有唯一 kernel bug。
4. 41 个新增 endpoint 证明了真实、可累积的 candidate/reference 差异，尚未解释差异为何
   在 endpoint 内产生；不能把 `add`、`rsqrt`、`bmm`、`mm` 或 `sum` 名称本身当成原因。
5. 没有项目案例完成 Flash Attention 论文式 M7 长训练 loss/pathology repair，因此目前不应
   使用“完全复现论文级训练失败机制”这一表述。
6. 在完成每个 case 的 M3 source decomposition 之前，不进行共同 property 归纳。

## 8. 机器证据索引

- 7 个保留案例复核：`results/coverage/existing_case_reaudit.json`
- 41 个 strict endpoint 和完整分母：`results/coverage/endpoint_case_rescreen.json.gz`
- 1,562 个具体 F+B 数学证明：
  `results/coverage/cases/directional_candidate_math_registry.json.gz`
- T3 property population：`results/property/hypothesis_matrix.json`
- 原有长说明：`case.md`
