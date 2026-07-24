# Qwen3 step236 原始 final-RMSNorm kernel-group 证据 v0.1

## 结论

本实验在未拆分的原始 whole-model compiled realization 中，对 step236
实际命中的 dynamic specialization 做了 live generated-kernel 干预。独立
audit 为 `VALID`。

当前最多可以声称：

> 原始 compiled final-residual/final-RMSNorm fused kernel group 在相同运行时
> 输入上，相对于声明的 ATen reference expression 产生了可重复局部差异；将
> reference 输出写回原预分配 buffer 后，固定不变的原 compiled suffix 产生了
> 可重复的连续 log-prob 变化。

不能声称该 kernel group 是 correctness bug、clipping fork root cause，或长期
训练差异来源。

## 分析对象与 provenance

- target specialization：dynamic sequence length `168`；
- reduction kernel：`triton_per_fused__unsafe_view_add_mean_pow_rsqrt_16`；
- intercepted pointwise kernel：
  `triton_poi_fused__to_copy__unsafe_view_add_mul_slice_18`；
- pointwise kernel provenance 同时覆盖 final residual additions、final RMSNorm
  scaling/cast 与送入 LM head 前的 slice；
- 这是 fused generated kernel group，不是唯一 source-level RMSNorm op。

选择 `_16/_18` 而不是 static-specialization 的 `_15/_17`，由实际 target state
的 specialization 与 runtime invocation 决定，不能仅凭 kernel 名称猜测。

## 可信性 gates

以下 gates 全部通过：

- 原 eager 与 compiled scorer anchors 各重复两次精确；
- 原 Dynamo graph family 精确；
- live generated kernel 从实际 Inductor artifact/runtime module 解析；
- candidate、no-op proxy、repair、restored candidate 均重复两次；
- no-op proxy 与 candidate bit-exact；
- 每次 target forward 恰好拦截一次 `_18`；
- same-input local records 跨重复完全相同；
- repair 跨重复完全相同；
- 干预期间无 backend recompile；
- restore 后 candidate anchor 精确恢复。

whole-model trace 对 forward 完全等价，但改变了 backward/update hash。因此 trace
只被授权提供 forward provenance；本结果不包含 backward/update claim。

## Production 与 mediation 分开报告

### Same-input discrepancy production

在完全相同的 norm weight 和三个 residual component buffers 上：

- compiled reduction 与 reference reduction：137 个元素不同，最大绝对差
  `7.45e-09`；
- compiled kernel-group output 与 reference expression output：27 个元素不同，
  最大绝对差 `0.00390625`，L2 `0.01040269`；
- 两次运行的输入 fingerprints、输出 hashes 和 discrepancy metrics 完全相同。

因此 production 成立，但只对这个 fused group、state、dtype/layout 与 reference
expression 成立。

### Fixed-original-suffix endpoint mediation

reference output 被写回原 `_18` output buffer，之后继续执行未替换的原 LM-head
及 scorer suffix：

- 512 个 selected-token log-prob 中 1 个发生变化；
- 最大绝对变化与 L2 均为 `2.670288e-05`；
- clipping decision 变化数为 0；
- eager--candidate scorer L2 为 `0.12844525`；repair 后为 `0.12844685`，即
  repair 轻微远离 eager，而不是解释/消除整体 discrepancy。

所以 continuous mediation 成立，semantic clipping mediation 在该 state 上不成立。

## 允许和禁止的解释

允许：

- 原始 generated fused kernel group 是一个可重复的 local discrepancy producer；
- 该局部差异能通过固定原 suffix 改变一个连续 scorer coordinate；
- 该 kernel group 对当前 clipping endpoint 没有观察到 mediation。

禁止：

- 将 effect 唯一分配给 `mean`、`rsqrt`、cast、residual add 或 RMSNorm；
- 将 repair 的非零效果称为 root cause；
- 将无 clipping 变化推广为其他 states 上恒为 null；
- 将 eager 当成数学真值；
- 推断 gradient、parameter update 或长期训练影响。

## 权威 artifacts

- manifest：`QWEN3_STEP236_ORIGINAL_FINAL_NORM_KERNEL_GROUP_V0_1.json`；
- whole-model forward inventory：
  `results/operator_oracle/qwen3_step236_whole_model_trace_v0_1/compiled_1/forward_kernel_inventory.json`；
- live result：
  `results/operator_oracle/qwen3_step236_live_final_norm_kernel_group_v0_3/result.json`；
- independent audit：
  `results/operator_oracle/qwen3_step236_live_final_norm_kernel_group_v0_3/audit.json`。

这条纵向切片证明当前仓库能够做可信的原始 generated-kernel-group 分析；下一步
应在预声明的多个 matched states 上重复同一 kernel-group production/mediation
协议，或选择另一个 provenance 完整的 group。它尚未解决 fused group 内部的唯一
source-op 归因。
