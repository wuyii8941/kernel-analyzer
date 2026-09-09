# Liger：把同一套训练的误差来源、更新和 loss 接起来

本轮复用已完成的 16 组 WikiText / 真实 tokenizer 训练，不换模型、不用 Qwen
的短程结果代替这里的小型 GPT-2，也不重新调整判断阈值。属于看到训练确认结果后的
机制检查，不是新的未见实现确认。讲稿未修改。

## 现在补到了哪一步

| 检查 | 实际结果 | 能说明什么 |
|---|---|---|
| 累加差异来源 | 一个真实末期状态的 128 个分块乘积完全一致；逐次舍入重构误差为零 | 这次 `dW` 差异来自累加及最终转换，而不是两个设置算出了不同的分块乘积 |
| 末期直接更新 | 16 组两条轨迹共 32 个 checkpoint，全部有非零直接更新差异 | 第 1024 步时仍有实现的直接作用，不是只剩此前分叉的反馈 |
| 跨输入方向 | 每个 checkpoint 的前后两半输入，平均更新差异的内积均为正 | 固定末期状态下，方向在另一半输入中仍同向；不是全程持续性检验 |
| 原先冻结的 loss 确认 | 16 组中 15 组参考 loss 更低，平均差 +0.00069609 | 声明设置下存在可重复的小幅训练后果 |

末期检查恢复真实 FP32 master 参数和 AdamW moments。每个 checkpoint 固定不变，
分别运行 32 批输入，两种实现共享每批输入与起始状态。只测声明的 tied embedding /
language-head 参数，不能扩大为全参数等价结论。

直接更新差异 RMS 是正常更新 RMS 的 **0.0910%–0.1025%**；平均分量约为
**0.0170%–0.0193%**。这些量直接由完整坐标累加得到，不是随机摘要的上界。
固定集合的均值非零不自动成为任意训练状态总体的数学保证。

## 数学核对具体证明了什么

同一批真实分块乘积记为 `p_j`。两个累加结果分别为 `s_j`（BF16）和 `f_j`（FP32）。
定义各步真实舍入误差：

```
e_BF16,j = s_j - s_(j-1) - p_j
e_FP32,j = f_j - f_(j-1) - p_j
```

因此，最终两种梯度之差等于各步舍入误差之差的总和，再扣除参考最后一次转回
BF16 的误差。这是逐项相消得到的恒等式。本次还独立核对了实际 Liger 返回的
两种 `dW` 和重构值完全相同，输入梯度完全相同。

该检查针对独立的 language-head 梯度；完整训练中 tied embedding 还会接收输入侧
梯度，末期更新实验使用的是实际合并后的梯度。不能把二者无说明地当成同一个量。

**这条恒等式解释已观察差异的来源，不证明舍入误差期望必定非零。** 要声明持续
bias，还需说明状态分布或检查真实训练过程中多个阶段的直接作用。末期方向与 loss
同时存在，也不证明该方向分量足以解释全部 loss 差异。

## 自动运行与原始记录

### 后续：在真实推进的训练状态中检查

上述固定 checkpoint 检查完成后，全部 16 组配对训练从原 1024 步续至 4096 步，
不重置参数或 AdamW。直接检查窗口事前固定为 1025–1056、2017–2048、
3041–3072、4065–4096；每步只把自然训练梯度写入，参考重放不改变训练状态。

32 条轨迹的后三个窗口平均直接差异均与第一个窗口同向；30 条在四个窗口内
前后两半均值也同向。第 13 组的两条轨迹在 3041–3072 的内部检查例外，完整保留。
最后窗口的直接差异 RMS 为正常更新的 0.1502%–0.1786%。

4096 步时 16 组 loss 均不同，但有 14 组原实现更低，平均差 −0.00132849。
因此不能把早期的正 loss 差外推成持续恶化。当前支持同一实现比较中的舍入来源、
已测后期直接方向与 loss 分叉共同存在，不证明每一步同向或方向分量解释全部 loss。

自动采集和核验入口分别为 `scripts/continue_liger_language_temporal.py`、
`scripts/summarize_liger_temporal_extension.py`。
[全部 32 条轨迹及四窗口结果](../results/property/training_numerical_analysis_v2/language_temporal_extension/summary.json)。

### 前述固定 checkpoint 检查

采集入口：`scripts/probe_liger_language_checkpoints.py`；结果汇总：
`scripts/summarize_liger_checkpoint_probe.py`；舍入恒等式核对：
`scripts/check_liger_accumulation_identity.py`。协议先保存，输出不覆盖。

- [32 个 checkpoint 的检查结果](../results/property/training_numerical_analysis_v2/language_checkpoint_direct_probe/summary.json)
- [真实分块累加的恒等式核对](../results/property/training_numerical_analysis_v2/language_accumulation_identity_retry/result.json)
- [独立初始化 loss 确认](../results/property/training_numerical_analysis_v2/language_training_confirmation_iid/summary.json)

首次源码核验在运行前遇到路径类型错误，未产生测量；失败记录保留，重试单独保存。
