# 当前数据对 optimizer 的结论

> 本页是 optimizer-stage 的详细证据，不单独定义 bias。当前统一口径先区分
> source/response formation，再比较 local、gradient 和 update。

当前证据不能说明 AdamW 是数值 bias 的原始来源。它说明 optimizer 会改变一个梯度差异最终如何进入参数更新：有时保留，有时压低。

## Phi

| 映射 | A32 |
|---|---:|
| 参数梯度差异 | 4.6827 |
| 无状态 SGD | 4.6827 |
| 使用已捕获 AdamW moments | 1.0296 |
| 每一步重置 moments | 1.0139 |

方向性在 optimizer 之前已经很强，AdamW 明显压低了它。因此 Phi 不能支持“AdamW 制造了这个 bias”。

## Liger

| 映射 | A32 |
|---|---:|
| 参数梯度差异 | 2.8382 |
| 无状态 SGD | 2.8382 |
| 使用已捕获 AdamW moments | 1.6809 |
| 每一步重置 moments | 1.8284 |

AdamW 压低但没有完全消除方向性。

## Qwen

| 映射 | A32 |
|---|---:|
| 参数梯度差异 | 1.3430 |
| 无状态 SGD | 1.3430 |
| 使用已捕获 AdamW moments | 0.9611 |
| 每一步重置 moments | 1.0000 |

Qwen 在这组 cold-start 32 步状态中，gradient 层存在一定方向性，但 AdamW 后直接更新接近抵消。这是短窗口结果，不是算子的永久标签。

## 自然训练阶段

Qwen 在 step 0、8、16 分别使用各自真实权重、输入和 moments 测量：

| 阶段 | 梯度差异 A | captured AdamW A | 重置 moments A |
|---|---:|---:|---:|
| step 0 | 1.0075 | 1.0000 | 1.0000 |
| step 8 | 1.0168 | 1.0092 | 1.0001 |
| step 16 | 0.9976 | 0.9942 | 1.0002 |

三个阶段都接近抵消。这只是同状态响应检查，不是新的 32 步 live trajectory，也不能外推成所有模型的 optimizer 规律。

## 统一的 checkpoint 与 optimizer 状态检查

后续实验固定了三个训练位置，并对每个位置使用同一套 32 个测量输入和 45 项整体
Holm 校正。这里的百分比表示 candidate 相对 repair 反复改变正常参数 update 的比例；
负数表示缩小。

| 模型与训练位置 | cold-start AdamW | warm 8 步 | warm 32 步 | warm 32 步后重置 moments | warm 32 步的 SGD |
|---|---:|---:|---:|---:|---:|
| DeepSeek seq128 attention projection | **−10.69%** | **−0.187%** | **−0.307%** | **−9.03%** | **−1.33%** |
| DeepSeek seq256 normalization | **−13.68%** | **−0.641%** | −0.547%（未确认） | **−15.13%** | +0.704%（未确认） |
| Phi loss / CE backward | +0.079%（未确认） | −0.078%（未确认） | −0.054%（未确认） | −0.216%（未确认） | −0.026%（未确认） |

这组结果把边界说得更清楚：两个 DeepSeek 位置在 moments 从零开始时有约 11%–14%
的缩小，真实推进 8 或 32 步后降到不足 1%；在同一 warm checkpoint 重置 moments，
大效应又恢复。因而它既不是只由 kernel 决定的永久标签，也不能只归因于参数值；
当前 AdamW 历史状态是效应大小的重要条件。Phi 负例在五种设置下都没有被升级。

机器结果：

- `results/property/optimizer_condition_benchmark_v1/summary.json`

## warm-state 4096 步更新

后续长程实验先 warm up 128 步，再测量 4096 步同状态 direct update。Qwen 的分数为：

```text
A32   = 1.084
A128  = 1.350
A1024 = 2.950
A4096 = 6.488
```

后半程 64/64 个窗口都有方向。因此，旧的“AdamW 把 Qwen 消掉”只能描述 cold-start 32 步；在 warm-state 长程协议中，方向重新出现并累积。新旧协议同时改变了参数、moments、输入状态和观测时长，当前不能把差异单独归因于某一个因素。

## Gemma 状态反馈对照

Gemma 的同状态梯度和无状态 SGD 都约为 `A=1.019`，captured AdamW 为 `0.9995`，每步重置 moments 为 `1.0001`。它不是直接持续正例。另一个 live trajectory 的最终分离主要由训练状态反馈维持，这说明最终轨迹分开不能替代同状态直接作用检查。

## 两个 Phi 数字不能混用

- `A=3.325 -> 0.956`：16 个共同状态、无 moments、无状态 SGD 下的随机舍入干预。
- `A=1.029`：32 步 cold-start AdamW 直接持续结果。

第一项不能解释或修复第二项。

## 最终措辞

> optimizer 和当前训练状态共同改变数值梯度差异是否进入有效参数更新。短程抑制不保证长程抵消；当前数据没有把 AdamW 确认为统一根因。

主要数据：

- `results/property/direct_persistence_v4/optimizer_state/liger_t128_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/phi_seq64_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/qwen_seq128_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/qwen_phase_conditioned_response.json`
- `results/property/direct_persistence_v4/optimizer_state/gemma4_feedback_same_state_ablation.json`
