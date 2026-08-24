# 当前数据对 optimizer 的结论

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

Qwen 在 gradient 层存在一定方向性，但 AdamW 后直接更新接近抵消。

## 自然训练阶段

Qwen 在 step 0、8、16 分别使用各自真实权重、输入和 moments 测量：

| 阶段 | 梯度差异 A | captured AdamW A | 重置 moments A |
|---|---:|---:|---:|
| step 0 | 1.0075 | 1.0000 | 1.0000 |
| step 8 | 1.0168 | 1.0092 | 1.0001 |
| step 16 | 0.9976 | 0.9942 | 1.0002 |

三个阶段都接近抵消。这只是同状态响应检查，不是新的 32 步 live trajectory，也不能外推成所有模型的 optimizer 规律。

## Gemma 状态反馈对照

Gemma 的同状态梯度和无状态 SGD 都约为 `A=1.019`，captured AdamW 为 `0.9995`，每步重置 moments 为 `1.0001`。它不是直接持续正例。另一个 live trajectory 的最终分离主要由训练状态反馈维持，这说明最终轨迹分开不能替代同状态直接作用检查。

## 两个 Phi 数字不能混用

- `A=3.325 -> 0.956`：16 个共同状态、无 moments、无状态 SGD 下的随机舍入干预。
- `A=1.029`：32 步 cold-start AdamW 直接持续结果。

第一项不能解释或修复第二项。

## 最终措辞

> optimizer 会改变数值梯度差异是否进入有效参数更新，并可能参与后续训练状态反馈；当前数据没有把 AdamW 确认为统一根因。

主要数据：

- `results/property/direct_persistence_v4/optimizer_state/liger_t128_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/phi_seq64_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/qwen_seq128_same_state_ablation.json`
- `results/property/direct_persistence_v4/optimizer_state/qwen_phase_conditioned_response.json`
- `results/property/direct_persistence_v4/optimizer_state/gemma4_feedback_same_state_ablation.json`
