# 五个补充案例的统一结果

本轮补充了三个普通 candidate-repair 案例，以及两个严格正负响应案例。普通实现的
27 个检验和正负响应的 6 个检验分别使用 Holm 校正。

> **v2 边界：** 这是方法开发阶段的 v1 固定-suite 结果。protocol、finalizer 与结果
> 位于同一 Git commit，且 32 个状态沿连续 repair trajectory 推进，因此不能称为
> 可审计的事前总体确认。表中数值仍描述该 suite；独立 training-unit 区间与多-seed
> 大向量复核需按 [`training_bias_profile_v2.md`](training_bias_profile_v2.md) 重采。

所有结果都使用前 16 个状态发现方向、后 16 个状态确认。对于“平均方向”和
“去掉正常缩放后的方向”，后半状态必须继续沿前半状态发现的方向；显著但反向不算
复现。

## 1. 三个普通实现

| 模型与位置 | local output | parameter gradient | AdamW update | 最简单的结论 |
|---|---|---|---|---|
| Qwen `lm_head dX` | 平均方向复现 | 平均方向复现 | 平均方向复现，但很小 | 方向从 backward endpoint 保留到参数 update |
| Qwen layer-0 `v_proj` | 未复现 | 未复现 | 未复现 | 有局部数值差异，不等于存在跨状态平均方向 |
| Mamba layer-0 `in_proj` | 未复现 | 正常 gradient 平均缩小约 0.97% | 未复现 | backward 中出现缩放，AdamW 将其消除 |

### Qwen `lm_head dX`

相对同阶段正常 training signal 的平均方向效应为：

| 阶段 | 效应量 | 95% CI | 27 项 Holm 后 |
|---|---:|---:|---|
| local | 0.00112% | [0.00072%, 0.00154%] | 通过 |
| gradient | 0.00767% | [0.00344%, 0.01277%] | 通过 |
| AdamW update | 0.000400% | [0.000215%, 0.000619%] | 通过 |

这修正了旧的简单说法：“AdamW 把 Qwen 完全抵消”。旧方向分数描述总体路径是否像
随机游走；新的分半检验专门问一个更小的平均方向能否在未参与选方向的状态中复现。
当前结果是：AdamW 强烈压低了它，但没有把平均分量降到零。

### Qwen `v_proj`

local、gradient 和 AdamW update 的平均方向区间都跨零，完整 Holm 后没有一项通过。
因此这个案例是重要负例：local RMS 或单一固定状态中的方向，不能替代跨状态确认。

### Mamba `in_proj`

local 没有稳定平均方向。gradient 阶段出现相对正常 gradient 的 `-0.967%` 缩放，
95% 区间为 `[-1.786%, -0.383%]`，完整 Holm 后 `p=0.0350`。AdamW update 阶段
没有任何分量通过确认。

因此它提供了一种与 Qwen `lm_head` 不同的变化：方向结构可以在 backward 中出现，
又在 optimizer update 中消失。它也说明结果不只存在于 Transformer。

## 2. 严格正负响应

这里测试的不是普通 candidate-repair prevalence，而是：人为构造完全相反的
gradient residual 后，AdamW 是否给出互为相反数的 update difference。

| 案例 | 后半状态的 response remainder | 95% CI | 结论 |
|---|---:|---:|---|
| Qwen saved-P | 正常 odd response 尺度的 2.90% | [1.17%, 4.93%] | 同一方向复现，通过 6 项 Holm |
| Qwen3-VL SiLU | -0.00255% | [-0.00372%, -0.00133%] | 与前半状态方向相反，不算复现 |

saved-P 说明，即使输入 residual 已严格做成正负公平，AdamW 后仍可能留下稳定的
非镜像 response。这是 response-side formation 的规范正例。

SiLU 仍然证明 AdamW response 并非逐状态严格镜像，且旧结果中的绝大部分 response
energy 集中在最初两步。但新的 16+16 检验没有确认一个跨状态保持的平均方向，因此
它保留为 optimizer cold-start / phase-sensitive 案例，不再与 saved-P 并列为稳定
response-direction positive。

## 3. 对主线的补充

这五个案例使当前证据不再只有一种结果：

```text
Qwen lm_head：方向被保留，但 optimizer 大幅压低
Qwen v_proj：局部数值差异没有形成稳定平均方向
Mamba in_proj：backward 中出现缩放，optimizer 中消失
saved-P：严格正负输入被 AdamW 转换成稳定非镜像 response
SiLU：response 不镜像，但方向依赖训练阶段，未通过分半复现
```

因此更准确的主张是：training bias 不是某个 kernel 的静态标签。它必须相对于具体
repair 和训练状态，沿 local、backward、optimizer 逐层测量。
