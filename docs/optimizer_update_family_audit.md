# Optimizer parameter and moment update 家族接入

## 实验问题

本实验检查统一框架能否分析一个不属于模型 forward/backward 算子的训练计算：
optimizer 的 moment 与参数更新。人工审核并冻结 TorchAO `AdamW8bit` 相对
`torch.optim.AdamW` 的比较。两者接收相同的真实 Mamba language-model gradient，
并在每一步从相同 FP32 参数值开始；各自的 optimizer moment history 自然演化。

这是固定的 32-state gradient 集合，不是随机训练状态总体，也不是自然递推训练。
前 16 个状态用于方向描述，后 16 个状态用于固定集合判断。主要结果是实际参数在
`step()` 前后的写入差，不是公式中的 proposed update。

## 自动与人工的边界

人工完成：审核两个 optimizer 的比较语义，选择模型参数、gradient 集合和 1% 固定
集合总 RMS 范围。冻结后程序自动完成：真实 gradient 读取、matched transition、两遍
重复性检查、原坐标统计、方向描述、固定集合判断、Triton 源码留存和报告复算。

因此本实验验证的是“带人工审核节点的自动化分析流程”，不要求程序自行发明 optimizer
修改或决定训练验证设置。

## 结果

32 个 transition 的两次执行逐项一致。确认集合结果如下：

| 测量位置 | 相对参考量的总 RMS 差 | 非零差异坐标范围 |
|---|---:|---:|
| gradient 输入 | 0 | 0 |
| 一阶 moment | 3.6248% | 122876–122880 |
| 二阶 moment | 0.5510% | 122830–122863 |
| 实际参数写入 | 5.4502% | 122842–122865 |

实际参数写入超过预声明的 1% 固定集合范围，统一分析输出 `NON_EQUIVALENT`。整体
aligned 系数约为 +0.2676%；大部分写入差异不能由简单整体缩放解释。方向投影仅作
描述，不参与全空间等价判断。

专用 Torch 编译缓存保存了 4 个生成源码文件，均含 `@triton.jit` 定义；生成代码明确
包含 moment 反量化、更新、重新量化和参数写入。这里可以称为编译生成 Triton optimizer
implementation，不依赖 Python 类名猜测 backend。

## 分块大小机制与自然训练确认

随后固定 `block_size = 64, 256, 1024` 三个变体，在同一组 32 个真实 gradient 上
事前预测：分块越细，共享量化 scale 覆盖的数值范围越小，一阶 moment 与参数写入
相对 FP32 AdamW 的 RMS 差异应越小。确认集合严格符合该顺序：

| block size | 一阶 moment RMS | 二阶 moment RMS | 参数写入 RMS |
|---:|---:|---:|---:|
| 64 | 3.3838% | 0.2813% | 4.3804% |
| 256 | 3.6248% | 0.5510% | 5.4502% |
| 1024 | 3.9850% | 0.7681% | 7.2198% |

这确认了一个具体机制：moment 量化分块大小控制 optimizer state 与参数写入的数值
差异。它不是从 loss 结果反推的解释。

机制结果之后，另行冻结 8 条互不重叠、随机选取的 WikiText token 数据流。每条从
同一个 Mamba-130m checkpoint 出发，分别使用 FP32 AdamW、默认 256-block
AdamW8bit 和机制建议的 64-block AdamW8bit 训练 1024 步；主要指标固定为第 1024
步同一 32-state 评估集合的 mean loss，实际幅度门槛固定为 0.01。

默认 AdamW8bit 相对 FP32 AdamW 的配对 loss 差为 8/8 正，平均 `+0.0271846`，
95% t 区间为 `[+0.0121724, +0.0421968]`，整个区间超过预声明的 `+0.01`
门槛。按冻结规则为 `MATERIAL_EFFECT`。对应 perplexity 比值约 `1.02756`，由
区间换算得到约 `[1.01225, 1.04310]`。未观察到预声明的训练崩溃。

64-block 在固定 gradient 上降低了 update distortion，但训练中是否更接近 FP32
没有确认：绝对 loss gap 改善均值为 `+0.0050801`，95% 区间
`[-0.0112490, +0.0214091]`。因此应分别写成：**机制预测成立、默认实现的实际
loss 后果成立、建议修改的训练改善尚未成立**。

在该训练修改未确认后，又冻结了一项更直接的 component-level 预测：只将一阶
moment 保留为 FP32 应降低参数写入 RMS；相反设置用于检查二阶 moment。结果中，
一阶 FP32 将参数写入 RMS 从 `5.4502%` 降至 `3.2082%`，二阶 FP32 降至
`4.4065%`。但冻结规则还要求相反设置的 FP32 二阶状态与标准 AdamW 逐位相同，
实际仍有约 `1.53e-7` 的相对 RMS 差。因此整项规则为 `NOT_CONFIRMED`，没有进入
原先的训练确认。该结果可以说明一阶 moment 是主要来源之一，不能称作已经确认的
新修复。

随后把“一阶 moment 保留 FP32 会改善实际训练”作为**看到 component 结果后提出的
第二轮假设**，另外生成 8 条未用于前述训练的新 token 数据流，并在训练前冻结
1024 步、固定评估集合、主要比较 `default block256 − FP32-first-moment` 和
`+0.01` 的实际改善门槛。三种 optimizer 共 24 个训练任务全部完成，独立复算结果为：

| 比较 | 配对均值 | 95% t 区间 | 冻结判断或解释 |
|---|---:|---:|---|
| default − FP32-first-moment | −0.01070 | [−0.02708, +0.00567] | `NOT_CONFIRMED` |
| default − FP32 AdamW | +0.02443 | [+0.00156, +0.04730] | 次要结果：方向可检测，未越过 +0.01 |
| FP32-first-moment − FP32 AdamW | +0.03513 | [+0.01040, +0.05987] | 次要结果：超过 +0.01 |

修改仅在 3/8 条数据流上优于默认实现，平均方向反而更差，且区间跨零。因此它没有
成为训练修复。这个结果给出一个重要边界：**单步参数写入 RMS 更接近 reference，
并不足以推出递推训练后的 loss 更接近 reference。** 三种条件均未达到预声明崩溃
标准。该第二轮使用全新输入，但假设由已见 component 结果提出，不能写成最初的盲测。

训练确认使用同一个 checkpoint，随机单位是 8 条数据流，而不是 checkpoint 或模型。
区间依赖这些数据流可作为近似独立单位的假设，不能推广到其他模型、checkpoint、
数据集或 optimizer 设置。当前环境缺少 Mamba 加速 kernel，模型使用相同的顺序路径；
被测 Triton 计算来自 Torch 编译生成的 AdamW8bit optimizer，不应把训练结果归因于
Mamba kernel。

## 边界

这个结果证明新算子族可复用统一统计入口，并在声明训练协议中得到超过预设门槛的
loss 差异。它没有证明崩溃、跨 checkpoint 泛化或 64-block 修改带来稳定改善。
1% update 范围与 0.01 loss 门槛也不是所有 LLM 共享的安全阈值。

## 可复算来源

- 协议与原始结果：`results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2/`
- 失败首轮：`results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v1/failure.json`
- 家族证据：`results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_family_evidence_v1.json`
- 汇总：`results/property/numerical_coverage_v1/operator_family_report_v8.json`
- 自动表：`results/property/numerical_coverage_v1/operator_family_table_v2.md`
- 分块机制：`results/property/numerical_coverage_v1/torchao_adamw8bit_block_mechanism_v1/summary.json`
- 训练 pilot：`results/property/numerical_coverage_v1/mamba_adamw8bit_training_pilot_v1/summary.json`
- 训练确认：`results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/`
- 独立复算：`results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json`
- component follow-up：`results/property/numerical_coverage_v1/torchao_adamw8bit_component_mechanism_v1/summary.json`
- FP32-first-moment 训练确认：`results/property/numerical_coverage_v1/mamba_adamw8bit_hybrid_training_confirmation_v1/`
- FP32-first-moment 独立复算：`results/property/numerical_coverage_v1/mamba_adamw8bit_hybrid_training_confirmation_v1/verification_v2.json`
