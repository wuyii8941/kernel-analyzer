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

## 修改失败后的分块误差诊断

为避免继续枚举 optimizer 变体，仓库使用已保存的 32 个真实 gradient，按 TorchAO
源码中的分块缩放、最近 qmap 编码、反量化和 AdamW 方程进行 CPU 数学重放。这是
结果后的开发诊断，不是新的真实 Triton 执行或训练确认。其默认参数写入 RMS 为
5.4816%，接近原真实执行的 5.4502%，但不把两者称为逐位相同。

在后 16 步中，一阶 moment 量化误差只有 0.391% 的能量能由“每个 256 元素块共享
一个加性偏移”解释；二阶 moment 也只有 0.976%。给一阶 moment 每块增加一个 FP32
均值修正量，参数写入 RMS 反而由 5.4816% 增至 5.4945%。进一步把已有 block scale
改成该块的最小平方误差 scale，虽然当步一阶 moment 的量化误差下降约 3.40%，递推后
参数写入 RMS 仍增至 5.6170%。

因此现有结果不支持“误差主要是每块共同漂移，去掉共同漂移即可改善训练”这一简单
解释，也再次表明局部状态重构误差下降不保证后续参数写入更接近参考。两项候选在开发
阶段即被否定，不进入昂贵训练确认。下一步若继续该链，应研究跨步误差与梯度/AdamW
分母的联合响应，而不是把单步 RMS 最小化当成训练目标。

## 跨步量化残差与新的针对性修改

默认 AdamW8bit 在第 $t$ 步保存量化后的 moment。以一阶 moment 为例，记未量化更新为

\[
m_t=\beta_1\widetilde m_{t-1}+(1-\beta_1)g_t,
\qquad \widetilde m_t=Q(m_t),
\]

并定义本步保存时丢失的残差 $e_t=m_t-Q(m_t)$。下一步实际读取的是
$\widetilde m_t=m_t-e_t$，所以 $e_t$ 会进入下一次 moment 递推；二阶 moment 同理。
参数写入又同时依赖一阶和二阶 moment，因此“本步写入 RMS 较小”没有给出这些跨步
残差如何进入以后训练的充分信息。这解释了为什么 block64 与 FP32-first-moment 的
单步改善不能直接推出 loss 改善，但它本身还不是训练结果的证明。

基于这个递推关系，新的开发变体仍使用 256 元素分块量化 moment，同时用 BF16 保存
$e_t$，下一步读取 $Q(m_t)+e_t$ 后再做递推。它针对的是被保存并带入下一步的量化
残差，而不是再次缩小当前分块。该变体只支持 dense FP32 参数与普通 AdamW，属于机制
验证代码，不是已经优化吞吐和显存的生产实现。

修改方案冻结后，在 16 条新的独立长度 8 gradient history 上进行确认。16/16 条中，
补偿版相对 FP32 AdamW 的参数写入 RMS 都低于默认 block256；改善概率的精确二项单侧
95% 下界为 82.93%。平均 RMS 从 4.32513% 降至 0.007392%。因此“补回跨步残差会降低
真实参数写入差异”的事前预测成立。这个实验尚不能单独证明 loss 改善；训练结论必须
由另行冻结的数据流实验给出。

随后按结果揭示前冻结的训练协议，在 8 条新的、不重叠的 WikiText 数据流上比较
默认 `ADAMW8BIT_BLOCK256` 与补偿版 `ADAMW8BIT_COMPENSATED_BLOCK256`，每条运行
1024 步，并以同一 32-state 评估集合的第 1024 步平均 loss 作为主要指标。主要差值
`default−compensated` 在 8/8 条数据流中为正，均值为 `+0.0248481`，95% 配对 t
区间为 `[+0.0168563,+0.0328399]`，整个区间超过事前声明的 `+0.01` 实际改善门槛，
因此本次主要判断为 `MATERIAL_IMPROVEMENT`。相对 FP32 AdamW，默认版均值为
`+0.0376128`，补偿版为 `+0.0127647`；补偿版仍有额外显存和速度代价（本次峰值约
4.1–5.1 GB，且低于默认版的步速）。

这闭合了本协议下“跨步量化残差 → 参数写入改善 → 固定训练设置中的实际 loss 改善”
这一条机制链，但范围只是一台 Mamba checkpoint、一个优化器设置和 8 条预注册数据流。
它不是生产优化器证明、跨模型结论或崩溃机制证明。独立复算状态为 `VERIFIED`；沙箱中
一次无法访问 GPU 的失败记录保留在训练结果目录的 `runtime_failures/` 中，不计入数据流。

训练确认使用同一个 checkpoint，随机单位是 8 条数据流，而不是 checkpoint 或模型。
区间依赖这些数据流可作为近似独立单位的假设，不能推广到其他模型、checkpoint、
数据集或 optimizer 设置。当前环境缺少 Mamba 加速 kernel，模型使用相同的顺序路径；
被测 Triton 计算来自 Torch 编译生成的 AdamW8bit optimizer，不应把训练结果归因于
Mamba kernel。

## 边界

这个结果证明新算子族可复用统一统计入口，并在声明训练协议中得到超过预设门槛的
loss 差异。它没有证明崩溃、跨 checkpoint 泛化或 64-block 修改带来稳定改善。
1% update 范围与 0.01 loss 门槛也不是所有 LLM 共享的安全阈值。

## 随机 optimizer history 总体

固定 32-state sequence 不能支持随机训练状态总体结论。后续协议将已有八个冻结 token
bank 合为一个 8192-block 经验分布，并在结果揭示前独立有放回抽取 32 条长度 8 的
gradient history。每条 history 都从相同 Mamba checkpoint 计算真实 gradient，重新
建立 AdamW8bit 与 FP32 AdamW state，在共同 FP32 参数上推进相同 gradient，最后读取
实际参数写入。

32/32 个独立单位的 statewise write RMS 均超过 1%，范围约为 3.51%–4.74%；描述性
平均能量 RMS 为 4.18%。精确二项单侧 95% 下界为 91.06%，所以相对于“该经验总体中
最多 5% history 超过 1%”的预声明合同，结果为 `NON_EQUIVALENT`。

这个结果关闭的是**超范围 history 的比例**，不控制少数超范围值的大小，也不是平均
$Q$ 的总体证书。gradient 由固定 checkpoint 计算、参数在每步写入前恢复共同值，因此
它分析 optimizer-state response，不代表两条自然递推训练轨迹或其他 checkpoint。

另一个结果后机制实验检查旧 block64 结论是否只是挑中了一个参数。它在 16 条新独立
history 上汇总全部 129,135,360 个模型参数的实际写入；block64 的全模型 RMS 在 16/16
个单位中低于 block256，精确二项单侧 95% 下界为 82.93%。两者平均 RMS 分别为
270.66% 与 483.20%。逐参数复算确认第一个单位与保存总量一致、没有零 reference-write
参数；该单位中 embedding 与最后一层 mixer output projection 合计贡献 block256
effect energy 的 99.84%，所以巨大比例不是均匀分布于所有参数，也不能解释成全模型
统一缩放。

这使训练阴性结果更有信息：block64 不仅在原 `x_proj`，也在全模型写入上稳定降低
RMS，但旧 1024-step 实验仍未确认 loss 改善。两种实现相对 FP32 的写入差异与最后一步
gradient 的一阶内积在 16/16 个单位中均为负，仍不能预测递推训练后的评估 loss。
因此下一步要解释 trajectory response，而不是继续把单步 RMS 最小化当作修复目标。

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
- 修改失败后的分块误差诊断：`results/property/numerical_coverage_v1/adamw8bit_block_residual_development_v1.json`
- 随机 history 总体协议、逐单位统计和判断：`results/property/numerical_coverage_v1/adamw8bit_population_update_v1/`
- 全模型 block-size 机制实验与逐参数核验：`results/property/numerical_coverage_v1/adamw8bit_full_model_block_probe_v1/`
