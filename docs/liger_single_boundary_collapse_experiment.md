# Liger：单处累加差异的全参数训练后果

## 目的与比较对象

只改变 Liger fused cross entropy 的 dW 累加精度，观察 bias 来源的修改是否改变
训练与 loss。原协议还检查能否出现类似 FlashAttention 的训练爆炸；**没有崩溃
不否定 loss 分叉，也不撤销已有成因证据**。

比较 `LigerFusedLinearCrossEntropyLoss(accum_dtype=None)` 与
`accum_dtype=torch.float32`。两边从相同模型、输入顺序、随机和 AdamW 状态出发，
随后自然演化；所有模型参数参与训练。

当前安装的 Liger 0.7.0 源码中，CE 使用 Triton，dW 在
`torch.mm(...).float()` 之后原地累加，最后转回权重类型。因此这是
**混合计算路径中的累加精度对照**，不能仅凭库名归因到 Triton 内部。
本次源码核查不能替代旧运行的执行身份记录。

模型是从零初始化的 4 层、隐藏维度 256 的小型 GPT-2，字符级文本编码；
配置词表大小 8192，序列长度 256，batch size 8，输入/输出权重共享。
不是历史 Qwen 单参数实验，也不是常规大词表 LLM 预训练的总体结论。
使用 BF16 模型计算、FP32 主参数和 AdamW moments。

## 原协议的崩溃标准

非有限数值，或 128 步平均 loss 比大于 1.5 并持续至少 256 步，只是触发条件。
原协议还要求参考继续稳定、差异超过无改动重复变化、三个冻结验证位置更差且在
规定长度内不恢复，并规定触发后的重复确认。完整条件见
[高压力协议](../results/property/single_point_collapse_v2/protocol.json)。

这些自然训练没有触发上述明显失败条件，不能把尚未触发的确认流程说成已全部
完成。有限 loss 差异不叫崩溃，但仍是本研究要记录的后果。

## 普通设置：2048 步

- AdamW 峰值学习率 3e-4；batch 内样本分别抽取。
- 最大 128 步平均 loss 比 1.000139。
- 最终验证 loss：被测实现 1.842756，参考 1.842701。
- 两边正常完成，存在很小的 loss 差异；不宣称稳定训练质量损害。

## 高压力设置：4096 步

模型不变；峰值学习率提高到 1e-3，每个 batch 重复同一段文本，以加强相似表示
和相关 gradient。这是刻意声明的压力条件，不代表普通数据分布。
两条数据流使用同一模型初始化，不是两次独立初始化的总体实验。

| 数据流 | 最大 128 步 loss 比 | 最后 128 步 loss 比 | 被测实现验证 loss | 参考验证 loss |
|---|---:|---:|---:|---:|
| 1 | 1.0061 | 1.0039 | 2.0389 | 2.0111 |
| 2 | 1.0267 | 1.0091 | 1.9915 | 1.9632 |

两条数据流都出现 loss 分叉，但没有声明的训练崩溃。
第一条最终参数相对 L2 距离为 19.69%。

## 第一条数据流原样续至 10000 步

保留第 4096 步参数及 AdamW 历史，学习率保持 1e-4，从后续数据位置继续训练。
不是重新初始化，也没有重新跑第二条 10000 步数据流。

| 指标 | 4096 步 | 10000 步 |
|---|---:|---:|
| 参数相对 L2 距离 | 19.69% | 23.50% |
| 被测实现验证 loss | 2.038870 | 1.821149 |
| 参考验证 loss | 2.011092 | 1.844404 |
| 验证 loss 差（被测 − 参考） | +0.027779 | −0.023255 |

续跑阶段最大 128 步训练 loss 比为 1.008756；最后 128 步约 1.000482，
平均差为 +0.000824。整个续跑阶段平均差为 +0.008895。
这些量描述不同窗口，不用其中一个替代其他窗口。

**参数继续分开，loss 差异仍存在，但验证 loss 的正负反向了。**
这支持这条配对轨迹的分叉，不支持持续恶化的外推。
本次到 10000 步没有观察到声明崩溃，不证明更长时间或其他设置永不崩溃。

这组自然长训练没有逐步重新计算同状态直接/反馈更新。日志中未用字段的零值
不是“测得直接 bias 为零”，也不能反过来借历史 Qwen 的直接分数声称本次后期
直接作用一直增强。

## 受控放大检查的位置

普通设置另测了真实 Liger 更新差形状的放大，以及校准方向的固定注入。
真实形状放大到累计 RMS 约为正常 update RMS 的 93% 时训练仍有限；
固定方向改变了验证 loss，但在测得范围内也未达到崩溃标准。

这是训练对不同扰动形状的敏感性检查，不能当成自然实现已经产生该幅度，
也不能凭这些非完全等能量的比较证明一般性的“方向总比能量重要”。
它不是未来训练的直线外推。

## 对主线的意义

已有累加成因与修改解释，配对全参数训练进一步展示 loss 分叉。
没有崩溃、loss 差反向，都是这个案例的真实边界，而不是必须隐藏的失败。
本实验不能证明 FlashAttention 的特定机制是所有训练崩溃的必要条件。

## 后续独立数据流

在新的分析协议下，另行冻结并运行了 stream 111 的 4096 步 candidate/reference
配对。candidate − reference 的验证 loss 为 −0.00946，与两条历史 4096 步数据流
的 +0.02778、+0.02821 方向相反。三条差的描述性 95% t 区间为
[−0.03820,+0.06923]。因此可重复的结论是训练轨迹发生变化，不是质量变化具有
固定方向。[机器汇总](../results/property/training_numerical_analysis_v1/training_utility_summary.json)

## 原始数据与复算

- [普通设置协议](../results/property/single_point_collapse_v1/protocol.json)与
  [汇总](../results/property/single_point_collapse_v1/summary.json)
- [高压力协议](../results/property/single_point_collapse_v2/protocol.json)与
  [4096 步汇总](../results/property/single_point_collapse_v2/summary.json)
- [续跑协议](../results/property/single_point_collapse_v2/continuation_to_10000_protocol.json)、
  [10000 步汇总](../results/property/single_point_collapse_v2/full_10000_summary.json)
- [训练脚本](../scripts/run_liger_single_boundary_collapse.py)、
  [汇总脚本](../scripts/summarize_liger_single_boundary_collapse.py)

原始 JSON 保留不改。checkpoint 位于 /data1/tzh 的实验缓存中，用于续跑与复核，
不因文档整理删除。
