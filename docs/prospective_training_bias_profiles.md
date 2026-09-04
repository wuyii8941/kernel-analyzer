# 新案例的 Training Bias Profile v2 结果

这轮的目的不是继续解释此前五个开发案例，而是固定方法后检查新训练位置。案例先按
冻结候选池的顺序选定，再运行相同的 32 状态、三阶段、cold-start AdamW 测量。所有
结果都保留，包括完整负例和无法判断。

这里的“新”是指没有参与 Training Bias Profile v2 的分支、阈值和校正规则开发。
它不是对整个仓库历史完全不可见的盲测；部分训练家族此前有工程探针。因此本页提供
有边界的新案例证据，不报告通用准确率或自然发生比例。

## 冻结批次

第一批预先固定四项，主要校正组为 `4 cases × 3 update branches = 12` 项：

| 模型 | 训练位置 | 结果 |
|---|---|---|
| Mamba-130M | 状态空间反向 | 无法判断：当前环境缺少 fast scan kernel，seq256 编译未完成第一状态 |
| DeepSeek-8B | normalization backward | 完成，确认正常 update 被系统缩小 |
| Qwen3-1.7B | attention 状态传递 backward | 无法判断：旧冻结运行包的 source-binding 文件损坏，拒绝近似重建 |
| Phi-4-mini | loss / cross-entropy backward | 完成，三个 update 分支均未确认 |

第二批单独冻结一个不同训练家族，主要校正组为 `1 × 3 = 3` 项：

| 模型 | 训练位置 | 结果 |
|---|---|---|
| DeepSeek-8B | attention projection backward | 完成，确认正常 update 被系统缩小 |

两批分开冻结和校正。第二批不是看到第一批结果后塞回第一批来改变门槛。

## 主要数值

`正常 update 变化`为负，表示 candidate 相对 repair 反复把该状态本来会发生的正常
参数更新压小。它不是固定参数方向，也不要求不同输入的正常 update 指向同一方向。

| 案例 | local 正常信号变化 | gradient 正常信号变化 | AdamW update 正常信号变化 | update 95% 区间 | 校正后 p | 判断 |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek normalization backward | −3.39% | −3.15% | **−13.68%** | **[−15.82%, −11.55%]** | 0.00600 | 确认 |
| DeepSeek attention projection backward | **−2.23%** | **−2.21%** | **−10.69%** | **[−12.02%, −9.37%]** | 0.00150 | 确认 |
| Phi loss / CE backward | +0.062% | +0.079% | +0.079% | [−0.138%, +0.296%] | 1.0 | 未确认 |

第一批 DeepSeek 的 local/gradient 区间不跨零，但没有通过该批 24 项解释组的 Holm
校正，因此只作为“约 −3%”的描述，不称为确认。第二批 attention 的 local/gradient
对齐分支在它自己预先冻结的 6 项解释组中通过。两项 DeepSeek 的主要 update 结果均在
各自预先冻结的 update 组中通过。

固定方向与去掉缩放后的剩余方向在三个完成案例的 update 层均未确认。新结果因此不是
又一个 Liger 式固定剩余方向，而是更清楚的另一种形式：

> 实现差异可以随着每个输入的正常 update 一起旋转，却反复把它缩小；AdamW 可能把
> local/gradient 中约 2%–3% 的缩小变成约 11%–14% 的 update 缩小。

这只是两个 DeepSeek 位置在一个 checkpoint 和 cold-start AdamW 下的观察，不能写成
AdamW 普遍放大所有数值差异。

## 配对训练后果

两个确认项目都按同一规则启动最多 4096 步的四臂配对重放，并在首次 loss 分叉时停止：

| 案例 | 停止步 | 参数距离 | 配对 loss gap | 可以说明什么 |
|---|---:|---:|---:|---|
| DeepSeek normalization backward | 1 | 0.17847 | 0.00950384 | 已确认的 update bias 能立即改变下一步 loss |
| DeepSeek attention projection backward | 1 | 0.09114 | 0.00934792 | 同上，且不是单一 normalization 位置 |

这两项证明轨迹不再相同，不证明方向保持 4096 步，也不证明最终训练质量有同等大小的
变化。`4096` 是预先声明的最大 horizon；由于达到停止条件，实际只运行 1 步。

## 这轮对主线的影响

1. 方法没有被调成“只要 residual 非零就报 bias”：Phi loss backward 是完整负例。
2. 新案例支持至少两种 update 几何：此前 Liger 的剩余共同方向，以及这轮 DeepSeek
   的随正常 update 旋转的稳定缩小。
3. local、gradient 与 AdamW update 的变化不同，说明只测算子输出或 gradient 仍不够。
4. 两个无法判断项留在分母中，说明当前工具的运行包和模型后端仍是实际限制。

机器结果：

- `results/property/training_bias_profile_v2/prospective_batch_1/summary.json`
- `results/property/training_bias_profile_v2/prospective_batch_2/summary.json`
- `results/property/training_bias_profile_v2/prospective_batch_1/consequence/deepseek_norm_4096.json`
- `results/property/training_bias_profile_v2/prospective_batch_2/consequence/deepseek_attn_projection_4096.json`

## 统一的 16 项冻结验证

在前两批之后，我们又从既有冻结候选池中按不读取数值结果的规则选定 16 个训练
位置，并把 16 项、三个 AdamW update 效应放进同一个 48 项 Holm 校正组。最终 15 项
完成测量，Mamba seq256 的实际 backward 图与冻结记录不一致，因此保留为无法判断，
没有换案例。

最终结果是：

- 9 项确认了相对正常 update 的稳定缩小，来自 DeepSeek、Qwen 和 Phi；
- 4 项的三个效应区间都落入预先冻结的工程范围，包括 Mamba seq64 和两个 Mamba
  seq128 位置；后两个位置的 AdamW update difference 在 32 个状态中逐位为零；
- 2 项现有区间仍不足以判断是否落入工程范围；
- 1 项因 backward 图不一致而无法判断。

确认结果中的正常 update 缩小范围为 `1.289%` 到 `15.818%`。其中值得单独指出：

| 模型与训练位置 | 正常 update 变化 | 95% 区间 |
|---|---:|---:|
| DeepSeek seq128 attention-state backward | −15.818% | [−19.590%, −12.047%] |
| DeepSeek seq256 normalization backward | −13.464% | [−16.803%, −10.125%] |
| Qwen seq256 attention-state backward | −12.984% | [−18.971%, −6.997%] |
| Qwen seq128 attention-projection backward | −5.882% | [−7.035%, −4.730%] |
| Phi seq256 normalization backward | −2.583% | [−2.907%, −2.259%] |

这轮扩大了模型和训练位置范围，但仍使用现有四个核心模型、同一个 checkpoint 类型
和 cold-start AdamW。它证明冻结方法不是只对开发案例有效，不证明跨所有模型、
checkpoint 或真实 warm optimizer state 都保持相同标签。机器结果见
`results/property/generalization_benchmark_v1/summary.json` 和
`results/property/generalization_benchmark_v1/equivalence.json`。
