# Training Bias Profile v2

这版方法用同一套规则回答一个问题：candidate 相对 repair 造成的差异，在算子输出、
parameter gradient 和 AdamW update 中，究竟表现为哪一种可重复结构。

## 统一输入和三种测量

对同一完整训练状态，定义：

```text
u = candidate - repair
r = repair 侧正常信号
```

local、gradient 和 optimizer update 都计算以下三项：

1. **固定方向**：前 16 个输入窗口找到一个平均方向，后 16 个窗口检查该方向是否重复；
2. **随正常信号缩放**：candidate 是否反复放大或缩小同一状态的 repair signal；
3. **剩余方向**：去掉每个状态上的缩放后，是否还留下可重复的平均方向。

三个分支对所有案例同时运行，不按案例挑选。真正写入参数的是 AdamW update，所以
update 层是主要结果；local 和 gradient 只解释结构在哪里出现、增强或消失。

## 冻结判定规则

五例使用 32 个冻结且不重叠的输入窗口：前 16 个只用于确定方向，后 16 个用于检查。
每个窗口都恢复同一 checkpoint、同一目标参数和零初始化 AdamW 历史状态；AdamW 在该
窗口内正常计算一次 update，`weight_decay=0`。

确认一个分支必须同时满足：

- 后 16 个窗口的 95% 区间不跨零；
- Holm 整体校正后 `p <= 0.05`；
- 大向量的三个预先声明 CountSketch seeds 方向一致，或使用完整向量；
- 固定方向与剩余方向不能相对前 16 个窗口反转。

主要检验组是 `5 cases × 3 update branches = 15` 项。local 和 gradient 组成另一组
`5 × 2 × 3 = 30` 项，只用于解释。规则在任何五例数值揭示之前提交；结果不通过时
只写“本协议未确认”，不输出 `SAFE`。

## 合成验证

实际五例使用一个输入窗口作为一个统计单位，因此 go/no-go 采用相同的单窗口设计。
每个场景重复 200 次：

| 场景 | 至少一个分支误报或检出 | 零效应区间覆盖率 |
|---|---:|---:|
| 零均值 | 1.5% | 93.5%–97.0% |
| 重尾零均值 | 1.5% | 95.0%–96.5% |
| 偏斜零均值 | 4.0% | 93.5%–95.0% |
| 正负交替 | 2.0% | 96.5%–98.5% |
| 固定平均方向 | 100% 检出 | — |
| 随正常 update 旋转的缩放 | 100% 检出 | — |
| repair 过小 | 200/200 无法判断 | — |

机器结果：
[`single_state_unit_validation.json`](../results/property/training_bias_profile_v2/single_state_unit_validation.json)。

## 五例统一结果

| 案例 | local | gradient | AdamW update 主要结果 |
|---|---|---|---|
| Liger fused CE | 剩余方向 `0.326%` | 剩余方向 `0.300%` | 剩余方向 `0.0151%`，95% 区间 `[0.00614%, 0.0240%]` |
| Phi `lm_head dX` | 剩余方向 `0.0375%` | 剩余方向 `0.372%` | 正常 update 缩小 `0.1506%`，区间 `[0.0966%, 0.2046%]` |
| Qwen `lm_head dX` | 很小的同向缩放 | 剩余方向 `0.00491%` | 三个分支均未确认 |
| Qwen `v_proj` | 未确认 | 未确认 | 正常 update 缩小 `6.04%`，区间 `[3.44%, 8.64%]` |
| Mamba `in_proj` | 极小缩放 | 未确认 | 正常 update 缩小 `3.12%`，区间 `[2.86%, 3.37%]` |

以上百分比都相对于本阶段 repair signal；三个阶段的分母不同，不能把 local 百分比与
update 百分比直接解释为传播倍率。完整区间、原始 p 值、Holm 校正值和三 seed 结果见
[`five_case_summary.json`](../results/property/training_bias_profile_v2/five_case_summary.json)，
人类可读审计见 [`five_case_training_bias_profile_v2.md`](five_case_training_bias_profile_v2.md)。

## 这组结果说明什么

- 五例并非同一种固定低秩偏差。Liger 在去掉正常 update 缩放后仍有共同方向；Phi、
  Qwen `v_proj` 和 Mamba 主要表现为反复缩小各自状态下会旋转的正常 update。
- Qwen `lm_head dX` 在 gradient 中有共同方向，但 cold-start AdamW update 未确认，
  说明 gradient 不能代替目标 optimizer 的结果。
- Qwen `v_proj` 和 Mamba 的 4096 步固定参数方向没有通过各自长程随机基线，但本次
  发现了随正常 update 旋转的稳定缩小。两种判断对象不同，不能都叫持久固定方向。
- 这五例是方法开发和解释集合，不是自然流行率样本，也不能证明跨模型泛化。

## 证据范围

输入窗口来自冻结数据银行，部分由确定性规则选择。它们互不重叠，但不能据此声称是
某个未定义训练总体的随机样本。因此这里的区间只描述一个 checkpoint 上这组输入窗口
之间的变化，不外推到其他 checkpoint、warm optimizer moments 或独立训练 runs。

事前协议曾把窗口写成独立抽样单位；结果揭示后的
[`empirical_protocol_amendment_3.json`](../results/property/training_bias_profile_v2/empirical_protocol_amendment_3.json)
只收紧这一表述，没有改变任何数值、阈值或标签。

方法冻结后的新案例结果见
[`prospective_training_bias_profiles.md`](prospective_training_bias_profiles.md)。第一批
四项得到一个确认 update effect、一个完整负例和两个无法判断；第二批 attention
projection 得到另一个确认 effect。两个 DeepSeek 案例都表现为相对正常 update 的
稳定缩小，而不是固定参数方向。它们有首次 loss 分叉后果，但不是 4096 步持续性结果。

随后完成的统一 16 项冻结验证中，15 项得到有效测量：9 项确认 update 缩小，4 项在
冻结工程范围内等价，2 项现有数据不足；Mamba seq256 因实际 backward 图不一致保留
无法判断。Qwen 原冻结位置与 Mamba seq64/128 均使用原案例完成，没有用替代案例补位。

这仍不是完整的未见实现泛化评估：新案例没有参与 v2 方法开发，但部分训练家族在仓库
历史中已有工程检查。跨 checkpoint 和真实 AdamW 历史状态的检查单独报告，不能回头
改变这些冻结规则。
