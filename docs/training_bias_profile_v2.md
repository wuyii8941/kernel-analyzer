# Training Bias Profile v2

这版方法修正了 v1 中两个会高估证据的问题：连续训练步骤被当成独立样本，以及大
向量按固定周期折叠。

## 一组状态上实际发生了什么

对同一完整训练状态，计算：

```text
u = candidate - repair
r = repair 侧正常信号
```

local、gradient 和 optimizer update 都使用同一计算。每层报告：

- implementation difference 的总能量；
- calibration 找到的平均方向是否在 confirmation 中重复；
- candidate 是否反复放大或缩小同状态的正常 repair signal；
- 去掉这种缩放后，是否仍留下可重复方向。

如果只运行了一条连续训练轨迹，这些数字精确描述该测试集合，但不生成“其他训练也
如此”的置信区间。

## 什么时候允许外推

只有同时满足以下条件才进行总体判断：

1. calibration 与 confirmation 来自互不重叠的独立 training runs 或独立捕获的
   training-state clusters；
2. 两边至少各有 8 个独立单位；
3. 一个单位内可以包含多个连续步骤，但这些步骤始终一起进入统计计算；
4. 判定规则和多重比较组在结果揭晓前提交到 Git。

确认一项 effect 需要同时满足：

- 95% 区间不跨零；
- Holm 校正后通过；
- additive 和 residual direction 在 confirmation 中没有相对 calibration 反转。

短筛未通过不输出 `SAFE`。

## 大向量怎样处理

v1 的固定折叠会让相隔 4096 个位置的坐标反复碰撞。v2 改用带冻结 seed 的
SplitMix64 CountSketch，并在 artifact 中保存算法、seed、dimension 和原坐标数。

一个 headline 结果还必须满足以下任一项：

- 至少三个预先声明的 sketch seeds 得出一致方向；
- 在完整向量 Gram 上复核。

## 合成验证

v2 的验证直接调用生产统计代码，而不是另写一套较宽松规则。每个独立单位包含 4 个
相关步骤，200 次重复得到：

| 场景 | 整体误报或检出率 | 零效应区间覆盖率 |
|---|---:|---:|
| 相关但零均值 | 1.5% | 94.0%–95.0% |
| 重尾且零均值 | 5.0% | 92.0%–92.5% |
| 偏斜且零均值 | 4.5% | 93.5%–95.0% |
| training units 间正负交替 | 1.5% | 97.0%–98.0% |
| 固定平均方向 | 100% 检出 | — |
| 随正常 update 旋转的缩放 | 100% 检出 | — |
| repair 过小 | 200/200 abstain | — |

机器结果：
[`synthetic_validation.json`](../results/property/training_bias_profile_v2/synthetic_validation.json)。

## 当前边界

合成验证说明 v2 值得进入 empirical recapture，不说明它已经跨模型、算子和 optimizer
泛化。已有 v1 案例保持探索性结果；下一步先重采 Liger、Phi、Qwen `lm_head dX`、
Qwen `v_proj` 和 Mamba `in_proj`，然后才运行完全未参与方法开发的 held-out pool。
