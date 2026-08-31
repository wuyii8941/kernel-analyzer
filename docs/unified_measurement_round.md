# 统一测量与验证结果

本页记录本轮已经完成的四项工作。所有数字来自机器 JSON；短程方向只称为
“32 个状态中的可复现方向”，不冒充长程训练结论。

> **v2 边界：** 本页是方法开发阶段的 v1 固定-suite 结果。32 个状态沿一条
> repair-driven trajectory 推进，因此下列逐状态区间不能解释为独立 training runs
> 的总体区间；protocol 与结果也没有以两个独立 Git commits 形成可审计的事前冻结。
> 数字和干预结果全部保留，但总体确认需按
> [`training_bias_profile_v2.md`](training_bias_profile_v2.md) 重采。

## 1. Liger 与 Phi 使用同一把尺子

两个案例都使用 16 个 calibration states 和 16 个不参与选方向的后半 states，
同时测量 local output、parameter gradient 和 cold-start AdamW update。18 个检验
在该轮分析中被定义为同一个检验组，并统一使用 Holm 校正。它是完整的回溯敏感性
分析，不再称为可审计的事前确认组。

下表给出相对正常 training signal 的效应量和 95% 区间。`residual direction` 表示
去掉“只是把正常 signal 放大或缩小”之后，仍留下的新方向。

| 案例 | 阶段 | 平均方向效应（95% CI） | 相对正常 signal 的缩放（95% CI） | residual direction（95% CI） |
|---|---|---:|---:|---:|
| Liger fused CE | local | 3.86% [3.38%, 4.35%] | -0.01% [-0.06%, 0.05%] | 3.86% [3.39%, 4.33%] |
| Liger fused CE | gradient | 2.98% [2.62%, 3.34%] | -0.74% [-1.48%, 0.01%] | 2.89% [2.52%, 3.25%] |
| Liger fused CE | AdamW update | 0.0327% [0.0266%, 0.0401%] | -0.0312% [-0.0360%, -0.0269%] | 0.0190% [0.0149%, 0.0239%] |
| Phi `lm_head dX` | local | 0.0381% [0.0348%, 0.0412%] | -0.0059% [-0.0068%, -0.0047%] | 0.0380% [0.0348%, 0.0412%] |
| Phi `lm_head dX` | gradient | 0.3985% [0.3570%, 0.4396%] | -0.0134% [-0.0589%, 0.0315%] | 0.3710% [0.3264%, 0.4174%] |
| Phi `lm_head dX` | AdamW update | 0.0129% [0.0111%, 0.0146%] | 0.0053% [0.0035%, 0.0074%] | 0.0126% [0.0109%, 0.0142%] |

除两项区间跨零的缩放量外，其余表中效应均通过完整 18 项 Holm 校正。最直接的
观察是：两个案例都不是单纯“正常 update 统一缩放”。它们在 local、gradient 和
AdamW update 中都留下了不能由一个缩放系数解释的方向；AdamW 把幅度压低了，
但没有完全消除。

Liger 的 candidate 是 BF16 chunk accumulation，repair 是保持相同 BF16 接口的
FP32 accumulator。因此这同时闭合了 accumulator precision 的 matched intervention。
它尚未验证更一般的 reduction-orbit predictor。

## 2. Phi 的同协议随机舍入干预

这次干预与上表使用相同的 state order、参数位置和 cold-start AdamW state path。

| arm | 32-state 方向分数 | 自身随机上界 | 更新误差能量 / natural | 结论 |
|---|---:|---:|---:|---|
| deterministic BF16 | 1.02971 | 1.00380 | 1.000 | 超过随机抵消范围 |
| no-op sham | 1.02971 | 1.00379 | 1.000 | 精确复现 natural |
| stochastic rounding 0 | 1.00016 | 1.00166 | 0.998 | 回到随机范围 |
| stochastic rounding 1 | 0.99974 | 1.00149 | 1.003 | 回到随机范围 |
| stochastic rounding 2 | 1.00033 | 1.00202 | 1.016 | 回到随机范围 |
| stochastic rounding 3 | 1.00181 | 1.00366 | 0.503 | 回到随机范围 |

前三个随机舍入重复保留了 natural 更新误差能量的 99.8%–101.6%，但稳定方向全部
消失。因此至少对这个案例，干预改变的是误差的平均方向结构，不是简单把误差做小。

## 3. 统计自检

在不运行模型的合成数据上，统一方法经过七种预先声明的检查：零均值大方差、固定
平均方向、随正常 update 旋转的缩放、正负交替、重尾零均值、稀疏平均方向和正常
update 过小。

- 三个零均值场景的整体误报率分别为 2%、2% 和 5%；
- 固定方向、旋转缩放和稀疏方向的目标分支检出率均为 100%；
- 正常 update 过小时 200/200 次选择 `ABSTAIN`；
- 每次同时检查三个分支，并在单次实验内使用 Holm 校正。

这组旧结果只覆盖独立 synthetic states，且 GO 门没有真正检查区间覆盖率。它已经由
v2 的相关-cluster 合成验证替代，不再作为当前统计方法的 go/no-go 证据。

## 4. 冻结的 DeepSeek 未见状态确认

DeepSeek layer-35 attention `dV` 已经完成一轮真正事前冻结的确认，因此没有重新
运行并把旧案例伪装成“新 held-out”。规则和方向先冻结，随后才揭示 32 个不参与
开发的状态。

- candidate 相对 repair 的 `v_proj.weight` gradient 平均缩小 0.370%；
- 95% 区间为缩小 0.059%–0.664%；
- 3 个候选中只有这一项复现；
- 2 个 sign-changing controls 均未被误报。

该结果说明固定参数方向不是唯一形态：不同状态的绝对 gradient 方向可以旋转，
但 candidate 仍可反复缩小同状态下的正常 gradient。它是 gradient-stage、stateless
SGD 映射下的确认，不是 AdamW update 结果，也不是 4096-step loss 后果。

## 5. 本轮结论边界

本轮已经闭合：统一三阶段测量的工程路径、完整 Holm 回溯分析、Phi 同 AdamW 协议
的因果干预，以及一个已有的冻结未见状态确认。v1 的逐状态区间与合成自检已由 v2
替代，不能继续承担独立 training-unit 总体推断。

它仍不等于通用安全判断。下一批最有价值的工作是用同一协议测 Qwen `lm_head dX`、
Qwen `v_proj`、Mamba `in_proj`，并把 saved-P / SiLU 的严格正负 response 保存成逐状态
数据。
