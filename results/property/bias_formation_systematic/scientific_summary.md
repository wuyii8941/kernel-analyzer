# 八案例 Bias Formation 系统审计

## 结论

8 个独立案例都具有完整或语义闭合的 F+B 边界和因果成对轨迹。它们不是同一个 kernel bug，也不应被包装成 8 个同质正例；但现在可以由同一个精确的两通道 Bias Formation Map 组织：

- Liger、Phi：`EVENT_PAIRING_ASYMMETRY` 的 matched positives；
- saved-P、Qwen3-VL SiLU：`RESPONSE_RECTIFICATION` 的两个独立 matched positives；
- layer-23：semantic-region transport/contract mechanism，不是单 kernel root；
- Qwen64 与 Mamba：partial source mechanisms；
- Qwen128 v_proj：source decomposition 与 trajectory repair 不是同一 contrast，暂不能拼接；
- 因此严格机制证据是 4/8；其余 4 个保留为 partial、consistent 或 unresolved，而不是重复计数。

## 为什么会出现系统性 bias

统一解释不是“任何环节都可能有偏”，而是一个精确的奇偶分解。固定训练条件 `c` 和从 F+B 数学边界预先声明的 `ε→-ε`，令 `p_s/p_a` 是事件分布的对称/反对称部分，`F_e/F_o` 是真实 F+B+optimizer 响应的偶/奇部分：

`E[F(ε)|c] = ∫p_sF_e + ∫p_aF_o`。

- `∫p_aF_o` 是事件/配对失衡：相反 residual 没有以相同条件质量出现，或 residual 与 transport 的真实配对不具反对称闭包。Liger 与 Phi 分别给出 schedule/source 和 composite transport 的 matched evidence。
- `∫p_sF_e` 是响应整流：即使人为构造严格等范数、反号的 `+δg/-δg`，真实映射仍不满足 `F(-δg)=-F(+δg)`。saved-P 与 SiLU 在相同 Adam state 下独立复现这一项，累计 non-oddness ratio 分别为 `0.6817` 与 `0.6956`。
- 按 response-even 能量加权，saved-P 与 SiLU 分别有 `99.48%` 和 `99.87%` 的偶分量落在梯度符号穿越坐标；两者又分别有 `99.51%` 和 `>99.99%` 的偶分量能量出现在前两步。这把响应整流定位到 Adam 冷启动时的小梯度/符号边界，而不是笼统归因于“优化器非线性”。
- 若事件配对闭合且响应为奇函数，两项同时为零，variance 无论多大都不会产生条件均值。这才是可以区分“有 bias/无 bias”的安全 property。
- 进入轨迹后，`D_(t+1)=D_t+L_t+B_t+r_t`；local effect 与 feedback 决定差异持续还是抵消。固定 global carrier 不是必要条件。

## 当前可以声称什么

可以声称：在四个具有 matched 机制证据的独立 F+B 案例中，training bias 均对应条件反对称消除失败；失败来自事件/transport 配对失衡或真实 optimizer 响应的偶分量。error norm、raw tensor mean、BF16 dtype 与固定 global carrier 都不是统一判据。

不能声称：该 map 已经能零样本预测所有未见算子；也不能把剩余 4 个 partial/unresolved 案例补写成 positives，或把 Qwen128 的 output-rounding source 与 accumulation repair trajectory 拼成同一因果链。

## 下一步

主线下一步不再是重复校验，而是把两个 formation channel 变成自动特征：从 event atomization 测 joint antithetic closure，从 `(g,m,v,δg)` 测 optimizer response-even susceptibility；再用剩余四个案例做预测而非反向拟合。其余个案缺口见 `gap_plan.json`。
