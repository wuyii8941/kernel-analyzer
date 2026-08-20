# 八案例统一证据审计

## 结论

“8 个案例”是审计分母，不是“8 个持久性 bias”。统一口径得到四个互不替代的计数：

- **8/8 causal paired separation artifacts**：都有闭合 F+B、repair/sham 和 live-weight 参数距离；这只证明 implementation contrast 会让两臂分开。
- **6/8 directional-persistence positives**：`liger_fused_ce, phi4_seq64_lmhead_dx, qwen64_vproj_mm, qwen_saved_p_seq128, mamba_seq64_input_proj, qwen_layer23_attention_state` 通过了轨迹局部的预声明/冻结方向门。
- **6/8 matched formation-mechanism positives**：Liger、Phi、Qwen64/128 `v_proj`、saved-P、Qwen3-VL SiLU；Mamba 和 layer-23 在当前 antithetic formation protocol 下保留 partial/unresolved。
- **4/8 same-contrast full chains**：`liger_fused_ce, phi4_seq64_lmhead_dx, qwen_saved_p_seq128, qwen_layer23_attention_state` 将当前 formation mechanism 和 directional persistence 用同一个或闭合语义超集 repair 串起来。

`qwen128_vproj_mm, qwen3vl_silu_layer0` 只有 causal separation，没有确认的方向性 persistence；不得称为完整 Flash-style case。

这四个集合故意不相同。一个案例可以形成条件 bias 但未证明持续，也可以有旧 T1--T4 持久轨迹但当前 formation 干预使用了另一种 repair。

## Formation mechanism

- Liger、Phi：`EVENT_PAIRING_ASYMMETRY` 的 matched positives；
- saved-P、Qwen3-VL SiLU：`RESPONSE_RECTIFICATION` 的两个独立 matched positives；
- layer-23：semantic-region transport/contract mechanism，不是单 kernel root；
- Qwen128：在 16 个固定 state 中，repair local residual 全部 centered，而 candidate-minus-repair 的 local、真实 gradient、SGD update 与 zero-moment AdamW update 全部 biased；这是新的 conditional source-formation positive；
- Qwen64：独立 16-repeat fixed-state confirmation 中，repair local residual 在 16/16 conditions centered，而 candidate-minus-repair 的 local、真实 gradient、SGD update 与 zero-moment AdamW update 均在 16/16 biased；
- Mamba：16-condition joint-repair confirmation 得到 local 与 zero-moment AdamW 16/16 biased、repair local 16/16 centered，但真实 gradient/SGD 为 13/16 biased、3/16 unresolved；因此仍是 partial，而不是第七个 matched positive；
- layer-23：16-condition exact projected-antithetic control 揭示 F+B 与 Adam response-even 分量，但自然 source fidelity 在部分条件未过冻结的 90% gate，因此不升级；
- 因此严格 formation-mechanism positives 是 6/8；Qwen64/128 的 fixed-state formation、Mamba 的 JOINT formation 与各自历史 KERNEL_ONLY 轨迹不能拼接。

## 为什么会出现系统性 bias

统一解释不是“任何环节都可能有偏”，而是一个精确的奇偶分解。固定训练条件 `c` 和从 F+B 数学边界预先声明的 `ε→-ε`，令 `p_s/p_a` 是事件分布的对称/反对称部分，`F_e/F_o` 是真实 F+B+optimizer 响应的偶/奇部分：

`E[F(ε)|c] = ∫p_sF_e + ∫p_aF_o`。

- `∫p_aF_o` 是事件/配对失衡：相反 residual 没有以相同条件质量出现，或 residual 与 transport 的真实配对不具反对称闭包。Liger 与 Phi 分别给出 schedule/source 和 composite transport 的 matched evidence。
- `∫p_sF_e` 是响应整流：即使人为构造严格等范数、反号的 `+δg/-δg`，真实映射仍不满足 `F(-δg)=-F(+δg)`。saved-P 与 SiLU 在相同 Adam state 下独立复现这一项，累计 non-oddness ratio 分别为 `0.6817` 与 `0.6956`。
- 按 response-even 能量加权，saved-P 与 SiLU 分别有 `99.48%` 和 `99.87%` 的偶分量落在梯度符号穿越坐标；两者又分别有 `99.51%` 和 `>99.99%` 的偶分量能量出现在前两步。这把响应整流定位到 Adam 冷启动时的小梯度/符号边界，而不是笼统归因于“优化器非线性”。
- 若事件配对闭合且响应为奇函数，两项同时为零，variance 无论多大都不会产生条件均值。这才是可以区分“有 bias/无 bias”的安全 property。
- 进入轨迹后，`D_(t+1)=D_t+L_t+B_t+r_t`；local effect 与 feedback 决定差异持续还是抵消。固定 global carrier 不是必要条件。

## 当前可以声称什么

可以声称：在六个具有 matched formation 证据的独立 F+B 案例中，conditional training bias 均对应条件反对称消除失败；失败来自 source/event/transport 配对失衡或真实 optimizer 响应的偶分量。另有六个案例具有轨迹局部方向性 persistence，但只有四个把当前 formation 机制与 persistence 在相同 contrast 下闭合。error norm、raw tensor mean、BF16 dtype 与跨无关 state 的固定 global carrier 都不是统一判据。

不能声称：8/8 都是持久性 bias；不能把 basis-free 参数距离增长自动叫 directional bias；不能把 local source repair 自动升级成完整 F+B/optimizer 去偏，也不能把旧 accumulation trajectory 与新的 rounding/joint repair 拼成同一条轨迹因果链。

## 下一步

这一轮 fixed-state conditional audit 已完成。结果不支持通过继续增加随机重复来强行统一 Mamba 或 layer-23；只有获得 exact downstream reference，才能签发 repair 自身的 downstream zero-bias certificate。后续若继续主线，应把两个 formation channel 自动化为 event antithetic closure 与 `(g,m,v,δg)` response-even susceptibility；个案边界见 `gap_plan.json`。
