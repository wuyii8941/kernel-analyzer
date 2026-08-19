# 八案例 Bias Formation 系统审计

## 结论

8 个案例都具有完整或语义闭合的 F+B 边界和因果成对轨迹，但它们**尚不能被解释为一个共同 property 的 8 个正例**。当前最严格的机制分层是：

- Liger：case-specific source mechanism；
- Phi：case-specific composite transport mechanism；
- saved-P：case-specific F/B numerical-contract mechanism；conditional formation 未测，四反事实 recurrence 已闭合且 local/feedback 累积同量级；
- layer-23：semantic-region transport/contract mechanism，不是单 kernel root；
- Qwen64 与 Mamba：partial source mechanisms；
- Qwen128 v_proj：source decomposition 与 trajectory repair 不是同一 contrast，暂不能拼接；
- Qwen3-VL SiLU：因果 backward implementation difference 和 trajectory 已闭合，bias formation mechanism 未闭合。

## 为什么会出现系统性 bias

统一解释不是“误差大”，而是条件化的一阶与高阶项：

`E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。

- `E[ε|c] != 0`：source arithmetic 在声明条件下已经有方向（Liger；MM source candidates）。
- `Cov(T,ε|c) != 0`：局部 residual 可近似居中，但真实 backward pairing 将其整流（Phi）。
- `E[R(ε)|c] != 0` 或 numerical contract 改变：saved/reconstructed state 与 backward 表示使语义区域产生方向（saved-P、layer-23）。
- optimizer 还可能把 centered gradient residual 变成 update bias，但 8 个案例中尚无严格 P5 positive。
- 进入轨迹后，`D_(t+1)=D_t+L_t+B_t+r_t`；local effect 与 feedback 决定差异持续还是抵消。固定 global carrier 不是必要条件。

## 当前可以声称什么

可以声称：多种 implementation difference 会通过 source asymmetry、backward transport 或 F/B contract 的不同路径形成训练相关的方向性更新，并在闭环轨迹中造成参数分离。

不能声称：已经发现一个跨全部 8 个案例的统一 property；也不能把 global-centered saved-P 称为 variance-only，或把 Qwen128 的 output-rounding source 与 accumulation repair trajectory 拼成同一因果链。

## 下一步

优先补三个最能改变结论的实验：Qwen128 output-rounding matched repair、saved-P conditional formation、SiLU conditional formation + sign-symmetric nonlinear control。其余缺口见 `gap_plan.json`。
