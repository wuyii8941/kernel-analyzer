# 统一实验方法

本页只定义当前方法。历史 T1–T4 名称仍可在审计 artifact 中出现，但论文主线按
“成因分解 → 三阶段测量 → 短程筛查 → 长程后果”组织。

## 1. Candidate、repair 与 matched state

测试单元是一个具体 forward invocation，加上真实消费其 saved tensors 和
cotangent 的 backward。candidate 与 repair 必须共享：

- weights、inputs、saved tensors 和 RNG；
- optimizer moments、step、scheduler 和 loss scaler；
- 除声明 implementation boundary 外的全部代码路径。

若边界不能闭合、参数不可达或 state identity 无法确认，结果为 `ABSTAIN`。

## 2. 先分解方向形成的两项来源

在预先声明的 residual boundary 上，令 `F_s(e)` 表示 residual `e` 经真实
backward 和 optimizer 后造成的 update difference。正负奇偶分解给出：

\[
\int F_s(e)\,dP_s(e)
=
\int F_s^{\mathrm{odd}}(e)\,dP_s^{\mathrm{asym}}(e)
+
\int F_s^{\mathrm{even}}(e)\,dP_s^{\mathrm{sym}}(e).
\]

- 第一项是 source-side asymmetry：residual 事件或它们与训练状态的配对不平衡；
- 第二项是 response-side rectification：严格 `+e/-e` 经过 downstream 后不再
  互为相反数。

这只是所选 boundary 上的精确两项账本，不穷尽量化、截断、饱和、underflow、
reduction order、scaling 或 saved-state 等底层成因。

## 3. 同一 matched contrast 测三阶段

对 state `i` 和 stage `k`：

\[
u_{i,k}=Y_{i,k}^{C}-Y_{i,k}^{R},
\qquad
r_{i,k}=Y_{i,k}^{R}.
\]

`k` 依次为：

1. local output；
2. parameter gradient；
3. target optimizer update。

update 是主要正确性端点；前两层用于判断方向是在 source、backward 还是 optimizer
阶段形成或消失。

## 4. 统一统计输出

每层保存并报告：

- total effect energy `E||u||²`；
- mean effect `||E[u]||`；
- normal training scale `E||r||²`；
- mean effect / normal update RMS；
- 与 repair signal 对齐的缩放分量；
- 去掉缩放后的 residual mean；
- calibration/confirmation split、效应量和置信区间。

对 repair energy 高于预声明 floor 的 state：

\[
u_i^{\parallel}
=\frac{\langle u_i,r_i\rangle}{\|r_i\|^2}r_i,
\qquad
u_i^{\perp}=u_i-u_i^{\parallel}.
\]

整体 aligned effect 使用稳定的加权形式：

\[
g=\frac{\sum_i\langle u_i,r_i\rangle}{\sum_i\|r_i\|^2}.
\]

它在 local、gradient、update 三层含义不同，不能跨阶段当作同一个缩放系数。

## 5. 统计单位与多重比较

- fixed suite：报告该 suite 的精确均值，不外推到随机训练总体；
- random matched states：用 untouched confirmation states 计算 held-out signed
  effect 和置信区间；
- 同一 run 的连续 states：按 run/cluster 处理，不能假装彼此独立；
- long-run loss：独立 training run 才是总体推断单位。

若同一论文表中同时判断多个 cases 或 stages，预先声明 confirmatory family 与
discovery family，主报告 Holm 校正；同时保留效应量、置信区间和未校正数值。
没有通过校正的候选保持 unresolved，不自动改成 negative。

## 6. Orbit mean 的限制

仅对 reduction、summation、reassociation 类 source，冻结合法 schedule distribution
`nu`：

\[
m_{\mathrm{orb}}(a;\nu)
=\mathbb E_{\pi\sim\nu}
\left[\operatorname{fl}_{\pi}(a)-y^\star(a)\right].
\]

它是 source-side candidate predictor，需要多个等价 schedules。它不表示每种
schedule 同号，也不能代替 backward、optimizer 或长程训练。

Liger 的预注册检查是：BF16 `m_orb` 与 local mean direction 对齐；改用 FP32
accumulator 后二者同步下降。只有这一预测在 confirmation states 成立时，才支持
reduction-family predictor。

## 7. Response contrast

对严格正负重放，保存独立的 contrast：

\[
u_i^{\mathrm{resp}}
=\tfrac12(Y_i^+ + Y_i^- -2Y_i^0).
\]

它与 candidate-repair contrast 使用同一统计输出，但不同 `contrast_id`、不同
prevalence denominator。

## 8. 短程与长程

16/32 步的 `A(T)`、prefix、lag 和本行 sign-flip null 只用于筛查与描述。长期
结论来自声明的 4096 步实验和 late rolling windows。

四臂实验拆分：

\[
\text{actual increment}=\text{direct effect}+\text{feedback effect}.
\]

interaction 只检查 direct effect 是否依赖当前 trajectory state，不加入上述恒等式。

loss split 是 consequence evidence，不单独证明 direct bias；4096 步也不等于 loss
收敛。

## 9. 最小机器可读字段

每条统一记录至少包含：

```text
case_id, contrast_id, stage, model, implementation_boundary,
repair_provenance, optimizer, moment_state, parameter_scope,
claim_scope, run_id, cluster_id, calibration_state_ids,
confirmation_state_ids, G_uu, G_rr, G_ur,
per_state_effect_energy, per_state_repair_energy, sham_result,
effect_size, confidence_interval, adjusted_p_value, decision
```

旧 artifact 不具备字段时保持 `PARTIAL_IDENTITY` 或 `UNRESOLVED`，不从相近运行补值。
