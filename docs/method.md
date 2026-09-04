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

三个方向指标用于解释差异是固定方向、随正常 update 缩放，还是去掉缩放后仍有
共同方向。它们不能排除其他方向，因此不能单独签发等价结论。对固定的后 16 个
输入，还必须计算覆盖全部坐标的 update difference 总体比例：

\[
s_{\mathrm{full}}
=
\sqrt{
\frac{\sum_i\|u_i\|^2}
{\sum_i\|r_i\|^2}
}.
\]

这个量直接由 `G_uu` 与 `G_rr` 的对角线求和得到，不依赖前 16 个输入学到的方向。
不超过 4096 个坐标时保存完整向量 Gram；更大的向量使用三个事先固定、彼此独立的
CountSketch，并取三者中最大的比例。后者是覆盖全部坐标的随机摘要，不等于保留原
向量。它的作用是防止后 16 个输入出现全新、与原方向垂直的大差异仍被放行。

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
对一个固定测试集合，上式直接在全部 confirmation states 上计算。需要外推到新的
训练状态时，先在每个独立 training unit 内计算同一加权量，再以 training unit 为
统计单位；不再平均每一步各自的比例。

## 5. 固定输入集合上的 update 等价

当前证书只判断声明的后 16 个输入，不外推到随机训练状态总体或完整训练质量。只有
同时满足下面两项，才返回 `FIXED_SUITE_UPDATE_EQUIVALENT`：

1. `s_full` 小于完整 update 的工程范围；
2. 固定方向、相对正常 update 的缩放、去掉缩放后的剩余方向，三项区间都落入各自
   工程范围。

只有在保存完整向量且所有 update difference 逐位为零时，才单独返回
`EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE`。三个随机摘要都为零不能写成逐位相同。若
总体 update 比例超过范围，即使三个方向指标都很小，也不能返回等价。训练后果若未
预先声明，机器结果必须写
`material_consequence_status = NOT_DECLARED`，不能默认成已经检查并通过。

完整 update 的 `1%` 范围是在看到 16 项验证结果后，为修复已发现的方向盲区而加入；
它取已有最大缩放范围 `1%`，避免新增兜底条件比原合同更严格。这个修正明确标为
post-reveal，不冒充事前冻结，也不是所有 LLM 训练共享的安全常数。旧三方向证书保留
作历史记录，但不再作为当前等价结论。

## 6. 统计单位与多重比较

- fixed suite：报告该 suite 的精确均值，不生成总体置信区间；
- random matched states：calibration 和 confirmation 必须来自互不重叠的独立
  training units，当前 v2 最少各需要 8 个；
- 同一 run 的连续 states：全部算一个 training unit。若同一 run 同时出现在
  calibration 和 confirmation，只保留描述结果，不作总体判断；
- long-run loss：独立 training run 才是总体推断单位。

v2 对独立 training-unit effects 报告 Student 区间，并使用 studentized sign-flip
作为辅助检验。最终确认同时要求：区间不跨零、Holm 校正后通过，以及从 calibration
学到的 additive/residual 方向在 confirmation 中没有反转。规则见
[`training_bias_profile_v2.md`](training_bias_profile_v2.md)。

若同一论文表中同时判断多个 cases 或 stages，预先声明 confirmatory family 与
discovery family，主报告 Holm 校正；同时保留效应量、置信区间和未校正数值。
没有通过校正的候选保持 unresolved，不自动改成 negative。

当前五案例的输入银行由冻结且不重叠的输入窗口组成，部分采用确定性选取，不能证明
是更大训练总体的随机样本。因此这次区间只作为“后 16 个窗口之间是否稳定”的确认门，
结论范围止于声明的窗口集合和 checkpoint；不能解释为跨 checkpoint 或独立 runs 的
总体区间。

## 7. Orbit mean 的限制

仅对 reduction、summation、reassociation 类 source，冻结合法 schedule distribution
`nu`：

\[
m_{\mathrm{orb}}(a;\nu)
=\mathbb E_{\pi\sim\nu}
\left[\operatorname{fl}_{\pi}(a)-y^\star(a)\right].
\]

它是 source-side candidate predictor，需要多个等价 schedules。它不表示每种
schedule 同号，也不能代替 backward、optimizer 或长程训练。

当前 Liger 检查让两个实现都使用 FP32，只改变 `dW` 分块结果的加法顺序。前 16 个
输入确定预测方向，后 16 个输入负责检查。它支持这一种 reduction 来源的预测价值，
不覆盖量化、saved state、一般 backward 或 optimizer 机制。BF16 与 FP32 accumulator
同步下降的更强联合预测仍不能由这项结果代替。

## 8. Response contrast

对严格正负重放，保存独立的 contrast：

\[
u_i^{\mathrm{resp}}
=\tfrac12(Y_i^+ + Y_i^- -2Y_i^0).
\]

它与 candidate-repair contrast 使用同一统计输出，但不同 `contrast_id`、不同
prevalence denominator。

## 9. 短程与长程

16/32 步的 `A(T)`、prefix、lag 和本行 sign-flip null 只用于筛查与描述。长期
结论来自声明的 4096 步实验和 late rolling windows。

四臂实验拆分：

\[
\text{actual increment}=\text{direct effect}+\text{feedback effect}.
\]

interaction 只检查 direct effect 是否依赖当前 trajectory state，不加入上述恒等式。

loss split 是 consequence evidence，不单独证明 direct bias；4096 步也不等于 loss
收敛。

## 10. 最小机器可读字段

每条统一记录至少包含：

```text
case_id, contrast_id, stage, model, implementation_boundary,
repair_provenance, optimizer, moment_state, parameter_scope,
claim_scope, run_id, cluster_id, calibration_state_ids,
confirmation_state_ids, G_uu, G_rr, G_ur,
per_state_effect_energy, per_state_repair_energy, sham_result,
effect_size, confidence_interval, full_update_rms, full_update_rms_margin,
material_consequence_status, adjusted_p_value, decision,
inference_unit_id, sketch_schema, sketch_seed, sketch_dimension
```

旧 artifact 不具备字段时保持 `PARTIAL_IDENTITY` 或 `UNRESOLVED`，不从相近运行补值。

## 11. v1 与 v2 的边界

Liger、Phi、Qwen、Mamba、saved-P 和 SiLU 的早期 16+16 数字属于 v1。它们仍是有效
的固定-suite 与机制观察，但不覆盖 v2 结果。

v2 已按同一协议重采 Liger、Phi `lm_head dX`、Qwen `lm_head dX`、Qwen `v_proj`
和 Mamba `in_proj`。它们全部满足：

- 32 个冻结输入窗口、16/16 分离；
- 使用 `SPLITMIX64_COUNT_SKETCH_V2` 或完整向量 Gram；
- 在看到 empirical result 前提交 protocol、检验组和判定规则；
- 对大向量结果使用两个额外 sketch seeds，或用完整向量复核；
- update 的 15 项与 local/gradient 的 30 项分别作 Holm 校正；
- 只把 update 作为主要训练端点。

完整结果见 [`five_case_training_bias_profile_v2.md`](five_case_training_bias_profile_v2.md)。
下一步保持方法不变，转向结果未知的 held-out implementation pool。
