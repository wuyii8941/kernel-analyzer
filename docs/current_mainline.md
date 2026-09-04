# 当前科研主线

这是项目唯一的论文口径。案例计数和标签以
`results/property/declared_persistent_4096/all_bias_case_audit.json` 为准；人类可读
逐行表与旧实验文档只用于追溯，不能覆盖机器 JSON 的计数。

## 一句话结论

普通 tolerance 告诉我们两个浮点结果相差多大。本项目进一步检查：

> 一个具体训练实现相对声明的 repair，是否在真实 backward 和目标 optimizer
> 后留下可复现的参数更新方向，以及这个方向是否在长程配对训练中留下参数或
> loss 分叉。

因此我们把“误差能量”“平均方向”和“训练后果”分开报告，不再用一个 RMS、
一个 16/32 步分数或最终参数距离代替全部结论。

## 1. 测试合同

每个结论都相对于一个声明的协议：

- 一个具体 candidate implementation；
- 一个只改变目标边界的 repair；
- 相同的模型参数、输入、saved tensors、RNG、optimizer state 和训练步；
- 真实 local output、actual backward 和目标 optimizer；
- 明确的参数范围、状态集合与测量长度。

若 repair 同时改变多个不可分离的训练语义区域，结论属于整个区域。证据不足时
输出 `ABSTAIN` 或 `UNRESOLVED`，不补造阴性。

## 2. 方向从哪里形成

先固定一个 residual boundary，记 residual 为 `e`，它在真实 backward 和
optimizer 后造成的参数更新差为 `F_s(e)`。把 residual distribution 和 response
各自拆成正负对称与不对称部分，可得精确恒等式：

\[
\int F_s(e)\,dP_s(e)
=
\int F_s^{\mathrm{odd}}(e)\,dP_s^{\mathrm{asym}}(e)
+
\int F_s^{\mathrm{even}}(e)\,dP_s^{\mathrm{sym}}(e).
\]

这给出两个互补来源：

1. **source-side asymmetry**：进入 downstream 之前，正负 residual 的数量、
   大小或它们与训练状态的配对已经不平衡。来源可以是 reduction order、量化
   网格、截断、饱和、underflow、scaling contract 或 saved-state mismatch。
2. **response-side rectification**：即使输入被严格构造成 `+e/-e`，真实 backward
   或 optimizer 仍没有产生互为相反数的参数更新。

这是相对于所选 residual boundary 的精确两项分解，**不是底层浮点机制的穷尽
分类**。一种底层机制也可能同时改变两项。

## 3. 三阶段测量负责定位

对同一个 matched state，分别记录：

```text
算子或语义区域的 output difference
                ↓
实际 parameter-gradient difference
                ↓
目标 optimizer 的 parameter-update difference
```

它们回答：

- local 已有平均方向：source boundary 已经留下平均结构；
- local 近似抵消、gradient 出现方向：backward transport 或 response 改变了结构；
- gradient 有方向、update 变弱：optimizer 抑制了方向；
- 严格 `+e/-e` 的 update 仍不互为相反数：downstream response 不保持镜像抵消。

真正写入训练参数的是 update，因此 update 是主要正确性端点；local 和 gradient
用于定位方向在哪里形成、增强或消失。

## 4. 每个阶段统一报告什么

对第 `i` 个 matched state 和某一阶段，定义：

\[
u_i=Y_i^C-Y_i^R,
\qquad
r_i=Y_i^R.
\]

每层至少报告三类量：

1. **总误差能量**：`E||u_i||²`；
2. **可复现平均效应**：`||E[u_i]||`，以及相对正常 training update 的比例；
3. **不确定性**：效应量的置信区间，而不是只给一个是否过线的标签。

为了区分“只是把正常信号整体放大/缩小”和“出现新的方向”，再写成：

\[
u_i^{\parallel}
=
\frac{\langle u_i,r_i\rangle}{\|r_i\|^2}r_i,
\qquad
u_i^{\perp}=u_i-u_i^{\parallel}.
\]

这里 `parallel` 表示相对当前 repair signal 的缩放，`perpendicular` 表示不能由
一个标量缩放解释的 residual。三阶段使用相同分解，但 local、gradient 和 update
上的缩放含义不同，不能把三个数当成同一个物理参数。

当我们需要对一组随机状态作总体结论时，用 calibration states 冻结平均方向，
再在 untouched confirmation states 上报告带符号的 held-out effect 和置信区间。
同一训练 run 内连续状态全部算一个 training unit，不能冒充独立 runs。v2 要求
calibration 与 confirmation 各至少 8 个互不重叠的独立 training units；不满足时
只报告固定测试集合上的结果，不生成总体置信区间。对多个阶段和多个案例同时作
判断时，预先声明检验组并用 Holm 控制误报；效应量和置信区间仍是主结果。

v2 同时修正了大向量摘要：不再按固定周期折叠坐标，而是保存带 seed 的 value-blind
CountSketch。headline 大向量结果还需要额外 seeds 或完整向量 Gram 复核。完整规则见
[`training_bias_profile_v2.md`](training_bias_profile_v2.md)。

## 5. 训练数值等价性

三个方向指标只能说明差异表现为什么，不能排除其他方向。当前证书因此增加一个必须
通过、覆盖全部坐标的 update 总体检查：

\[
s_{\mathrm{full}}
=
\sqrt{
\frac{\sum_i\|u_i\|^2}
{\sum_i\|r_i\|^2}
}.
\]

只有当这个比例低于工程范围，而且固定方向、相对正常 update 的缩放、去掉缩放后的
剩余方向三项区间也全部落入各自范围，才称 candidate 与 repair 在这组固定输入上
`FIXED_SUITE_UPDATE_EQUIVALENT`。只有保存完整向量并验证 update difference 逐位为零，
才单独记为 `EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE`。

否则区分：

- `MATERIAL_EFFECT`：效应区间超过工程范围；
- `FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN`：三个方向指标可能很小，但完整 update
  difference 的总体比例已经超过范围；
- `INCONCLUSIVE`：现有状态不足以区分零效应与工程上重要的效应；
- `ABSTAIN`：repair、状态绑定或参数可达性不成立。

新增的总体 update 范围修复了旧三方向证书的方向盲区：后 16 个输入即使出现与
前 16 个输入方向垂直的新差异，也不能再被错误放行。五类完整向量测试已经通过。
完整 update 的 `1%` 范围是在 16 项结果揭示后加入的保守修正，来源是已有最大缩放
范围，不冒充事前冻结。当前标签只适用于声明模型、参数位置、optimizer 和这组固定
输入，不是随机训练状态总体、完整训练质量或所有 LLM 共享的安全结论。大向量没有
保留原向量，而是取三个事先固定的全坐标随机摘要中最保守的结果；因此不能把摘要为
零写成逐位相同。本次验证没有
预先声明训练后果门槛，因此机器记录明确写 `NOT_DECLARED`。

## 6. Orbit mean：只用于 reduction 类 source predictor

对 reduction、summation 或 reassociation 一类实现，预先声明一组数学等价的
schedule `pi ~ nu`：

\[
m_{\mathrm{orb}}(a;\nu)
=
\mathbb E_{\pi\sim\nu}
\left[\operatorname{fl}_{\pi}(a)-y^\star(a)\right].
\]

它检查：在一组等价 reduction schedules 上，source residual 的平均是否仍为
非零。它是一个 **source-side candidate predictor**，边界如下：

- 只用于 reduction / summation / reassociation，不覆盖一般 backward、量化、
  saved-state 或 optimizer 机制；
- `m_orb != 0` 不表示每一种 schedule 都同号，也不表示不存在更好的 schedule；
- 它通常需要运行多个等价 schedule，不是“一次 forward 的免费静态判断”；
- 它不能替代 local → gradient → update 和训练后果检查。

当前已完成一个更直接的 Liger 检查：两个实现都使用 FP32，只改变 `dW` 分块结果的
加法顺序。前 16 个输入确定预测方向，后 16 个输入中 15 个沿同一方向；gradient 和
AdamW update 的结果在整体校正后仍成立。这个效应非常小，只能说明“同一精度下的
求和顺序也可能留下可重复方向”，不能据此声称存在长程 loss 后果，也不能升级成
通用静态 Oracle。

## 7. 严格正负响应实验

对 saved-P、SiLU 等 response 问题，在同一 state 上构造：

\[
u_i^{\mathrm{resp}}
=
\tfrac12\left(Y_i^+ + Y_i^- - 2Y_i^0\right).
\]

如果它的平均不为零，说明严格相反的输入经过 backward / optimizer 后仍留下同向
剩余。它与普通 candidate-repair difference 使用同一统计代码，但必须保留不同
的 `contrast_id`，不能混进同一个 prevalence 分母。

## 8. 短程筛查与长程后果

16/32 步只用于便宜排序和机制定位。描述性方向分数为：

\[
A_X(T)=
\frac{\left\|\sum_{t=1}^{T}X_t\right\|}
{\sqrt{\sum_{t=1}^{T}\|X_t\|^2}}.
\]

它必须和本行的随机符号基线、prefix 和 rolling windows 一起解释。`A=1` 或
`A=2` 都不是所有训练任务共享的安全常数。

长程配对训练还要拆开：

\[
D_{t+1}-D_t=L_t+B_t,
\]

其中 `L_t` 是同状态下目标实现的直接 update difference，`B_t` 是两条训练轨迹
分开后由 weights 和 optimizer state 不同带来的反馈。最终参数或 loss 分叉不能
反推目标算子每一步都在同向推动。

4096 步结果是当前的长程 consequence evidence，不是 loss 收敛证明，也不是完整
全参数训练的总体推断。

## 9. 当前证据

### 覆盖

四个核心模型、三个序列长度完成了全量 F+B 与首轮数值检查：

- 466,419 次 eager 调用；
- 70,171 次被测编译实现调用；
- 186,807 个绑定真实 forward/backward 的计算单元；
- 1,562/1,562 个具体 output positions 得到首轮处置。

这不是 1,562 个 32 步或 4096 步训练实验。

### 已有代表结果

Training Bias Profile v2 已对五个开发案例完成统一重采。每例使用 32 个冻结且不重叠
的输入窗口，前 16 个确定方向、后 16 个检查；每个窗口恢复同一 checkpoint 和零
AdamW moments。主要检验是 `5 cases × 3 update branches = 15` 项，local/gradient
的 30 项只用于解释，两组分别使用 Holm 校正。大向量要求三个预先声明的摘要方向
一致。输入银行并非已证明的随机总体，因此区间只描述这组冻结窗口。

| 案例 | local / gradient 中首先看到什么 | cold-start AdamW update | 4096 步对照 |
|---|---|---|---|
| Liger fused CE | local 已有剩余共同方向，gradient 保留 | 剩余方向 `0.0151%`，区间 `[0.00614%, 0.0240%]` | 固定方向稳健，且有 paired loss split |
| Phi `lm_head dX` | local 较弱，backward 将剩余方向放大 | 正常 update 缩小 `0.1506%`，区间 `[0.0966%, 0.2046%]` | 固定方向稳健，且有 paired loss split |
| Qwen `lm_head dX` | gradient 出现剩余共同方向 | 三个分支均未确认 | warm/long-run 协议中固定方向稳健 |
| Qwen `v_proj` | local、gradient 均未确认 | 正常 update 缩小 `6.04%`，区间 `[3.44%, 8.64%]` | 固定方向不稳健，但有 loss split |
| Mamba `in_proj` | local 只有极小缩放，gradient 未确认 | 正常 update 缩小 `3.12%`，区间 `[2.86%, 3.37%]` | 固定方向不稳健，但有 loss split |

这五例不是同一种低秩偏差：Liger 是去掉正常 update 缩放后仍存在的共同方向；Phi、
Qwen `v_proj` 和 Mamba 的 update 主要表现为随各自正常 update 方向旋转的稳定缩小；
Qwen `lm_head` 则显示 gradient 中的方向可以在当前 AdamW 条件下消失。

短窗口与 4096 步并不矛盾，它们问的是不同问题。Qwen `v_proj` 和 Mamba 的稳定缩小
不要求固定参数方向跨状态相同；Qwen `lm_head` 的 cold-start 与 warm/long-run 条件也
不同。任何案例都不能脱离 optimizer state、输入集合和 horizon 获得永久标签。

完整结果、区间、原始 p 值、Holm 校正值和三 seed 审计见
[`five_case_training_bias_profile_v2.md`](five_case_training_bias_profile_v2.md)。历史 v1
结果继续保留在 `unified_measurement_round.md`、`extended_unified_profiles.md` 和
`three_mechanism_profiles.md` 中，但不覆盖 v2 结果。

另外两类证据仍保持不同 contrast：saved-P / SiLU 使用严格 `+delta/-delta` response
remainder；DeepSeek `dV` 是 gradient-stage、stateless SGD 的冻结未见状态确认。它们
不能混进上述五例 AdamW update 检验组。

### 方法冻结后的新案例

Training Bias Profile v2 冻结后又运行了两个独立批次。第一批四项保留了一个确认项、
一个完整负例和两个无法判断；第二批预先固定一个 attention projection 项并完成确认。

| 新案例 | cold-start AdamW update | 配对训练后果 |
|---|---|---|
| DeepSeek normalization backward | 正常 update 缩小 `13.68%`，区间 `[11.55%, 15.82%]` | 第 1 步 loss gap `0.00950384` |
| DeepSeek attention projection backward | 正常 update 缩小 `10.69%`，区间 `[9.37%, 12.02%]` | 第 1 步 loss gap `0.00934792` |
| Phi loss / CE backward | 三个 update 分支均未确认 | 未升级后果实验 |
| Mamba recurrent backward | 环境无法完成精确编译，无法判断 | 未运行 |
| Qwen attention-state backward | 冻结 source-binding 文件损坏，无法判断 | 未运行 |

两个 DeepSeek 结果表明新方法能在不同 backward 家族和参数位置发现随正常 update
旋转的稳定缩小；Phi 负例表明它不会把任何非零 residual 都升级。两个 loss 实验达到
事前停止条件后在第 1 步结束，因此只证明轨迹不再相同，不是 4096 步持续性或最终
训练质量结论。详细证据见
[`prospective_training_bias_profiles.md`](prospective_training_bias_profiles.md)。

### 后续状态与训练后果检查

为了判断 cold-start 结果能否代表后续训练，我们对两个 DeepSeek 正例和一个 Phi
负例固定了五种设置。两个 DeepSeek 位置在 moments 从零开始时分别缩小正常 update
约 `10.69%` 和 `13.68%`；真实推进 8 或 32 步后都降到不足 `1%`。在同一 warm
checkpoint 重置 moments，大效应分别恢复到 `9.03%` 和 `15.13%`。Phi 在五种设置
下都没有确认效应。这说明当前效应明显依赖 AdamW 历史状态，不能给 kernel 一个脱离
checkpoint 和 optimizer state 的永久标签。完整 45 项整体校正结果见
`results/property/optimizer_condition_benchmark_v1/summary.json`。

三项训练位置还各自使用四组互不重叠的 32 步输入流做了配对训练。三项的实际参数
分离和反馈方向均在 `4/4` 输入流中超过各自随机抵消范围，但同状态直接作用均为
`0/4`。稳定窗口的平均 loss gap 在三个案例中都跨过零；因此当前结果支持“轨迹和
参数不再相同，且分离主要由反馈维持”，不支持稳定的 loss 变好或变坏。这里四组输入
流来自同一个预训练 checkpoint，也不是四次完整预训练。机器结果见
`results/property/independent_consequence_v1/summary.json`。

我们还直接检查了已测参数偏移是否沿 loss 敏感方向。Liger 的测得方向在原尺度上使
平均评估 loss 增加 `1.1998e-4`，绝对变化约为同长度随机方向中位数的 `1039` 倍；
Phi 的原尺度变化为 `-3.7365e-6`，约为随机方向中位数的 `9.55` 倍，并在放大到
`1000` 倍后改变符号。它们说明测得方向不是任意随机方向，但只是保存参数附近的
局部敏感性检查，不能把直线放大当作未来训练轨迹。

最后，Gemma 4 的两个历史冻结位置已在与原记录一致的运行环境中接入统一方法：一个
位置的 gradient 和 AdamW update 与 repair 完全相同，另一个位置的所有区间均跨零。
这是跨模型、新实现方法接入的两个负例，不是新选择的前瞻发现集合。机器结果见
`results/property/generalization_benchmark_v1/gemma4_method_bridge_result.json`。

统一的 16 项冻结验证也已完成：15 项得到有效测量，Mamba seq256 因实际 backward 图
与冻结记录不一致而保留为无法判断。48 项整体 Holm 校正后，9 项确认了正常 AdamW
update 被稳定缩小，覆盖 DeepSeek、Qwen 和 Phi。修正后的证书给出：3 项同时通过总体
update 范围和三个方向范围；其中两个 Mamba seq128 案例在三个预先固定的全坐标摘要
中均为零，不能进一步写成逐位相同。另有 3 项总体 update 比例超过 `1%`，不能签为
等价；1 项无法判断。原来三方向证书中的 Mamba seq64 被重新归类，因为三个摘要中的
最大比例为 `2.142%`。两个 Phi 案例虽然方向区间仍不足以确认稳定方向，但最大比例约
为 `22.33%` 和 `21.74%`，同样不能签为等价。
确认的缩小效应从 `1.289%` 到 `15.818%`，并未被这次修正推翻。完整结果见
`results/property/generalization_benchmark_v1/summary.json` 与
`results/property/generalization_benchmark_v1/equivalence_v2.json`；旧
`equivalence.json` 仅保留为修正前记录。

### 当前长程计数

当前机器审计包含 23 个唯一主矩阵 case ID、301 条逐行记录：

- 43 条记录同时有 long-run bias 证据与 paired loss split；
- 其中 3 条是已导出后半程窗口的 direct cases；
- 8 条有整段 long-run direct evidence，但 late-window statistics 尚未单独导出；
- 32 条为 feedback-sustained cases；
- 目前只有 4 条记录具有显式 late rolling-window confirmation；
- 另有 5 条出现 paired loss split，但 direct bias 未通过 long-run gate；
- 更宽的 `outcome_relevant` 口径为 105 条，其中包含 57 条尚未测量 4096-step
  persistence 的历史 coherent candidates，不能全部称为 final persistent cases；
- 45 条为 unresolved 或 abstain，其余还包括不适用与未完成 persistence 复核的记录。

这些数字来自
`results/property/declared_persistent_4096/all_bias_case_audit.json`。人类可读逐行表见
[`all_bias_long_horizon_audit.md`](all_bias_long_horizon_audit.md)；若文字表正在更新，
以机器 JSON 的字段和每行 artifact 为准。

## 10. 下一阶段实验

1. 五个开发案例的 v2 重采已经完成。停止围绕它们调整分支、阈值或校正组；任何后续
   解释都必须从冻结 JSON 生成。
2. 方法冻结后的前两批和统一 16 项验证均已完成。Qwen 原冻结案例已经用同一冻结 ID
   恢复；Mamba seq64/128 已完成，seq256 因 backward 图不一致保留无法判断。后续若
   扩展，应冻结跨模型、checkpoint 和 warm optimizer state 的新输入，不能用替换失败
   项的方式提高成功率。
3. 新 pool 不以“找到新正例”为成功条件。positive、negative 和无法判断都完整报告；
   若全为 negative，只报告升级率和无法判断率，不补造 recall 或泛化准确率。
4. 对 held-out 中确认的项目，才做 prediction-first 干预：先根据 local → gradient →
   update profile 写下修复应改变哪一层，再运行修复，不允许看到结果后换解释。
5. 三个方向范围在新的 16 项验证揭示前冻结；完整 update 的 `1%` 范围是在揭示后为
   修复方向盲区而加入。修正后的证书只签发固定输入集合上的 update 结论，并已通过
   完整 Gram 路径的方向漂移、巨大零均值差异、剩余方向、repair 能量不均和范围边界
   测试。大向量取三个冻结随机摘要中的最大结果，不声称逐位相同；它也不签发随机
   训练状态总体或完整训练质量等价。
6. 现有 4096 步继续只负责 paired consequence；若要声称跨训练 run 的 material loss
   影响，必须以独立 training runs 为统计单位。

## 11. 当前能说与不能说

当前可以说：

> 对具体 LLM training implementation，误差大小与训练提交后的平均方向是两类
> 不同信息。matched local/backward/optimizer measurement 能定位方向在哪里形成、
> 被压制或留下。五个开发案例还表明，系统效应既可以是在固定参数坐标中留下共同
> 方向，也可以是反复缩小会随输入变化的正常 update。短程检查负责描述结构，长程
> 配对实验负责检查后果。

当前不能说：

- 已经得到对所有实现都适用的静态 property 或安全 Oracle；
- residual direction 在多数算子上普遍存在；
- orbit mean 能预测 reduction 之外的机制；
- 16/32 步未升级等于安全；
- 4096 步方向等于 loss 已收敛到不同值；
- 单条训练轨迹等价于独立 runs 的总体结论。
- v1 的逐状态区间等价于 v2 的独立 training-unit 区间。
- 五个开发案例中 4/5 的 update 确认率代表自然发生比例或 held-out 泛化能力；
- Qwen `v_proj` 与 Mamba 的稳定 update 缩小等于 4096 步固定方向持久；
- 把当前冻结的工程范围解释成所有模型、optimizer 和训练任务共享的安全阈值。
