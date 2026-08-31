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
同一训练 run 内连续状态不能冒充独立 runs。对多个阶段和多个案例同时作显著性
判断时，预先声明检验组并用 Holm 控制误报；效应量和置信区间仍是主结果。

## 5. 训练数值等价性

在声明协议下，只有当 update effect 的置信区间整体落入预先声明的工程范围，
且声明的训练后果检查没有失败，才称 candidate 与 repair 在该协议下
`TRAINING_EQUIVALENT`。

否则区分：

- `DETECTABLE_BUT_SMALL`：方向可复现，但低于工程范围；
- `MATERIAL_EFFECT`：效应区间超过工程范围；
- `INCONCLUSIVE`：现有状态不足以区分零效应与工程上重要的效应；
- `ABSTAIN`：repair、状态绑定或参数可达性不成立。

这是一套待用统一统计实验闭合的操作性定义，不是当前已经验证过所有模型的通用
安全 Oracle。

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

当前预注册的 Liger 实验是：冻结一组合法 chunk/reduction schedules，先计算
BF16 accumulator 的 `m_orb`，事前预测它与 local-stage mean effect 方向对齐；
再换成 FP32 accumulator，预测二者同步下降。成功后只支持这一 source mechanism
的预测价值，不升级成通用静态 Oracle。

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

- **Phi `lm_head dX`**：现有统一 formation artifact 是
  `LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED`。同一 cold-start
  AdamW 协议中，deterministic BF16 的短程方向超过自身随机基线；四个真实
  stochastic-rounding 重复回到各自随机范围，前三个的误差能量与 natural 基本
  相当。
- **Liger fused CE**：已经用与 Phi 相同的 16 calibration + 16 confirmation、
  cold-start AdamW 协议重测。local、gradient 和 AdamW update 都留下可复现的
  residual direction；AdamW update 的平均方向效应为正常 update RMS 的 0.0327%，
  95% 区间为 0.0266%–0.0401%。candidate 是 BF16 chunk accumulation，repair 是
  保持 BF16 接口的 FP32 accumulator，因此 accumulator precision 的 matched
  intervention 已闭合。已有 4096 步直接作用和 loss 分叉。
- **统一 Phi / Liger 结果**：两个 anchor 的三阶段效应量、区间和 18 项 Holm 校正
  见 [`unified_measurement_round.md`](unified_measurement_round.md)。Phi 的同协议
  随机舍入干预中，natural 方向分数为 1.02971，四个随机舍入重复均回到约 1.000；
  前三个重复的更新误差能量仍为 natural 的 99.8%–101.6%。
- **Qwen `lm_head dX`**：统一 16+16 结果显示 local、gradient 和 AdamW update
  的平均方向均在后半状态复现；AdamW update 效应只有正常 update RMS 的
  0.000400%，95% 区间为 0.000215%–0.000619%。因此 AdamW 是强烈压低，而不是
  完全消除。warm-state 4096 步中也有直接方向；optimizer verdict 仍必须和训练
  状态一起描述。
- **Qwen `v_proj`**：local、gradient 和 AdamW update 均未通过统一 16+16 确认。
  它是“局部差异不等于跨状态平均方向”的负例。
- **Mamba `in_proj`**：local 未确认平均方向；gradient 相对正常 gradient 平均缩小
  0.967%，95% 区间为 0.383%–1.786%，但 AdamW update 中消失。这提供了
  backward 产生、optimizer 消除的非 Transformer 例子。
- **saved-P / SiLU**：saved-P 的严格 `+delta/-delta` response remainder 在后半
  状态复现，效应为正常 odd response 尺度的 2.90%，95% 区间为 1.17%–4.93%。
  SiLU 的逐状态 response 不是严格镜像，但后半状态方向与前半相反，未通过统一
  方向确认；它保留为 cold-start / phase-sensitive 案例。saved-P 的直接方向未通过
  4096 步门；SiLU 的长程结果主要由 feedback 维持。
- **三类统一补测**：Gemma normalization、Llama softmax backward 和 Llama
  attention BMM 已按相同的 16 calibration + 16 confirmation 协议完成 local、
  gradient、cold-start AdamW update 三层测量。Gemma 的 local 方向到 gradient
  消失；softmax 的 gradient 信号在 AdamW 后消失；BMM 的固定方向不稳定，但
  local repair-relative scaling 通过完整 Holm 校正。这说明现有案例不只包含固定
  低秩方向。完整数字见 [`three_mechanism_profiles.md`](three_mechanism_profiles.md)。
- **五个补充案例**：Qwen `lm_head`、Qwen `v_proj`、Mamba `in_proj`、saved-P 和
  SiLU 已完成统一分半统计与两个预先冻结的 Holm 检验组。完整数字和解释见
  [`extended_unified_profiles.md`](extended_unified_profiles.md)。
- **DeepSeek layer-35 attention `dV`**：已有事前冻结的 32-state unseen
  confirmation。candidate 相对 repair 的 `v_proj.weight` gradient 在同状态正常
  gradient 方向上平均缩小 0.370%，95% 区间为 0.059%–0.664%；3 个候选中只有
  这一项复现，2 个控制项均未误报。它是 gradient-stage、stateless SGD 映射下的
  证据，不能和上面的 AdamW update 数字混为同一协议。

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

1. 合成数据自检已经覆盖固定平均方向、相对正常 update 的缩放、零均值大方差、
   正负交替、重尾、稀疏方向和低 repair energy；独立 synthetic states 下的结果
   通过 go/no-go gate。下一步补相关 state cluster 和多 run 的区间校准。
2. 继续统一保存 joint `G_uu`、`G_rr`、`G_ur`，使 local、gradient、update 都能由
   同一份 artifact 重算效应量、held-out direction 和置信区间。
3. Phi、Liger、Qwen `lm_head dX`、Qwen `v_proj`、Mamba `in_proj`、normalization、
   softmax backward 和 attention BMM 已完成统一补测；saved-P / SiLU 也已按独立
   response contrast 重算。下一步不再为解释旧案例继续扩张，而是在完全冻结的方法
   下运行一个新的 held-out implementation pool。
4. DeepSeek `dV` 的冻结 unseen-state gradient confirmation 已完成。今后的 held-out
   pool 继续要求先冻结等价性范围、检验组和 Holm 规则，再揭示结果。
5. Phi stochastic rounding 和 Liger accumulator 已完成同目标协议的 matched
   intervention；saved-P 已形成稳定 response-side 证据，后续机制干预可作为增强项，
   不再阻塞当前主线。
6. 现有 4096 步结果只负责 consequence；需要更强 loss 结论时，以独立 training
   runs 为统计单位。

## 11. 当前能说与不能说

当前可以说：

> 对具体 LLM training implementation，误差大小与训练提交后的平均方向是两类
> 不同信息。matched local/backward/optimizer measurement 能定位方向在哪里形成、
> 被压制或留下；短程筛查可排序，长程配对实验负责检查后果。

当前不能说：

- 已经得到对所有实现都适用的静态 property 或安全 Oracle；
- residual direction 在多数算子上普遍存在；
- orbit mean 能预测 reduction 之外的机制；
- 16/32 步未升级等于安全；
- 4096 步方向等于 loss 已收敛到不同值；
- 单条训练轨迹等价于独立 runs 的总体结论。
