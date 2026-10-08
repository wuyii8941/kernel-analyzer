# 本质错误一轮·2b 记录 第二版（收束包 v2.0 之后，2026-10-08）

本版取代 `docs/essential_bugs_phase2b_record_20261007.md` 中的**结论性文字**；该历史文件不改（任务书第 4 节）。变化来自四处：
(1) 规格 v0.2 入库后，F 与模式 B 的 FR 对 15 个第一档家族中有规格的全部开放；(2) 数值契约 v3：精度不变性代替「条件数」类说法，三条
「无法判断」规则（S₀ = 2、N₀ = 64、n_min = 16）；(3) 契约 v2 的五类输入；(4) 第 2 行复审表（`docs/readjudication_20261008.md`）对九个模式
命中的改写。工具版本 2.3（`docs/detector_changelog.md`）。协议 `docs/protocol_closure_v3_20261008.md`（偏离 10 条，见其第 10 节）。

逐家族的四组、四栏与成本：`docs/closure_family_tables_20261008.md`（数据 `results/closure/family_tables.json`）。三类组合：
`docs/composition_checklists_20261008.md`。保证链：`results/closure/chain_table.json`、`docs/rules_to_code_20261008.md`。

## 1 三种结论的写法

每个条目分开写「差异」（大小、方向、平均作用是否非零）、「缺陷」（是否违反契约：文档明文 / 声明解释 / 契约允许的数值行为）与「重要性」
（本轮不判，留给阶段 B 的尺子）。「在误差界内」「eager 也一样」只描述误差大小，不回答是否违反；契约没有规定的行为记「契约外，待审阅」。

## 2 四组的结果（第一档 15 个家族；计数均带分母）

**F（K − f，不经 K_R）。** 有规格的 12 个家族（matmul_linear、reductions、activations、gather_layout、embedding、attention、packing、rope、
normalization、optimizers、schedulers、clip_amp）全部运行；training_program 的 F 在 2b 已用 `spec_accumulation` 执行；checkpoint 没有整程序
的 f（由 P 的重算等价性质检查）。float32 / float64 候选的超出 τ（τ₃₂ = 2⁻¹²、τ₆₄ = 10⁻⁹）：
- matmul_linear、reductions、gather_layout、rope、packing、moe、attention、clip_amp、schedulers（T_max 之前）：**0 超出**（例如 attention 每个 float32
  候选 0 / 322944；matmul 0 / 20700 输出、0 / 43956 dA、0 / 71820 dB）。
- activations：0 / 14850；swiglu / geglu 的 huge 条件中实数乘积超过 FLT_MAX，K = ±inf（第 4 栏，595 个元素）；ReLU 在 0 处的梯度全部落在文档
  允许的子梯度集合内（0 / 2970 超出）。
- embedding：float32 候选 0 超出；float64 eager 的 max_norm 条件 891 / 3552（out）、340 / 960（weight_after）超出 τ₆₄——实现按 (norm + 1e-7)
  重归一化（除以 norm + 1e-7 后残差 ≤ 7.3e-17），属契约 v2 的 EMB-A1，**第 2 栏**（声明解释下的差异）。weight_after 的作用范围（E-D2）待审阅。
- normalization：float32 的 huge_offset 条件超出 τ₃₂（eager CPU 247 / 5400、eager CUDA 250 / 5400、Inductor 266 / 5400，均在
  batch_norm_train / group_norm / layer_norm 的 huge_offset）；精度不变性判为「无语义元素」（138 个条件）——**差异：数值**（偏离随精度缩小）；
  契约未规定精度要求，是否缺陷**契约外，待审阅**。
- optimizers：全部候选的参数轨迹 0 超出；SGD maximize 条件的 momentum_buffer 306 / 3672 超出——等于规格 b 取负（≤ 1.5e-16），规格 O-D2 与
  2.10 文档框不符（SPEC-ISSUE-1），**条款待审阅**。
- bf16 候选只记录不判定。

**精度不变性（黑箱语义 / 数值分类）。** 同设备的 float32 / float64 配对（本轮补跑了 attention、optimizers、packing、rope、moe 的配对）：
全部家族「无语义元素」，例外两处且都已归属——embedding max_norm 的 24 个条件（EMB-A1 的常数 1e-7，第 2 栏）、SGD maximize 的 12 条缓冲记录
（O-D2 待审阅）。S3 index：467 个条件无语义元素，1 个异常（float32 恰好正确、float64 偏离 1.6e-12 相对，超出 float64 求值噪声，已复核并记录）。

**P。** 2b 预注册性质全部保留（数据不变）。本轮补充 gradcheck（float64 eager）与 `torch.func.jvp` 对 VJP 的点积（全部候选）：
matmul、gather、normalization、rope（参照实现）、moe 全部通过；reductions 与 activations 的失败只出现在无定义或不可微处（BASE-A1 的单元素
方差、−∞ 行、ReLU 在 0 处）与 |x| ≈ 1e30 的有限差分不适用处；activations 的 gelu_tanh huge 条件 float32 / bf16 的 jvp 出现非有限值，即已登记的
**B023**（第 4 栏）；embedding 的 padding_idx 与 scale_grad_by_freq 按文档「梯度不是导数」，导数检查不适用。前向 AD 不支持的候选（SDPA
memory-efficient、flex eager、embedding_bag）记「不支持」。D1 用 2b 的 amax / amin 并列检查（0 / 21 违反）；D2、D3 的对象不在本档家族中。

**FR（模式 B：K_R 对 f；e_num 与 e_sem）。** Triton 候选：Inductor（基础四族、embedding、normalization）、flex（attention、packing）、编译的
optimizer step、编译的 MoE。结论分三类陈述：
- e_num（K − K_R）：有限 K 的元素全部在 τ₃₂ 内，除 normalization 的 huge_offset（Inductor float32，266 个元素，与 F 的超出同一批）。
- e_sem 只在**纯 Triton**输出上作语义判据（输出不读 cuBLAS / ATen 在窗口内产生的值）；读了这些值的输出（optimizers 全部、moe、embedding
  max_norm、组合 C1 的 y）e_sem 为混合，不出语义结论，改由 F 判。Inductor 注意力与 matmul 的主输出由 cuBLAS 写出，未建立。
- 纯 Triton 输出上 K_R 与 f_r 的包围不相交（差异已确证）的元素全部来自**编译期 float32 常数**：Inductor GELU 的 √(2/π)、0.044715、1/√2——
  把这些常数换成 TTIR 中的 float32 值后，不相交元素 1780 → 0；flex 的 `RCP_LN2 = 1.44269504`（生成的 kernel 源码，按机制归因）；
  normalization 的 eps 与 momentum 按 float32 持有后 y 的不相交元素降为 0；batch_norm_train 的 running mean / var 仍有 144 个元素不相交——
  生成代码中 1 − momentum 与无偏修正 n/(n−1) 是编译期折叠后写入的常数（0.9、1.1111111111111112，按 float32 持有），按机制归因。
  纯输出上的最大相对差 ≤ 2.0e-8。
  这是「差异」；契约 v2 没有关于编译期常数精度的条款，**契约外，待审阅**。

**E。** 2b 的 E 全部保留（`docs/closure_family_tables_20261008.md` 逐候选列出）。

## 3 状态序列与调用场景
2b 的 A → B → A 60 条序列：每次调用与 eager 的比较 0 违反；逐位比较作为信息记录（LayerNorm 自动动态重编译后 2 条序列差几个 ulp）。
本轮 C3：AdamW 序列的 state_dict 重载后一步与不中断的一步在 9 个输出上逐位相同。

## 4 改写的 2b 结论（逐条见 `docs/readjudication_20261008.md` 第 4–8 行）
- normalization（原「数值（条件数）」）：没有语义偏离的元素（精度不变性，138 / 138 条件）；超出 τ₃₂ 的大小见上；是否缺陷：契约外，待审阅。
- G7 的 gelu_tanh_bf16（原「来自正确舍入本身，不是实现的缺陷」）：差异——平均作用非零，在独立 seed 上复现（见第 5 节）；机制：输出 99.99%
  等于精确值就近舍入的 bf16 值，|x| ≳ 3 时 gelu(x) = x − δ 被舍回 x；缺陷：契约未规定此类数值行为，**契约外，待审阅**；重要性：留给阶段 B。
- 「FR 把条件数与 bf16 的偏离归到 e_num」改为：这些偏离在 FR 中落在 e_num（K − K_R），e_sem 不含它们。

## 5 数值作用流（协议 v3 附录 A，提交 84fe39e 在测量之前）
20 个低精度 Inductor 程序，模式 A，32 个开发 + 64 个确认单位（seed 2000–2095，此前未用），工具 2.3。参照全部建立（20 / 20，完整比例 1.0）；
所有规则 n = 64、|偏度| ≤ 0.92，「无法判断」未触发。至少一条方向规则「平均作用非零」：9 / 20——batch_norm_train_bf16（R2 +）、
sum_long_bf16（R1 −）、mean_long_fp16（R1 +）、cumsum_bf16（R1 +）、gelu_tanh_bf16（R1 / R2 / R3 −，R5 +）、geglu_fp16（R5 +）、
attention_softmax_bf16（R5 −）、rope_rotate_half_bf16（R2 / R3 +）、clip_scale_bf16（R5 +）；其余 11 个四条规则均「未确认」。相对 RMS：bf16 约
1.5–1.7·10⁻³，fp16 约 2·10⁻⁴。与 2b G7（seed 0–95）对照：gelu_tanh_bf16 复现（同号）；sum_long_bf16 在 2b 为 R2 / R3、在 2b 的第二组 seed
上不复现、本轮为 R1——不同规则、不同 seed 组之间的结果不一致，不下总体结论。这些是实现的数值作用，不是缺陷判定。数据
`results/closure/numerical_stream.json`。

## 6 发现
- **SPEC-ISSUE-1**（规格问题，交审阅方）：`spec_optimizers` O-D2 的 maximize 位置与 2.10 文档框不符（`docs/spec_issues_phase2.md`）。
- **检测器 2.3**（本项目工具）：上游非 Triton 来源检测的两处漏判（同字节陈旧缓冲区；启动内原地修改的输入），已修正并加 4 项回归测试。
- 实现侧：本轮没有新的上游问题；B023 被 P 的 jvp 对 VJP 检查再次看到。没有新的上游草稿。

## 7 不写入的说法
本报告不把 G6 的参照复用率当作对全部组合的覆盖（G6 是 10 个程序上的复用测量）；不以发现的数量作为完成标准。
