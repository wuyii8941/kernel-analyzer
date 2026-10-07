# 本质错误一轮·2b 运行记录（逐家族追加）

依据：协议 v2 第 3 节（候选清单 3.1、覆盖矩阵 3.2，均在 2b 运行之前提交）、第 6 节（预算）、第 7 节（状态）、第 9 节（偏离）。
F 与模式 B 的 FR 在各家族的规格入库前关闭（训练程序层的累加窗口除外）；E、P、状态序列、模式 A 的 FR 不等待规格。检测器 `detector-v2.2`。

## 家族状态

| 家族 | E | P（预注册） | 状态序列 | FR | F | 预算（试跑估算 / 实际） |
|---|---|---|---|---|---|---|
| optimizers | 已执行且可裁决 | 已执行且可裁决 | 已执行且可裁决 | 模式 A：已执行且可裁决（Adafactor 未建立） | 未运行（规格未交付） | GPU 约 2.5 分钟、CPU 约 10 秒；人工 736 行、约 1.5 小时 |
| attention | 已执行且可裁决（float32）；bf16 只记录 | 已执行且可裁决 | 不适用 | 模式 A：flex 已执行且可裁决；Inductor 手写注意力的输出由 cuBLAS 写出，未建立 | 未运行（规格未交付） | GPU 约 13 分钟（其中 FR 10 分钟）；人工约 560 行、约 2 小时 |
| packing | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 未运行（flex 的 kernel 与 attention 家族相同，FR 已在那里执行） | 未运行（规格未交付） | GPU 约 2 分钟；人工 258 行、约 0.5 小时 |
| training_program | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 不适用 | **已执行且可裁决**（`spec_accumulation.py`） | CPU 约 27 分钟（6 个候选并行约 5.5 分钟）；人工 358 行、约 1.5 小时 |
| normalization | 已执行且可裁决（float32）；bf16 只记录 | 已执行且可裁决 | 不适用 | 模式 A：Inductor float32 / bf16 已执行且可裁决；Liger 未运行（liger 环境没有求值依赖） | 未运行（规格未交付） | GPU 约 1 分钟（FR 另计）；人工约 480 行、约 2 小时 |
| embedding | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 未运行（排在基础算子之后） | 未运行（规格未交付） | GPU 约 1 分钟；人工 334 行、约 1 小时 |
| schedulers | 已执行且可裁决（nightly 对 2.10） | 已执行且可裁决 | 不适用 | 不适用 | 未运行（规格未交付） | CPU 数秒；人工 225 行、约 0.5 小时 |
| clip_amp | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 不适用 | 未运行（规格未交付） | 数秒；人工 285 行、约 0.5 小时 |
| matmul_linear / reductions / activations / gather_layout | 已执行且可裁决（float32）；bf16 只记录 | 已执行且可裁决 | 不适用 | 模式 A：Inductor float32 已执行且可裁决（matmul 输出由 cuBLAS 写出，未建立） | 未运行（规格未交付） | GPU 约 6 分钟（FR 另约 10 分钟）；人工约 620 行（四个家族共用）、约 2.5 小时 |
| checkpoint / rope / moe | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 未运行（moe 的编译候选只作 E；rope 的 Liger / Unsloth 只作 E 与 P） | 未运行（规格未交付） | GPU 约 2 分钟；人工约 400 行、约 1.5 小时 |
| 调用场景 A → B → A（编译候选） | 已执行且可裁决 | — | — | — | — | GPU 约 1 分钟；人工约 200 行 |
| G6 陌生组合（10 个程序） | 已执行且可裁决 | — | — | 模式 A：已执行且可裁决 | — | 工具时间 37 秒；人工约 250 行（含 G7） |
| G8 发现驱动队列（Inductor CPU） | 已执行且可裁决 | 已执行且可裁决 | — | 不适用（C++ 后端，不是 Triton） | **已执行且可裁决**（第一阶段规格） | CPU 约 4 核·小时；人工约 30 行 |
| G7 数值作用流（20 个程序） | — | — | — | 模式 A + 统计层：已执行且可裁决（17 个建立参照；两个检出已核验） | — | 约 4 分钟 + 复现约 3 分钟 |

## 1. optimizers

**设置。** 条件 24 个（覆盖计划去掉 `impl` 后 17 个 + 每个优化器一条状态序列 1；偏离 7）× 3 seed，每个运行 3–8 步，每步之后记录参数、
全部状态张量与学习率。候选 10 个：torch 2.10 的 for_loop / foreach / fused（CPU、CUDA）、`torch.compile(opt.step)`（CUDA）、bnb
`AdamW32bit`、torchao `_AdamW`（fp32 状态）、HF `Adafactor`；DeepSpeed FusedAdam 环境不可用。数据 `results/essential/phase2b/optimizers/`
（`analysis.json`、`fr_modeA.json`），脚本 `scripts/essential/p2b_optimizers.py`、`p2b_optimizers_analyse.py`、`p2b_fr_modeA.py`。

**E（K − K_eager，同设备，K_eager = for_loop；τ₃₂ 逐元素，step 计数逐位）。** 单位是（条件 × seed）的整条轨迹。

| 候选 | 比较的轨迹 | 超出 τ₃₂ | 参数轨迹逐位相同 | 最大 \|K − K_eager\| / τ₃₂ |
|---|---|---|---|---|
| foreach CPU / CUDA | 69 / 69 | 0 / 0 | 63 / 37 | 0.0002 / 0.0004 |
| fused CPU / CUDA（无 RMSprop、Adafactor） | 57 / 57 | 0 / 0 | 39 / 1 | 0.0015 / 0.0017 |
| 编译的 step（CUDA） | 69 | 0 | 8 | 0.0015 |
| bnb AdamW32bit（只比参数与映射的 state1 / state2） | 24 | 0 | 0 | 0.11 |
| torchao _AdamW | 24 | 0 | 0 | 0.0005 |
| HF Adafactor | 不适用（与 torch.optim.Adafactor 算法不同） | — | — | — |

预注册 P 的「for_loop == foreach == fused」就是上表的 E，不重复计数。

**P（预注册）与状态序列。** 全部候选零违反：

| 性质 | torch 各候选（每个） | bnb / torchao | HF Adafactor |
|---|---|---|---|
| maximize(g) == minimize(−g)，逐位 | 0/69（fused 0/57） | 不适用（无 maximize） | 不适用 |
| grad = None 跳过该参数（值与状态逐位不变）；零梯度推进步数 | 0/6 | 0/6 | 不适用（条件中无 Adafactor） |
| save → 新优化器 → load → 继续 == 不中断，逐位 | 0/27（fused 0/21） | 0/6 | 0/3 |
| amsgrad：max_exp_avg_sq 不减且 ≥ exp_avg_sq | 0/9 | 不适用 | 不适用 |
| 状态序列「非有限跳过 → 下一步正常」：跳过处快照不变，其后轨迹 == 去掉该步的运行，逐位（GradScaler；fused 走 found_inf 内核路径） | 0/3 | 0/3 | 不适用 |

**事后性质（单独报告）。** 步数计数 = 实际应用的步数、全部有限、SGD 第一步动量缓冲 = 有效梯度（文档 b₁ = g₁，逐位，0/15）、wd = 0 时
Adam 与 AdamW 逐位相同（0/6）、Adafactor 行 / 列二阶矩均值一致（torch 0/6、HF 0/6）：全部零违反。

**FR（模式 A，编译的 step）。** 从 for_loop 轨迹第 2 步的参数、状态与梯度出发执行一次编译的 step（第 3 步），10 个设置（优化器 ×
maximize × 权重衰减）× 3 seed：9 个设置的全部输出（参数与状态张量）由 Triton 写出、TTIR 覆盖完整、K_R 全部完整有限，K 全部落在
K_R 区间加 τ₃₂ 之内（最大距离 / τ₃₂ = 0.0007）。Adafactor：没有 Triton 启动（断图后以 eager 运行），记为「未建立」。

**结论。** 优化器家族在已覆盖的组合上没有发现共有错误或候选错误。这是零结果：预注册的 P 都是不变性与单调性，查不到「所有实现
一致地按错误公式更新」——那需要 F（规格未交付）。

**预算（协议第 6 节）。** 试跑与完整运行重合（偏离 7）。CPU：eager 候选合计约 10 秒；GPU：eager 约 8 秒、编译约 86 秒、模式 A 的
FR 约 25 秒、库候选约 8 秒；最慢单例 3.9 秒（编译、第一次编译）。接入人工：736 行脚本，约 1.5 小时。远低于单家族上限。

## 2. attention

**设置。** 条件 24 个（覆盖计划去重）× 3 seed，前向与反向（dq、dk、dv，上游梯度随机、在无定义行上为零）。掩码按 SDPA 文档的
左上对齐（`tril(diagonal=0)`）；SDPA 的「causal」用 `is_causal=True`，其余用显式布尔掩码。没有任何允许键的行（fully_masked_row 的
两行、q>k 时滑动窗口末尾的行）规格层面无定义：不进 E 与 P，按候选记录约定（偏离 8）。候选 11 个受裁决或记录：SDPA math / efficient
（float32、bf16）、flash（bf16）、cuDNN（bf16）、math CPU float64、Inductor 编译的手写注意力、编译的 flex_attention（float32）、HF
`eager_attention_forward`（float32）、xformers `memory_efficient_attention`（bf16）；另有两个只作参照的 eager：手写注意力 eager、
未编译的 flex_attention。数据 `results/essential/phase2b/attention/`，脚本 `scripts/essential/p2b_attention.py`。

**不能运行的组合（记为「不支持」，不计入）。** flash（bf16）只接受无掩码与 q = k 的 causal（任意掩码与 q ≠ k 的 causal 都「No available
kernel」），24 个条件中 9 个可运行；efficient 与 xformers 不支持 GQA（xformers 是反向没有算子）；flex 编译在 head_dim 72 时 Triton
编译失败（「Shape element 2 must be a power of 2」，flex decoding 模板用了未取整的 V_HEAD_DIM）——**召回（已知）**：#164931 于
2026-01-28 修复，晚于 2.10 分支；nightly 20260907 编译与运行正常。

**E（同设备同 dtype 的 eager 参照；float32 按 τ₃₂ 裁决，bf16 只记录）。** float32：SDPA efficient 对 math 0/24 个条件、Inductor 对手写 eager
0/24、flex 编译对 flex eager 0/23、HF 对 SDPA math 0/24；最大 |K − K_eager| / τ₃₂ 为 0.015。bf16（记录）：前向全部在 2⁻⁸(1 + |K_eager|) 内；
反向在 13–17 个条件上有少数元素超过 2⁻⁸ 的相对量（最大 3.84 倍，四个 bf16 候选的最大值相同，偏差落在参照 SDPA math bf16 一侧——math
在 bf16 中逐步舍入）。

**P（预注册）。** 全部候选零违反：被掩码的键（扰动成 10 倍的随机值）不影响其余行的输出与 dq（逐位，因果 0/12、填充与滑动窗口 0/24；
全部候选两次运行逐位相同，逐位判定有效）；对所有行都被掩码的键 dk = dv = 0（0/24）；V = 1 时每个有定义行的输出为 1（0/72，flash 0/27）；
GQA 等于把 KV 头重复后不开 GQA（0/3）。

**无定义行的约定（记录，不计分）。**

| 候选 | 没有允许键的行 |
|---|---|
| SDPA math（float32 / bf16 / CPU float64）、efficient、flex（eager 与编译）、xformers | 0（SDPA 自 #131863 起的约定） |
| HF eager | V 的均值（加性掩码用 finfo.min，整行相等后均匀注意） |
| 手写注意力（eager 与 Inductor） | NaN，且 NaN 进入 dk、dv |
| **SDPA cuDNN（bf16，2.10）** | **忽略掩码：等于不加掩码的注意力**（与不加掩码的结果相差在 bf16 舍入内） |

cuDNN 一行的原因：2.10 的 `convert_boolean_attn_mask_cudnn` 把布尔掩码的 False 填成有限值 −65504（`aten/.../attention.cpp`），整行都被
掩码时 softmax 平移不变，退化为不加掩码；传入 −∞ 的加性掩码则得 0。nightly 20260907（同为 cuDNN 9.10）两种掩码都得 0。**召回（已知）**：
#177842「When using the cuDNN backend, custom masks become ineffective」，由 #177868 于 2026-03-21 修复（晚于 2.10）。在本机 sm_86 上 cuDNN
不是默认后端，只有显式选择才会走到。

**FR（模式 A）。** flex 编译：23 个条件 × 3 seed，out / dq / dk / dv 全部由 Triton 写出、K_R 全部完整有限，K 全在 K_R 区间加 τ₃₂ 之内
（0 个元素超出）。Inductor 手写注意力：输出与梯度由 cuBLAS 的矩阵乘写出（不是 Triton），24 个条件中只有 GQA 条件的 dk、dv（组内求和
由 Triton 写出）可判，也全部在区间内；其余记为「未建立（输出不由 Triton 写出）」。

**结论。** 没有新的本质错误；两处召回（#164931、#177842）都是 2.10 中已在上游修复的错误。

## 3. packing

**设置。** 条件 9 个（覆盖计划的 (docs, lengths) 组合，`impl` 去掉后每个实现都是候选；长度见脚本头）× 3 seed。候选：SDPA（efficient）
带块对角因果布尔掩码、编译的 flex_attention 带文档 & 因果 mask_mod（参照：SDPA math、未编译的 flex），以及 2 层 Llama（transformers 4.57.3，
随机权重，float32）用按文档重置的 position_ids、不给 attention mask 的打包路径（attn_implementation = sdpa；参照：同权重的 eager）。
数据 `results/essential/phase2b/packing/`，脚本 `scripts/essential/p2b_packing.py`。

**E。** SDPA efficient 对 math 0/27（最大 0.0096 τ₃₂）、flex 编译对 flex eager 0/27（0.0041）、HF sdpa 对 HF eager 0/27（0.040）。

**P（预注册：打包后每个文档的输出等于该文档单独运行）。** 前向与反向，全部候选 0/27（HF 比较 logits 与对输入嵌入的梯度，最大 0.056 τ₃₂）。
单 token 文档（登记的高风险组合）与 5 个文档的情形都没有串扰；HF 4.57.3 的 position_ids 打包路径对 sdpa 与 eager 都隔离了文档。

**结论。** 零结果。

## 4. training_program

**设置。** 条件 5 个（覆盖计划的 (tokens, ranks) 组合）× 3 seed；窗口固定为 4 条序列、k = 2；2 个 rank 用 torchrun + gloo（CPU）。
候选 6 个：HF Trainer 4.45.2 / 4.46.0 / 4.57.3 / 5.19.0（各自环境），Accelerate 1.7.0 按其梯度累积指南基础示例写的循环
（`accelerator.accumulate` + `accelerator.backward(loss)`），PyTorch 按 AMP / DDP 示例写的普通循环（`loss / k`，DDP 除最后一个
micro-batch 外 `no_sync`）。**F 开放**：`specs/phase1/spec_accumulation.py` 的窗口要求（全局批次所有非忽略 token 的平均）。数据
`results/essential/phase2b/training_program/`，脚本 `scripts/essential/p2b_training_program.py`。

| 候选 | F（偏离的条件 / 15） | K / f（unequal，1 rank） | E（不累积对累积） | P 重新划分 | P padding-free | P rank 不变 | 归属 |
|---|---|---|---|---|---|---|---|
| HF 4.45.2 | 9 | 0.974 | 9 | 6 | 不适用 | 3/6 | 召回：均值的均值（#34191，4.46 修复） |
| HF 4.46.0 | **3**（只在 2 rank、token 不等） | 1.000 | 0 | 3 | 不适用 | **3/6** | 召回：跨 rank 不汇总 token 数（#34242；之后加 `average_tokens_across_devices`） |
| HF 4.57.3 | **15**（含 token 相等） | 0.857 | 0 | 0 | **9/9** | 0/6 | 召回：每行多计首个标签（#46204，第一阶段 W7 已见） |
| HF 5.19.0 | 0 | 1.000 | 0 | 0 | 0/9 | 0/6 | 全部相容 |
| Accelerate 1.7.0 循环 | 9 | 0.974 | 9 | 6 | 不适用 | 3/6 | 按文档基础示例写的循环即均值的均值；同一份文档另有「可变长度样本的梯度累积」一节给出按 token 归一的写法 |
| PyTorch 普通循环 | 9 | 0.974 | 9 | 6 | 不适用 | 3/6 | 同上（示例的 `loss / k`） |

每个 2 rank 的运行两个 rank 的梯度逐位相同。方法上：E（不累积对累积）只看到 4.45.2 式的累积错误；跨 rank 问题只有 F 与 rank 不变性
看得到；4.57.3 的计数问题只有 F 与 **padding-free 不变性**看得到——第一阶段预注册的重新划分不变性看不到它，2b 预注册的
padding-free 不变性 9/9 看到（与上游的发现路径一致）。**结论：** 无新错误；三个 HF 版本的已知问题全部被重新发现，5.19.0 全部相容。

## 5. normalization

**设置。** 条件 23 个（覆盖计划去重）× 3 seed，前向与反向（y、dx、dw、db，BN 训练另有 running_mean / running_var）。候选：eager CPU /
CUDA（float64、float32、CUDA bf16）、Inductor（float32、bf16）、nightly eager / Inductor（float32）、Liger RMSNorm / LayerNorm（只覆盖各自的
算子、带仿射的条件）。所有候选收到同一组 float32 可表示的输入（偏离 9）。数据 `results/essential/phase2b/normalization/`。

**E 与 E64。** 超出 τ₃₂ 的 float32 偏离全部落在两类输入上：huge_offset（1e4 + N(0, 1)：LayerNorm、GroupNorm、BN 训练；eager、Inductor、
nightly、Liger LayerNorm 都有）与 GroupNorm 的 constant_rows（Inductor）。**裁决：数值（条件数），不是语义差异**，两条独立证据：
(1) 在精确平移后的输入上（x − 1e4，或减去行常数；float32 中精确）每个候选都与 float64 eager 相容（偏离 9 的规则）；(2) 模式 A 的 FR：
Inductor 的 K_R（自身 TTIR 的精确求值）与 float64 eager 在 τ₃₂ 内一致（全部 23 个条件，最大 0.005 τ₃₂），超出的部分全部是 e_num = K − K_R
（舍入）。bf16（只记录）：K_R 与在同一 bf16 输入上的 float64 eager 一致（最大 0.005 τ₃₂），K 的偏离全部是 e_num。

**P。** 尺度不变（norm(2x) = norm(x)，前提：集合方差 ≥ 1e3·eps）：Inductor、Liger 与 float64 零违反；eager float32（CPU、CUDA、nightly）的
GroupNorm 与 BN 训练在 huge_offset 上 6/36——eager 按 y = x·s + t（s = rstd·w，t = b − μ·s）计算，两项都约 1e4 而相消；在平移后的输入上
尺度不变恢复，归为数值。BN running_var 用 n / (n − 1)：0/12。BN eval 只用 running 统计量：batch ≥ 2 的条件不在覆盖计划中，未执行。
事后：行独立（扰动其他行，第 0 行逐位不变）0/45。

**FR（模式 A）。** Inductor float32 与 bf16 全部 46 个（条件, dtype）的全部输出由 Triton 写出、K_R 完整。Liger 的 FR 未运行：liger 环境没有
求值依赖（gmpy2），需要「liger 中捕获、ka_main 中求值」的捕获包流程，留待之后。

**结论。** 零结果；E 与 P 的全部 float32 超差都由条件数解释，并由 FR 归到 e_num。

## 6. embedding

**设置。** 条件 13 个（覆盖计划；`mode` / `bags` / `per_sample_weights` 的单因素边界移到 embedding_bag，偏离 9）× 3 seed。候选：eager CPU /
CUDA（float64、float32）、Inductor、nightly eager / Inductor。数据 `results/essential/phase2b/embedding/`。

**E / E64** 全部 0 超出（Inductor 最大 0.00075 τ₃₂）。**P** 零违反：padding_idx 行梯度为零（0/9）；scale_grad_by_freq 等于按频次相除（0/3）；
embedding_bag(sum) 等于逐行 F.embedding 之和（0/15）；空 bag 为零（0/6）。事后：mean / max 同样等于逐行结果（0/6、0/3）；只含 padding 的
bag 为零（0/3）；max_norm 原地重归一化（被引用且超限的行缩到 max_norm、未超限行与未引用行不动、输出行范数 ≤ max_norm）0/6。**结论：** 零结果。

## 7. schedulers

**设置。** 条件 10 个（覆盖计划）× 3 个基础学习率。候选：torch 2.10、torch nightly 20261005（CPU）的 CosineAnnealingLR / OneCycleLR / LinearLR，
transformers 4.57.3 的 `get_cosine_schedule_with_warmup`。**P：** 链式 step 等于类自身的闭式（`_get_closed_form_lr`）0/12；中途 state_dict 恢复等于
连续运行（逐位）0/12；阶段边界值：steps ≥ 10 全部相符；**steps = 1 的退化计划**中文档给出的两个边界值落在同一步、互相矛盾（OneCycleLR：
起点 max_lr / div_factor 与终点 initial_lr / final_div_factor 同在第 0 步，实现取终点值；HF：warmup 结束值 base 与训练结束值 0 同在第 1 步，
实现取 base）——记为约定，不计违反。OneCycleLR 在 total_steps 之后拒绝再 step（文档行为）。**E：** nightly 对 2.10 逐位相同 0/18。**结论：** 零结果。

## 8. clip_amp

**设置。** 条件 11 个（覆盖计划合并空组合）+ 2 个事后条件（GradScaler 性质所需的 scaler_step × with_inf / with_nan）× 3 seed。候选：
`clip_grad_norm_` / `clip_grad_value_` / GradScaler（torch 2.10 CPU、CUDA）、Accelerate 1.7.0 的 `Accelerator.clip_grad_norm_`；另有只作参照的
float64 CPU；DeepSpeed 环境不可用。**P** 零违反：总范数 ≤ c 时逐位不变（0/15）、否则缩到 c（0/18）；clip_value 精确截断（0/3）；分片范数合并
等于全局范数（0/3）；非有限一步被跳过且 scale 减半（0/6）；有限一步精确反缩放（事后，0/3）。**E**：Accelerate 对 torch CUDA 0/27；E64 0/39。
非有限梯度（error_if_nonfinite = False，默认）：总范数为 NaN / inf，裁剪后梯度非有限——文档行为，记为约定。**结论：** 零结果。

## 9. 基础算子：matmul_linear、reductions、activations、gather_layout

**设置。** 条件取各自的覆盖计划（19、53、25、18 个，去重后）× 3 seed，前向与反向；所有候选收到同一组 float32 可表示的输入。候选：eager
CPU / CUDA（float64、float32、CUDA bf16）、Inductor（float32、bf16）、nightly eager / Inductor（float32），以及登记的库候选（Liger softmax、
Liger SwiGLU、Unsloth SwiGLU，只在各自实现的条件上）。TF32 关闭。数据 `results/essential/phase2b/<family>/`，脚本
`scripts/essential/p2b_basic.py`（四个家族共用一个框架）。

**E / E64（float32）。** matmul、reductions、gather：全部 0 超出。activations：Inductor 与 nightly 对 eager 0 超出；E64 在 3 个条件上不相容：
swiglu / geglu × huge 的输出为 inf（真实值约 1e60，超出 float32 表示范围：IEEE 溢出，记为约定），以及 **gelu_tanh × huge 的反向全部 NaN
（B023，见下）**。reductions 的 amax / amin 在空轴上报错（eager 与 Inductor 都报，文档行为，记为约定）。bf16（只记录）的偏离都在 bf16 舍入量级。

**P。** 全部零违反：matmul 线性（48）、转置恒等（48）、两次反向的梯度累加（57）；reductions 的 softmax / log_softmax 平移不变（24）、
logsumexp 平移（12）、cumsum 末元素等于总和（3）、小整数全归约的置换不变（6）、**D1：amax / amin 并列时梯度平均分配**（21，文档：amax/amin
evenly distributes gradient between equal values）；activations 的奇函数恒等 f(x) − f(−x) = x（27）、relu 在 0 处导数为 0（3）；gather 的
反向计数（36）、split → cat 恒等（9）、视图写穿（9）。

**新发现 B023。** `gelu(approximate='tanh')` 的反向在 |x| ≥ 1.8447e19 时返回 NaN（float32 与 bf16；eager CPU 标量与向量化路径、eager CUDA、
Inductor 分解写的是同一公式：`x_sq` 溢出为 inf，乘上恰为 0 的 tanh 导数），真实导数为 1 / 0；erf 形式与 float64 有限；nightly CUDA 20260907
与 CPU 20261005 同样。由预注册的高风险组合（gelu_tanh × huge）触发；E 看不到（各后端一致地 NaN），E64 与模式 A 的 FR（K 为 NaN、K_R 有限
且包含 float64 结果，594/594）看得到。实际影响小。记录 `bugs/B023_gelu_tanh_backward_nan_large_input.md`，草稿待用户决定。

**FR（模式 A，Inductor float32）。** 四个家族 115 个条件中，由 Triton 写出的输出 K_R 全部完整，K_R 区间都包含 float64 eager 的结果；K 与 K_R
的特殊值类别不一致只出现在 activations 的 huge 条件（gelu_tanh 反向 = B023；swiglu / geglu 输出 = 溢出）；matmul 的输出由 cuBLAS 写出，
未建立。

## 10. checkpoint、rope、moe

**checkpoint**（10 个条件）：non-reentrant / reentrant 的 `torch.utils.checkpoint` 与 HF `gradient_checkpointing_enable`，mlp / 共享参数 /
中间量用两次，编译开与关。P「checkpoint 开 == 关」（CPU float64 逐位，其余 τ₃₂）：全部候选零违反（CPU 21、CUDA 21、编译 21、HF 9）。
**rope**（6 个条件）：仓库内参照（rotate-half 与交错两种约定）、HF `apply_rotary_pos_emb`、Liger、Unsloth（三者只实现 rotate-half；交错约定
记为「不适用」而不是偏离）。E（对仓库内参照）0/12；P：成对范数不变、相对位置（位置整体平移 5 后点积不变）、R(0) = 恒等，全部零违反。
**moe**（10 个条件）：登记的仓库内参照（CPU 逐 token 循环、CUDA 排序 + index_add、其编译版本）。P：permute → unpermute 恒等、每个 token 的
合并权重和为 1、被丢弃的分配贡献为零（8 个有丢弃的运行）、Switch 辅助损失等于其计数公式，全部零违反；E 0/30。这一家族的候选都是本仓库
写的参照实现，结论只说明这些参照之间一致。

## 11. 调用场景 A → B → A（编译候选）

同一个 `torch.compile` 的函数依次以输入 A、B（换形状、dtype、存储布局、Python 标量参数、广播或 requires_grad，迫使守卫失败或走动态形状）、
A 调用，`dynamic=False` 与默认自动动态两种；14 个程序 30 种 B、共 60 条序列，另有 BatchNorm 的 train → eval → train 与编译的 AdamW 在两组
参数间交替。契约（每次调用都与 eager 在 τ₃₂ 内一致，B 为 bf16 时按 bf16 记录）：**0 违反**。逐位比较作为信息记录：LayerNorm 在自动动态下
换形状 / 布局后重编译为动态图，A 的第二次调用与第一次、B 与单独编译各差几个 ulp（2 条序列），都在 τ₃₂ 内。BatchNorm 模式切换与 AdamW 交替
步骤都与 eager 一致。数据 `results/essential/phase2b/compile_sequences/results.json`。

## 12. G6 陌生组合：参照复用

10 个由 2b 家族组合成的程序（预归一化 Transformer 块、带 RoPE 的注意力、LayerNorm + GELU MLP、GroupNorm + SiLU、交叉熵头、MoE 块、归约链、
gather / scatter 往返、BatchNorm + 池化链、损失 + AdamW 更新），Inductor 编译、float32、前向与反向；每个程序只写组合本身，接入工具用同一个
通用包装（约 60 行，十个程序共用），**没有为任何程序写参照或声明**。

| 量 | 结果 |
|---|---|
| Triton 启动 | 88 个，TTIR 覆盖全部完整 |
| 输出 | 42 个；27 个由 Triton 写出，15 个由 cuBLAS 矩阵乘写出（工具范围之外，记为「不由 Triton 写出」） |
| **自动参照 K_R 完整** | **27 / 27**（每个元素都有完整有限的区间） |
| K_R 区间包含同一组合的 float64 eager 结果 | 27 / 27 个输出的全部元素 |
| E（编译对 eager，float32） | 42 个输出全部 0 超出 τ₃₂ |
| 工具时间 | 合计 36.6 秒（编译与预热 17.4、参照求值 15.8、其余捕获与统计） |

结论：在这 10 个陌生组合上，工具对所有由 Triton 写出的输出自动建立了完整参照，不需要逐程序的工作；参照复用的边界是「输出由 Triton 写出」
（矩阵乘的输出不在范围内）。数据 `results/essential/phase2b/g6_compositions/results.json`。

## 13. G7 数值作用流

20 个低精度程序（bf16 / fp16 的 Inductor 编译：归一化、softmax、logsumexp、长求和 / 均值、方差、cumsum、激活、交叉熵、注意力 softmax、
scatter_add、embedding_bag、AdamW 更新、残差 + RMSNorm），模式 A，工具默认的 32 个开发单位 + 64 个确认单位，方向规则与默认检测器判定
e_num 有无经确认的系统（平均）作用。这些是实现的数值作用，不是本质错误，单独报告。

| 结果 | 程序 |
|---|---|
| 参照建立、无经确认的系统作用（15） | layer_norm、rms_norm、group_norm、batch_norm（bf16）；softmax、log_softmax（fp16）、logsumexp、cumsum；silu·mul、geglu（fp16）；交叉熵；注意力 softmax；AdamW 更新；残差 + RMSNorm（fp16）；mean（fp16） |
| 参照建立、**检出系统作用**（2） | sum_long_bf16（R2 / R3；新 seed 上不复现，见下）；gelu_tanh_bf16（R1 / R2 / R3；新 seed 上复现，来自正确舍入本身，见下） |
| 参照未建立（3） | var_bf16（长行的方差用了工具 TTIR 映射未识别的 Welford 归约合并函数：`tt.reduce` 覆盖不完整，工具缺口）；scatter_add_bf16、embedding_bag_mean_bf16（输出不由 Triton 写出：Inductor 回退到 ATen） |

数据 `results/essential/phase2b/g7_numerical_stream/results.json`。

**两个检出的核验**（`results/essential/phase2b/g7_numerical_stream/replication.json`，探针 `scripts/essential/probes/g7_*.py`）：
- sum_long_bf16：在新的 96 个 seed（1000–1095）上**不复现**（四条规则都未确认）；直接检查 4,096 行，Inductor 与 eager 的结果 100% 等于
  精确和按就近舍入到 bf16 的值，带号相对误差均值 1.0e-5 ± 2.6e-5（不显著）。归为偶然检出，不是系统作用。
- gelu_tanh_bf16：在新 seed 上**复现**（R1 / R2 / R3，另 R5）。机制：Inductor 与 eager 的输出 99.99% 等于精确值按就近舍入的 bf16 值，二者的
  带号相对误差均值几乎相同（5.30e-5 ± 1.2e-6 对 5.27e-5）——系统作用来自「正确舍入」本身：|x| ≳ 3 时 gelu(x) = x − δ（x 本身是 bf16 值，
  δ 小于半个 ulp），就近舍入把输出推回 x，残差与参照同号、幅度被推离零。这是函数结构与 bf16 格点的关系，不是实现的缺陷（eager 相同）。

## 14. G8 发现驱动队列：Inductor CPU（C++ 后端）

W5 列出的被跳过、只跑第一个样例或以 eager 为参照的算子（index_reduce、scatter_reduce 的 amax / amin / mean、max_pool、cross_entropy）在第一阶段
的条件上补跑 Inductor CPU float32——第一阶段与 2b 此前都没有运行过 Inductor 的 CPU 后端。条件、输入、规格与 eager 运行与第一阶段共用，因此
**F 开放**（第一阶段的规格 v0.4）。index 462、pool 392（max_pool）、ce 274 个条件 × 3 seed。数据 `results/essential/phase2b/g8/`。

| 家族 | Inductor CPU 前向 | 反向 | 归属 |
|---|---|---|---|
| index（462） | 420 相容；42 编译失败 | 371 相容；49 共有偏离；42 编译失败 | 共有偏离 = B020（eager CPU 同样 50 个）；编译失败：2.10 的 C++ 代码生成在单元素目标上编译 index_reduce prod / amax / amin（及部分 mean）的反向时 `AssertionError`；nightly 20261005 用 gcc 11 编译后全部正确（本机 gcc 9.3 不支持 nightly 要求的 `-std=c++20`，nightly 的 Inductor CPU 只能换编译器运行）——已在上游修好，与 B016 记录的 CPU 编译断言同族 |
| pool（392） | 377 相容（15 个暂停：文档版本分歧） | 303 相容；69 集合式检查；3 候选错误；2 候选相容而 eager 偏离 | 3 个候选错误 = 第一阶段 CUDA Inductor 的同 3 个条件（真实 −∞ 与 padding 并列时下标 −1、梯度不路由：约定，见 E5 / B022）；2 个 = B021（eager CPU 越界写，Inductor CPU 正确） |
| ce（274） | 253 相容；21 规格无定义 | 同前向 | 零偏离 |

结论：Inductor CPU 与 CUDA Inductor 共享 lowering 的地方行为一致（B020 共有、−∞ 并列约定相同）；队列中唯一的新现象（单元素目标的反向编译
失败）是 2.10 的编译崩溃，nightly 已修好，不另立编号。探针 `scripts/essential/probes/g8_inductor_cpu_single_element_scatter.py`，输出
`results/essential/phase2b/g8/inductor_cpu_single_element_probe.txt`。float16 的队列项需要重求规格，本轮未运行。

## 15. 2b 小结

**覆盖。** 15 个第一档家族全部执行（E、P、登记的状态序列；Triton 候选的模式 A FR；F 只在训练程序层与 G8 中开放），另有 S3 边界补充、调用场景
A → B → A、G6 陌生组合、G7 数值作用流、G8 发现驱动队列。每个家族的条件取预注册的覆盖计划（只做去重与偏离 7–11 记录的映射）。所有家族都远低于
预算上限（单家族最多约 30 分钟 GPU / 4 核·小时 CPU）。

**发现。**

| 类别 | 内容 |
|---|---|
| 新的本质错误 | **B023**：gelu tanh 形式的反向在 \|x\| ≥ 1.8447e19 时 NaN（eager CPU / CUDA / Inductor 共有；实际影响小） |
| 召回（已知或上游已修） | flex decoding head_dim 72 编译失败（#164931）；2.10 cuDNN 布尔掩码用 −65504（#177842）；HF Trainer 4.45.2（#34191）、4.46.0 跨 rank（#34242）、4.57.3 计数（#46204）；S3：CUDA avg_pool channels-last 反向（#188344）、Inductor max_pool3d 交换存储下标（#197434）；G8：2.10 Inductor CPU 单元素目标反向编译断言（nightly 已修） |
| 数值（不是本质错误） | 条件数造成的超差（归一化 huge_offset / constant_rows、index 前向的抵消）；bf16 / 输出溢出；G7 中 gelu_tanh bf16 的系统作用来自正确舍入本身 |
| 约定（无定义输入，记录不计分） | SDPA 各后端在没有允许键的行上（0 / V 的均值 / NaN / 2.10 cuDNN 不加掩码）；调度器 steps = 1 的边界矛盾；空轴 amax 报错；非有限梯度的裁剪 |

**方法对照（2b 的新例）。** B023：E 否、E64 是、预注册 P 否（huge 数值不查奇偶恒等）、模式 A 的 FR 是（特殊值类别不一致）。HF 4.57.3 计数：E 否、
F 是、P（padding-free 不变）是。HF 4.46.0 跨 rank：E 否、F 是、P（rank 不变）是。2.10 cuDNN 掩码：E 不比较无定义行，约定表看到。Inductor max_pool3d
（S3）：E 是、P（下标在 argmax 集合内）是。决定规则（协议第 4 节）：在 F 开放的家族里，F 发现的问题 2b 预注册的轻量性质也都发现了；本轮没有
「只有 F 能发现」的新例（第一阶段的 HF 4.57.3 例在 2b 中被 padding-free 不变性补上）。论文侧重维持 2a 的结论：前向共有错误、归因（FR 把条件数
与 bf16 的偏离全部归到 e_num）与陌生组合的参照复用（G6：由 Triton 写出的输出 27 / 27 自动建立完整参照）。

**仍待外部输入。** 14 个家族的独立规格（交付后开放 F 与模式 B 的 FR）；Liger 归一化 kernel 的 FR（需要捕获包流程）；G8 的 float16 项；S5；
上游提交（B020–B023、B015 评论、文档 issue）由用户决定。
