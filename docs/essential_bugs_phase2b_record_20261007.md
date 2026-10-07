# 本质错误一轮·2b 运行记录（逐家族追加）

依据：协议 v2 第 3 节（候选清单 3.1、覆盖矩阵 3.2，均在 2b 运行之前提交）、第 6 节（预算）、第 7 节（状态）、第 9 节（偏离）。
F 与模式 B 的 FR 在各家族的规格入库前关闭（训练程序层的累加窗口除外）；E、P、状态序列、模式 A 的 FR 不等待规格。检测器 `detector-v2.2`。

## 家族状态

| 家族 | E | P（预注册） | 状态序列 | FR | F | 预算（试跑估算 / 实际） |
|---|---|---|---|---|---|---|
| optimizers | 已执行且可裁决 | 已执行且可裁决 | 已执行且可裁决 | 模式 A：已执行且可裁决（Adafactor 未建立） | 未运行（规格未交付） | GPU 约 2.5 分钟、CPU 约 10 秒；人工 736 行、约 1.5 小时 |
| attention | 已执行且可裁决（float32）；bf16 只记录 | 已执行且可裁决 | 不适用 | 模式 A：flex 已执行且可裁决；Inductor 手写注意力的输出由 cuBLAS 写出，未建立 | 未运行（规格未交付） | GPU 约 13 分钟（其中 FR 10 分钟）；人工约 560 行、约 2 小时 |
| packing | 已执行且可裁决 | 已执行且可裁决 | 不适用 | 未运行（flex 的 kernel 与 attention 家族相同，FR 已在那里执行） | 未运行（规格未交付） | GPU 约 2 分钟；人工 258 行、约 0.5 小时 |

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
