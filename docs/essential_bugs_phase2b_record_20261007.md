# 本质错误一轮·2b 运行记录（逐家族追加）

依据：协议 v2 第 3 节（候选清单 3.1、覆盖矩阵 3.2，均在 2b 运行之前提交）、第 6 节（预算）、第 7 节（状态）、第 9 节（偏离）。
F 与模式 B 的 FR 在各家族的规格入库前关闭（训练程序层的累加窗口除外）；E、P、状态序列、模式 A 的 FR 不等待规格。检测器 `detector-v2.2`。

## 家族状态

| 家族 | E | P（预注册） | 状态序列 | FR | F | 预算（试跑估算 / 实际） |
|---|---|---|---|---|---|---|
| optimizers | 已执行且可裁决 | 已执行且可裁决 | 已执行且可裁决 | 模式 A：已执行且可裁决（Adafactor 未建立） | 未运行（规格未交付） | GPU 约 2.5 分钟、CPU 约 10 秒；人工 736 行、约 1.5 小时 |

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
