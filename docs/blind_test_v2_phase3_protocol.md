# blind_test_v2 阶段 3（参数更新层）的冻结口径

本文件在运行阶段 3 之前提交，按出题方 v2 包第 4 节与阶段 1 审阅的两点意见写定共同状态与测量口径。代码：
`scripts/blind_test_v2_phase3.py`、`src/kernel_analyzer/reference_eval/update_layer.py`。判定层仍是 `detector-v2.1`。

## 1 共同状态（每个测量 seed s，K 与 K_R 共用）

| 项 | 规则 |
|---|---|
| 梯度 | 程序输出 y 按行优先展平；K 为 kernel 的 FP32 输出 |
| 参数 θ_s | `torch.randn(n, generator=torch.Generator("cpu").manual_seed(5000 + s), dtype=torch.float32)` |
| 历史梯度 | 对 i = 0..15，输入 `make_inputs(family, 20000 + 16s + i)`，规格 f 的严格包围（阶段 2 的求值代码），取包围中点再按最近偶数舍入到 FP32（RN32）。包围两端舍入到不同 FP32 值的坐标计数并报告 |
| AdamW 状态 | 同一个 FP32 optimizer（下行）从零状态开始，在一个副本参数上依次用 16 个历史梯度各走一步；保存 exp_avg、exp_avg_sq 与 step = 16。weight_decay = 0，副本参数的取值不影响状态 |
| optimizer | torch 2.10 CPU 上的 `torch.optim.SGD(lr=2⁻¹⁰)` 与 `torch.optim.AdamW(lr=2⁻¹⁰, betas=(0.9, 0.999), eps=2⁻²⁷, weight_decay=0)`，`foreach=False, fused=False`，全部 FP32 |
| 三种设置 | SGD；AdamW（带历史）：载入上述状态，测量步 t = 17；AdamW（零动量）：零状态，t = 1（预先登记的近似抵消对照） |

用 f 生成历史状态是出题方同意的替代办法（不需要知道哪个程序是干净版本）；它不保证等于原协议中「干净 kernel 实际输出的历史」，
出题方需按这一状态核实更新层的预期。

## 2 主表：实际写入差

同一 optimizer、同一 θ_s 与状态，只替换梯度：

    u = θ′(K) − θ′(g_R)，两个 FP32 结果之差在 float64 中精确。

- g_R 是参照梯度 K_R 的 FP32 舍入。某坐标 K_R 区间两端的 RN32 相同时，g_R 唯一，直接重放；
- 不唯一时，枚举区间 [RN32(L_R), RN32(U_R)] 内的全部 FP32 值（不超过 4 个）。SGD 与 AdamW 都逐元素计算，所以逐坐标对每个候选
  重放，u 的区间取 θ′(K) 减去各候选 θ′ 的最小、最大值——这是候选集合上的可靠包围；
- 候选超过 4 个，或 K_R 不是完整组合参照的坐标：参照未建立，按阶段 1 的坐标集规则处理（在开发 seed 上固定坐标集）。
- 参照更新 r = θ′(RN32(K_R 中点)) − θ_s（FP32 重放，float64 中求差），用于 R2、R3 的方向与默认检测器的对齐检验。

## 3 副表：理想响应（实数 optimizer）

同一状态下，optimizer 的实数语义用区间算术分别对 K（点）与 K_R（区间）求一步，u = step(K) − step(K_R)（θ 在差中抵消），不含
optimizer 自身的 FP32 舍入。它是理想响应分析，**不是实际写入差**，两者可能结论不同（例：θ = 1，η = 2⁻¹⁰，g_R = 1，
g = 1 + 2⁻²³ 时实数 SGD 响应差为 −2⁻³³，两侧 FP32 写入的参数相同，实际差为 0）。r 取理想参照更新的中点。

## 4 规则与报告

规则 R1（−1/√n）、R2（−sign(r)/√n）、R3（−r/‖r‖，正号表示沿参照更新方向推得少）、R5（开发 seed 上学习，程序级复现）；seed
0–95（32 开发、64 确认）与 96–191 复现；每种 optimizer、每个口径、每轮各自对全部「程序 × 规则」做 Holm。每种 optimizer 一个
CSV（模板列：程序、规则、两轮 Holm 后判定、μ 的端点保守区间、复现标记、u 的区间最大宽度、参照重放方法），主表为实际写入差，
理想响应另列；默认检测器作为第二指标另列。
