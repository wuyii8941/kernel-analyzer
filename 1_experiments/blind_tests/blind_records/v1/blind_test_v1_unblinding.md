# 盲测 v1 揭盲：答案表、勘误与初步符号核对

独立核实已完成（见比对表：0 处区间违反，138 个投影值全部在工具区间内）。按协议揭盲。答案文件 `blind_test_v1_answer_key_UNSEALED.json` 与封存文件逐字节相同，SHA-256 `d346a53adde9fc974f26787ab0e95a97cdbae03e7f8288b2c58b3fd2acc3bd77`。

## 1. 答案表

| 程序 | 家族 | 变体 | 类别 | 机制 | 预期（阶段 1：K − K_R；语义类：阶段 2 K_R − f） |
|---|---|---|---|---|---|
| prog_01 | F7 | clean | clean-only | none injected; uses tl.exp/tl.log (approximate lowering) and max-shifted log-sum-exp | not pre-specified |
| prog_02 | F8 | clean | clean-only | none injected; RN ops, where/select, div_rn by a runtime count | not pre-specified |
| prog_03 | F2 | U-reassoc | unknown | g written as 0.5 + 0.5*d instead of 0.5*(1+d): different rounding sequence, same real value | not pre-specified |
| prog_04 | F5 | N-comm | negative | q*sc written as sc*q | none |
| prog_05 | F3 | P-sq-rz | positive | squared deviations (d*d) computed with mul.rz in every chunk: M2 underestimated | + along w = +1 on the rstd outputs (rstd overestimated by ~2e-8 rel); mean outputs unaffected |
| prog_06 | F2 | P-mul-rm | positive | final product (a*g)*b computed with mul.rm (round toward -inf) | + along w = -1/sqrt(n); residual in (-ulp(y), 0] per element |
| prog_07 | F1 | U-chunk | unknown | sum of squares accumulated as 8 chunk partials elementwise then reduced: summation order changed | not pre-specified |
| prog_08 | F3 | clean | clean | none | none pre-specified |
| prog_09 | F2S | N-copy | negative | verbatim copy of clean | none |
| prog_10 | F6 | S-unmasked-den | semantic | denominator sums g over all positions, mask applied to numerator only (spec: both masked) | K_R - f: - along K_R (y scaled by sum(m*g)/sum(g) < 1, about 0.8) |
| prog_11 | F1 | clean | clean | none | none pre-specified |
| prog_12 | F1 | N-comm | negative | (1+s) written as (s+1): commutation of one addf | none |
| prog_13 | F5 | U-twostage | unknown | sum done as 32 group partial sums then a sum of partials (order change) | not pre-specified |
| prog_14 | F2S | P-ftz | positive | final scale multiply executed as mul.rn.ftz.f32: subnormal results flushed to zero | + along w = -sign(K_R)/sqrt(n) (residual = -K_R on flushed elements, 0 elsewhere) |
| prog_15 | F2 | U-divfull | unknown | gate division via '/' (div.full.f32, hardware reciprocal) instead of div_rn | not pre-specified (depends on hardware reciprocal table) |
| prog_16 | F4 | P-fma-dir | positive | y1 via fma.rm (round toward -inf), y2 via fma.rp (round toward +inf), both before the multiply by t > 0 | + along w = (-1 on y1 dims, +1 on y2 dims, 0 on passthrough)/sqrt(count); not along K_R |
| prog_17 | F6 | clean | clean | none | none pre-specified |
| prog_18 | F1 | S-noeps | semantic | eps omitted from the normaliser (spec: (ms+eps)^-1/2) | K_R - f: + along K_R, relative +eps/(2*ms) ~ +5e-7 for ms ~ 1 |
| prog_19 | F6 | U-divfull | unknown | final division num/den via '/' (div.full.f32) instead of div_rn | not pre-specified |
| prog_20 | F5 | S-group | semantic | scale index off by one element: (k+1)//G clipped (spec: k//G) | K_R - f: not pre-specified |
| prog_21 | F1 | P-sq-rz | positive | x*x computed with mul.rz (truncation toward zero); every squared term is underestimated | + along w = K_R/||K_R|| (outputs scaled up) |
| prog_22 | F6 | N-copy | negative | verbatim copy of clean | none |
| prog_23 | F2S | clean | clean | none; outputs scaled by t = 2^-124, many subnormal | none pre-specified |
| prog_24 | F2 | clean | clean | none | none pre-specified |
| prog_25 | F3 | U-naive | unknown | variance by E[x^2]-E[x]^2 (cancellation) instead of chunked Welford merge | not pre-specified |
| prog_26 | F1 | P-fma-rm | positive | final affine computed by fma.rm (round toward -inf); residual of every element lies in (-ulp(y), 0] | + along w = -1/sqrt(n) (uniform downward shift); not along K_R direction |
| prog_27 | F3 | S-dm1 | semantic | variance divided by D-1 (spec: D) | K_R - f on rstd: negative, relative -1/(2D) ~ -1.2e-4; mean output unaffected |
| prog_28 | F4 | U-nofusion | unknown | identical source, launched with enable_fp_fusion=False | not pre-specified |
| prog_29 | F2 | N-copy | negative | verbatim copy of clean | none |
| prog_30 | F4 | clean | clean | none (default enable_fp_fusion=True) | none pre-specified |
| prog_31 | F5 | P-mul-rz | positive | each product deq*a computed with mul.rz: every term shrinks toward zero by a relative amount in [0, 2^-23) | + along w = -K_R/||K_R|| (E[residual] ~ -c*K_R with c ~ 4e-8) |
| prog_32 | F6 | P-mul-rz | positive | numerator products w*x computed with mul.rz: each term shrinks toward zero | + along w = -K_R/||K_R|| (numerator, hence y, shrinks toward zero by ~4e-8 rel in expectation) |
| prog_33 | F2 | S-swap | semantic | arguments swapped: computes (b*g(b))*a instead of (a*g(a))*b | K_R - f: not pre-specified |
| prog_34 | F5 | clean | clean | none | none pre-specified |
| prog_35 | F4 | S-sign | semantic | y2 computed as x2*c - x1*s (spec: x1*s + x2*c) | K_R - f: not pre-specified (x symmetric) |
| prog_36 | F4 | N-comm | negative | operands of the multiplies commuted (x2*s -> s*x2 etc.) | none |
| prog_37 | F4 | U-bf16 | unknown | x1, x2 rounded to bf16 before the rotation (precision downgrade) | not pre-specified (x symmetric) |

阳性的预期量级与定位节点见答案 JSON 的 `predicted_relative_magnitude` 与 `localization` 字段。

## 2. 勘误（评分前生效）

1. **prog_27（F3 S-dm1）**：答案写「K − K_R 只含舍入残差」是错的。把 D 改成 D−1 后，除以 4095 不再是 2 的幂，编译为 `div.full.f32`，K − K_R 有真实数值差异。阶段 1 对它的任何数值检出按「差异存在」处理。
2. **F3 干净写法（prog_08）与 F3 全部程序**：答案假设干净写法与规格 f 完全一致。分块合并的系数 512/(k+512)、k·512/(k+512) 在 k = 1024, 2048, 2560, 3072 处不是二进制有限小数，被编译器存成 float32 常数，程序声明语义与 f 相差约 1e-8 的相对量，传到 rstd 上约 1e-11、mean 上约 1e-11。阶段 2 在 prog_05、prog_08、prog_27 的 mean/rstd 列上检出的微小 K_R − f 为真实差异（常量折叠），不是误报。独立核实按 float32 常数重算后与工具一致。
3. **R4 的分类**：修订 v1.1 把 R4 写进对齐规则组是错的，R4 是事前固定方向，与 R1、R5 同组。只改分类。
4. **F4 的 t**：规格里的运行时标量指 kernel 实际接收的 float32 值。阶段 2 以 float32 的 t 算 f 为正式读数；按双精度算出的那组检出（prog_16、28、30、36、37 的 R2+、R3+）记为接口精度差异，不计。

## 3. 核实包内 8 个阳性的初步符号核对（两个 seed 的重算值，不是统计判定）

| 程序 | 机制 | 答案指定规则与符号 | seed 0 | seed 32 | 一致 |
|---|---|---|---|---|---|
| prog_21 | F1 x*x 用 mul.rz | R3 为负（输出等比放大） | -9.48e-06 | -1.22e-05 | 是 |
| prog_26 | F1 末步 fma.rm | R1 为正（整体向下偏移） | +1.25e-05 | +1.25e-05 | 是 |
| prog_06 | F2 末步 mul.rm | R1 为正 | +1.05e-05 | +1.05e-05 | 是 |
| prog_14 | F2S 末步 mul.rn.ftz | R2 为正（刷零即向零收缩） | +1.30e-36 | +1.30e-36 | 是 |
| prog_05 | F3 平方用 mul.rz | rstd 列 R1 为负（rstd 被高估） | -1.23e-08 | -7.46e-08 | 是 |
| prog_16 | F4 fma.rm / fma.rp | R4 为正 | +2.21e-06 | +2.21e-06 | 是 |
| prog_31 | F5 乘积用 mul.rz | R3 为正（向零收缩） | +3.69e-05 | +3.88e-05 | 是 |
| prog_32 | F6 分子乘积用 mul.rz | R3 为正（向零收缩） | +1.49e-07 | +1.64e-07 | 是 |

量级对照：prog_26 的 R1 ≈ 1.25e-5 对应每元素约 −0.5 ulp(|y|)，与答案一致；prog_21 的 R3 ≈ −1e-5 对应相对放大约 2.6e-8，答案预测约 2e-8；prog_31、prog_32 的 R3 对应相对收缩约 4e-8，答案预测约 4e-8。

## 4. 核实包内其他程序的观察

- prog_11（F1 干净）与 prog_12（F1 阴性对照）两个 seed 的全部投影值完全相同，与逐位相同一致。两者在 seed 32 上 R2、R3 的投影为 −3.3e-6、−4.1e-6，seed 0 上为 −2.2e-8、−7.5e-8：干净程序的残差在参照方向上不是零，这正是修订 v1.1 不再把干净程序当作必须未确认的原因。它们的 μ 是否确认，以你们两轮的统计为准，我已独立核实其数值。
- prog_27（F3 S-dm1）阶段 1 的 rstd R1 ≈ +6.8e-8、+5.6e-8：K 比 K_R 小，来源是 `div.full.f32` 除以 4095，与勘误 1 一致，也与你们此前关于硬件倒数的发现同类。
- prog_15（F2 div.full 门控）、prog_19（F6 div.full 输出）、prog_07（F1 分块求和）：均值未预设类，数值差异在两个 seed 上都存在。
- prog_01（F7）：K − K_R 在两个 seed 上 R1 ≈ −9e-7、−1.1e-6，即 KL 的实际值系统性大于声明语义值；它来自 tl.exp、tl.log 的近似下降，属于真实发现，按协议不计分，值得单独归因。

## 5. 正式计分需要的材料

阶段 1 两轮与阶段 2 的 37 × 5 判定矩阵（含 Holm 后判定、μ 区间、定位节点与区域大小），以完整 raw 链接给出。计分按修订 v1.1 加本文件勘误：阳性按 Δμ = μ(变体) − μ(同家族干净版本)；阴性按与伙伴逐位相同且判定一致；语义类按阶段 1 写明未检验、阶段 2 符号；干净版本的检出按「待核实的发现」处理，其中 F1 的两个已由本次核实确认数值无误。